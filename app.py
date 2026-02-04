"""
Aplicação Gradio para rascunho de laudos a partir de estudos DICOM (MedGemma 1.5).

Aviso: somente para pesquisa/educação. Não usar para decisão clínica.
"""

# IMPORTANTE: no Hugging Face Spaces, importe `spaces` ANTES de torch/transformers.
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

import os
import tempfile
import traceback
from typing import Tuple, List

import gradio as gr
import torch

# Desativa TF32 para evitar erros do tipo CUBLAS_STATUS_INVALID_VALUE em alguns formatos/GPUs.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from dicom_processor import process_dicom_study
from phi_sanitizer import mask_patient_id, sanitize_phi_text

# ============================================================================
# Carregamento do modelo (precisa estar no nível do módulo para ZeroGPU/Spaces)
# ============================================================================

def _select_model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            if fn():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


print("Carregando o modelo MedGemma na inicialização...")
MODEL_ID = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_DTYPE = _select_model_dtype()

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=MODEL_DTYPE,
    token=HF_TOKEN,
)
model.generation_config.do_sample = True
print(f"Modelo carregado: {MODEL_ID}")
print(f"Device do modelo: {model.device}")
print(f"DType do modelo: {next(model.parameters()).dtype}")

# Cache de dados processados (evita reprocessar em todo clique)
cached_data = {
    "zip_bytes": None,
    "images": None,
    "modality": None,
    "study_info": None
}


def process_dicom_file(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool
) -> Tuple[str, str, List[Image.Image]]:
    """Processa o ZIP DICOM e devolve imagens de pré-visualização."""
    global cached_data

    try:
        if file_path is None:
            return "Nenhum arquivo enviado.", "", []

        with open(file_path, 'rb') as f:
            zip_bytes = f.read()

        # Use per-series sampling if max_slices_per_series > 0
        slices_per_series = max_slices_per_series if max_slices_per_series > 0 else None

        # Use auto window if checkbox is checked
        wc = None if use_auto_window else window_center
        ww = None if use_auto_window else window_width

        modality, images, study_info = process_dicom_study(
            zip_bytes,
            max_slices_per_series=slices_per_series,
            image_size=image_size,
            window_center=wc,
            window_width=ww
        )

        # Cache for later use in report generation
        cached_data["zip_bytes"] = zip_bytes
        cached_data["images"] = images
        cached_data["modality"] = modality
        cached_data["study_info"] = study_info

        max_per_series = study_info.get('MaxSlicesPerSeries', None)
        sampling_info = (
            f"Máx. de slices por série: {max_per_series}"
            if max_per_series
            else "Amostragem: global (todas as séries juntas)"
        )

        # Get window info
        default_wc = study_info.get('DefaultWindowCenter', 'N/A')
        default_ww = study_info.get('DefaultWindowWidth', 'N/A')
        window_info = (
            f"Janela: automática (WC={default_wc}, WW={default_ww})"
            if use_auto_window
            else f"Janela: manual (WC={window_center}, WW={window_width})"
        )

        # Estimate VRAM usage based on actual image size
        num_images = study_info.get('ProcessedImages', 0)
        img_size = study_info.get('ImageSize', 896)
        model_vram_gb = 8.0
        base_per_image_mb = 50
        size_factor = (img_size / 896) ** 2
        per_image_vram_mb = base_per_image_mb * size_factor
        images_vram_gb = (num_images * per_image_vram_mb) / 1024
        total_vram_gb = model_vram_gb + images_vram_gb

        patient_id_masked = mask_patient_id(str(study_info.get("PatientID", "")))

        info_text = f"""Informações do estudo (exibição segura / PHI-safe):

Modalidade: {study_info['Modality']}
Descrição do estudo: {study_info['StudyDescription']}
Data do estudo: {study_info['StudyDate']}
ID do paciente: {patient_id_masked}

Qtde. de séries: {study_info.get('SeriesCount', 'N/A')}
Total de slices (originais): {study_info.get('TotalOriginalSlices', 'N/A')}
{sampling_info}
Imagens processadas: {num_images}
Tamanho da imagem: {img_size}x{img_size}
{window_info}

--- Estimativa de VRAM ---
Modelo: ~{model_vram_gb:.1f} GB
Imagens ({num_images} x {img_size}x{img_size}): ~{images_vram_gb:.1f} GB
Total estimado: ~{total_vram_gb:.1f} GB
"""

        status = f"Processado: {len(images)} imagens ({study_info['Modality']})"

        return status, info_text, images

    except Exception as e:
        error_msg = f"Erro ao processar DICOM: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, "", []


def _generate_report_impl(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    sanitize_phi: bool,
) -> str:
    """Gera um rascunho de laudo com MedGemma."""
    global cached_data

    try:
        if file_path is None:
            return "Envie um ZIP DICOM antes de gerar o laudo."

        # Check if we can use cached images
        use_cache = (
            cached_data["images"] is not None and
            cached_data["zip_bytes"] is not None
        )

        if use_cache:
            images = cached_data["images"]
            modality = cached_data["modality"]
        else:
            with open(file_path, 'rb') as f:
                zip_bytes = f.read()

            slices_per_series = max_slices_per_series if max_slices_per_series > 0 else None
            wc = None if use_auto_window else window_center
            ww = None if use_auto_window else window_width

            modality, images, study_info = process_dicom_study(
                zip_bytes,
                max_slices_per_series=slices_per_series,
                image_size=image_size,
                window_center=wc,
                window_width=ww
            )

        print(f"Processando {len(images)} imagens...")

        # Use custom prompt or default
        if not prompt.strip():
            prompt = (
                f"Você é um médico radiologista. Gere um laudo estruturado para o exame ({modality}) "
                "com as seções: Técnica, Achados e Impressão."
            )
        if sanitize_phi:
            prompt, _ = sanitize_phi_text(prompt)

        # Salva imagens em arquivos temporários e monta o conteúdo no formato esperado pelo model/processor.
        temp_files = []
        content = []
        for i, img in enumerate(images):
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(temp_file.name, format="PNG")
            temp_files.append(temp_file.name)
            content.append({"type": "image", "url": temp_file.name})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device=model.device)
        model_dtype = next(model.parameters()).dtype
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=model_dtype)

        input_len = inputs["input_ids"].shape[-1]
        print(f"Tamanho da sequência de entrada: {input_len}")

        # Generate report
        with torch.inference_mode():
            if do_sample and temperature > 0:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            generation = generation[0][input_len:]

        report = processor.decode(generation, skip_special_tokens=True)
        if sanitize_phi:
            report, _ = sanitize_phi_text(report)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        return report

    except Exception as e:
        error_msg = f"Erro ao gerar o laudo: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        # Limpa temporários em caso de erro
        if 'temp_files' in locals():
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        return error_msg


# Aplica @spaces.GPU quando estiver rodando no Hugging Face Spaces
if SPACES_AVAILABLE:
    @spaces.GPU(duration=120)
    def generate_report(
        file_path: str,
        max_slices_per_series: int,
        image_size: int,
        window_center: float,
        window_width: float,
        use_auto_window: bool,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        sanitize_phi: bool,
    ) -> str:
        """Gera laudo com MedGemma (acelerado por GPU no HF Spaces)."""
        return _generate_report_impl(
            file_path, max_slices_per_series, image_size,
            window_center, window_width, use_auto_window,
            prompt, max_tokens, temperature, top_p, top_k, do_sample, sanitize_phi
        )
else:
    def generate_report(
        file_path: str,
        max_slices_per_series: int,
        image_size: int,
        window_center: float,
        window_width: float,
        use_auto_window: bool,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        sanitize_phi: bool,
    ) -> str:
        """Gera laudo com MedGemma."""
        return _generate_report_impl(
            file_path, max_slices_per_series, image_size,
            window_center, window_width, use_auto_window,
            prompt, max_tokens, temperature, top_p, top_k, do_sample, sanitize_phi
        )


def create_interface():
    """Cria a interface do Gradio."""

    with gr.Blocks(title="Gerador de Laudo DICOM (MedGemma 1.5)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Gerador de Laudo DICOM (MedGemma 1.5)")
        gr.Markdown("Envie um arquivo ZIP com imagens DICOM para gerar um rascunho de laudo estruturado.")

        with gr.Row():
            # Left column: Upload and settings
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Enviar ZIP DICOM",
                    file_types=[".zip"],
                    type="filepath"
                )

                with gr.Accordion("Processamento de imagens", open=True):
                    max_slices_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Máx. de slices por série",
                        info="0 = usa todas as slices (global). Reduza para economizar VRAM."
                    )

                    image_size_slider = gr.Slider(
                        minimum=224,
                        maximum=1024,
                        value=512,
                        step=32,
                        label="Tamanho da imagem",
                        info="Menor = menos VRAM, menos qualidade"
                    )

                    gr.Markdown("**Windowing (TC / raio-X)**")
                    use_auto_window = gr.Checkbox(
                        label="Usar janela automática (metadados DICOM)",
                        value=True
                    )
                    with gr.Row():
                        window_center_slider = gr.Slider(
                            minimum=-1000,
                            maximum=3000,
                            value=40,
                            step=10,
                            label="Window Center (WC)",
                            info="Ex.: cérebro=40, pulmão=-600, osso=400"
                        )
                        window_width_slider = gr.Slider(
                            minimum=1,
                            maximum=4000,
                            value=400,
                            step=10,
                            label="Window Width (WW)",
                            info="Ex.: cérebro=80, pulmão=1500, osso=1800"
                        )

                process_btn = gr.Button("Processar & pré-visualizar", variant="primary", size="lg")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                study_info_box = gr.Textbox(
                    label="Informações do estudo & estimativa de VRAM",
                    interactive=False,
                    lines=14
                )

            # Middle column: Image preview
            with gr.Column(scale=1):
                gr.Markdown("### Pré-visualização")
                gr.Markdown("*Preview das slices amostradas que serão enviadas ao modelo*")

                image_gallery = gr.Gallery(
                    label="Slices amostradas",
                    show_label=False,
                    columns=4,
                    rows=3,
                    height=400,
                    object_fit="contain",
                    preview=True
                )

            # Right column: Generation settings and output
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="Você é um médico radiologista. Gere um laudo estruturado com: Técnica, Achados e Impressão.",
                    info="Personalize o prompt. Deixe em branco para usar o padrão."
                )
                sanitize_phi_checkbox = gr.Checkbox(
                    label="Sanitizador de PHI/PII (recomendado)",
                    value=True,
                    info="Redige possíveis identificadores em texto livre (prompt + saída). Heurístico.",
                )

                with gr.Accordion("Configurações do modelo", open=False):
                    with gr.Row():
                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=350,
                            step=10,
                            label="Máx. tokens"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperatura"
                        )
                    with gr.Row():
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-k"
                        )
                    do_sample_checkbox = gr.Checkbox(
                        label="Ativar sampling",
                        value=True,
                        info="Desmarque para saída determinística"
                    )

                generate_btn = gr.Button("Gerar laudo", variant="primary", size="lg")

                report_output = gr.Textbox(
                    label="Laudo gerado",
                    interactive=False,
                    lines=18,
                    placeholder="O laudo vai aparecer aqui..."
                )

        # Presets de janela
        with gr.Accordion("Presets de janela (clique para aplicar)", open=False):
            gr.Markdown("**Presets de TC:**")
            with gr.Row():
                brain_btn = gr.Button("Cérebro (40/80)", size="sm")
                subdural_btn = gr.Button("Subdural (75/215)", size="sm")
                stroke_btn = gr.Button("Stroke (32/8)", size="sm")
                lung_btn = gr.Button("Pulmão (-600/1500)", size="sm")
                mediastinum_btn = gr.Button("Mediastinum (50/350)", size="sm")
                bone_btn = gr.Button("Bone (400/1800)", size="sm")
                abdomen_btn = gr.Button("Abdome (40/400)", size="sm")
                liver_btn = gr.Button("Liver (60/150)", size="sm")

        # Event handlers for presets
        brain_btn.click(lambda: (40, 80, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        subdural_btn.click(lambda: (75, 215, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        stroke_btn.click(lambda: (32, 8, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        lung_btn.click(lambda: (-600, 1500, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        mediastinum_btn.click(lambda: (50, 350, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        bone_btn.click(lambda: (400, 1800, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        abdomen_btn.click(lambda: (40, 400, False), outputs=[window_center_slider, window_width_slider, use_auto_window])
        liver_btn.click(lambda: (60, 150, False), outputs=[window_center_slider, window_width_slider, use_auto_window])

        # Main event handlers
        process_btn.click(
            fn=process_dicom_file,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window
            ],
            outputs=[status_output, study_info_box, image_gallery]
        )

        generate_btn.click(
            fn=generate_report,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window,
                prompt_input,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                do_sample_checkbox,
                sanitize_phi_checkbox,
            ],
            outputs=[report_output]
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Modalidades suportadas:** CT, MR, CR, DX | "
            "**Dica:** use menos slices e um tamanho de imagem menor para reduzir o consumo de VRAM"
        )

    return demo


def main():
    """Ponto de entrada."""
    print("Iniciando o Gerador de Laudo DICOM (MedGemma 1.5)...")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
