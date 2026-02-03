---
title: MedGemma 1.5 Report Generator (Enhanced)
emoji:
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
---

# MedGemma 1.5 DICOM Report Generator — Enhanced

Aplicação web em Gradio que usa o modelo MedGemma 1.5 (Google) para gerar um rascunho de laudo radiológico estruturado a partir de estudos DICOM.

> **Aviso**: somente para pesquisa/educação. **Não** é para uso clínico nem diagnóstico médico.

## O que tem de diferente (vs. demo base)

- **Pipeline “chunked / map-reduce”** para aguentar estudos grandes (ex.: centenas de slices em TC) sem estourar VRAM:
  - Passo por chunks: imagens → **extração de achados em JSON**
  - Passo final: achados (JSON) → **laudo final (só texto)**
- **Refinamento opcional (só texto)** para melhorar estrutura/consistência (sem inventar achados).
- **Seleção de séries** + **exclusão de localizer/scout**.
- **Amostragem “smart”** (melhor de 3 por faixa) para pegar slices mais representativas com custo parecido.
- **Auto-fit para VRAM** (sugere tamanho de imagem e máx. de slices por série).
- **Reload do modelo** com **quantização** (4-bit / 8-bit) quando `bitsandbytes` estiver disponível.
- **Sanitização de PHI/PII (recomendado)**: tenta remover possíveis nomes/IDs/contatos do texto (entrada e saída) para reduzir risco de vazamento.

## Como usar

1. Envie um ZIP com as imagens DICOM.
2. Clique em **Processar & pré-visualizar** (preenche lista de séries, galeria e estimativa de memória).
3. Escolha o pipeline:
   - **chunked** para estudos grandes (recomendado)
   - **single** para estudos pequenos
   - adicione **refine** quando a qualidade variar
4. Clique em **Gerar laudo**.

## Dicas

- Se estiver batendo em VRAM:
  - use **Auto-fit para VRAM**
  - reduza **Máx. de slices por série**
  - reduza **Tamanho da imagem**
  - prefira o pipeline **chunked**

## Requisitos

- Python 3.10+ (recomendado 3.10–3.12)
- GPU CUDA recomendada
- Acesso (Hugging Face) ao modelo `google/medgemma-1.5-4b-it`

## Rodando localmente (rápido)

1. Crie um venv com Python 3.11/3.12.
2. Instale dependências.
3. Rode o `app.py`.

Observação: baixar/carregar o modelo pode demorar e consumir bastante RAM/VRAM.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# opcional: evita baixar o modelo ao iniciar
export AUTOLOAD_MODEL=0

python app.py
```

## Variáveis de ambiente úteis

- `HF_TOKEN`: token do Hugging Face (necessário para baixar o modelo, se for privado/restrito).
- `MODEL_ID`: sobrescreve o ID do modelo (padrão: `google/medgemma-1.5-4b-it`).
- `AUTOLOAD_MODEL`: `1` para pré-carregar o modelo ao iniciar o app (padrão em Spaces); `0` para carregar só quando for gerar.
