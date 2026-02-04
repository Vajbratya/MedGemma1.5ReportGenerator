"""
Sanitizador heurístico de PHI/PII.

Objetivo: reduzir vazamento acidental de identificadores (nome, ID, contatos) em texto livre.
Isto NÃO é garantia médica/jurídica; é uma camada de segurança “best-effort”.
"""

from __future__ import annotations

import re
from typing import List, Tuple


_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\+?\d{1,3}\s*)?(?:\(?\d{2,3}\)?\s*)?(?:\d[\s\-]*){7,}\d(?:(?=\s)|$)"
)
_UID_LIKE_RE = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")

_PHI_LABEL_RE = re.compile(
    r"(?im)^\s*((?:"
    r"patient\s*name|name|patient\s*id|patient\s*identifier|mrn|dob|ssn|accession(?:\s*number)?|"
    r"nome|paciente|id\s*do\s*paciente|prontu[aá]rio|cpf|rg|cns|data\s*de\s*nascimento|"
    r"nombre|paciente|id\s*del\s*paciente|historia\s*cl[ií]nica|dni"
    r"))\s*[:\-]\s*(.+?)\s*$"
)


def mask_patient_id(patient_id: str) -> str:
    """Mascara um ID para exibição no UI (mantém os últimos 4 caracteres quando possível)."""
    if not patient_id:
        return "REDACTED"
    s = str(patient_id).strip()
    if len(s) <= 4:
        return "REDACTED"
    return ("*" * (len(s) - 4)) + s[-4:]


def sanitize_phi_text(text: str) -> Tuple[str, List[str]]:
    """
    Sanitizador “best-effort” de PHI/PII para texto livre.
    Retorna (texto_sanitizado, avisos).
    """
    if not text:
        return text, []

    warnings: List[str] = []
    s = str(text)

    def _redact_label_line(m: re.Match) -> str:
        # IMPORTANTE: não reutilizar m.group(0), porque pode conter o valor (ex.: quando o separador é "-").
        label = str(m.group(1)).strip()
        return f"{label}: [REDACTED]"

    before = s
    s = _PHI_LABEL_RE.sub(_redact_label_line, s)
    if s != before:
        warnings.append("Sanitização: removi valores em linhas que parecem conter dados do paciente (ex.: nome/ID).")

    if _EMAIL_RE.search(s):
        s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
        warnings.append("Sanitização: removi possível e-mail.")

    if _PHONE_RE.search(s):
        s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
        warnings.append("Sanitização: removi possível telefone.")

    if _UID_LIKE_RE.search(s):
        s = _UID_LIKE_RE.sub("[REDACTED_UID]", s)
        warnings.append("Sanitização: removi possível UID/identificador técnico.")

    if _LONG_DIGITS_RE.search(s):
        s = _LONG_DIGITS_RE.sub("[REDACTED_ID]", s)
        warnings.append("Sanitização: removi sequência numérica longa (possível ID).")

    warnings = list(dict.fromkeys(warnings))
    return s.strip(), warnings
