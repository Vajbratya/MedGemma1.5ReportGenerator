"""
Small, heuristic PHI/PII sanitizer.

Goal: reduce accidental leakage of identifiers (names, IDs, contacts) in free text.
This is NOT a medical/legal guarantee; it's a best-effort safety layer.
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
    r"(?im)^\s*(?:"
    r"patient\s*name|name|patient\s*id|patient\s*identifier|mrn|dob|ssn|accession(?:\s*number)?|"
    r"nome|paciente|id\s*do\s*paciente|prontu[aá]rio|cpf|rg|cns|data\s*de\s*nascimento|"
    r"nombre|paciente|id\s*del\s*paciente|historia\s*cl[ií]nica|dni"
    r")\s*[:\-]\s*(.+?)\s*$"
)


def mask_patient_id(patient_id: str) -> str:
    """Mask an ID for UI display (keeps last 4 chars when possible)."""
    if not patient_id:
        return "REDACTED"
    s = str(patient_id).strip()
    if len(s) <= 4:
        return "REDACTED"
    return ("*" * (len(s) - 4)) + s[-4:]


def sanitize_phi_text(text: str) -> Tuple[str, List[str]]:
    """
    Best-effort PHI/PII sanitizer for free text.
    Returns (sanitized_text, warnings).
    """
    if not text:
        return text, []

    warnings: List[str] = []
    s = str(text)

    def _redact_label_line(m: re.Match) -> str:
        prefix = m.group(0).split(":", 1)[0]
        return f"{prefix}: [REDACTED]"

    before = s
    s = _PHI_LABEL_RE.sub(_redact_label_line, s)
    if s != before:
        warnings.append("Sanitizer: redacted likely patient-identifying fields (name/ID).")

    if _EMAIL_RE.search(s):
        s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
        warnings.append("Sanitizer: redacted a possible email.")

    if _PHONE_RE.search(s):
        s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
        warnings.append("Sanitizer: redacted a possible phone number.")

    if _UID_LIKE_RE.search(s):
        s = _UID_LIKE_RE.sub("[REDACTED_UID]", s)
        warnings.append("Sanitizer: redacted a UID-like identifier.")

    if _LONG_DIGITS_RE.search(s):
        s = _LONG_DIGITS_RE.sub("[REDACTED_ID]", s)
        warnings.append("Sanitizer: redacted a long numeric identifier.")

    warnings = list(dict.fromkeys(warnings))
    return s.strip(), warnings

