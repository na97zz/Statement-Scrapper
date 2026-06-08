from __future__ import annotations

from pathlib import Path


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return "\n\n".join(page.get_text("text") for page in doc).strip()
    except Exception:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts).strip()
