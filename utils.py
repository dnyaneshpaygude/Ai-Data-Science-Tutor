"""
Helper utilities: environment loading, validation, sample PDF creation, and input checks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


def load_environment() -> None:
    """Load variables from .env in the project root."""
    load_dotenv(PROJECT_ROOT / ".env")


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key from the environment, or None if missing."""
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        return None
    stripped = key.strip()
    return stripped or None


def validate_api_key() -> tuple[bool, str]:
    """
    Check that OPENAI_API_KEY looks configured.
    Returns (ok, message_for_user).
    """
    key = get_openai_api_key()
    if not key:
        return False, (
            "Missing `OPENAI_API_KEY`. Add it to your `.env` file in the project root "
            "(see README)."
        )
    lowered = key.lower()
    if lowered.startswith("your_") or "your-openai-api-key" in lowered or key == "sk-your-key-here":
        return False, "Please replace the placeholder in `.env` with your real OpenAI API key."
    return True, ""


def sanitize_filename(name: str) -> str:
    """Reduce path traversal / odd characters from uploaded file names."""
    base = Path(name).name
    base = re.sub(r"[^\w.\-]", "_", base)
    return base or "uploaded.pdf"


def validate_user_question(text: str) -> tuple[bool, str]:
    """
    Basic validation for chat input.
    Returns (ok, error_message). error_message is empty when ok is True.
    """
    if text is None:
        return False, "Please enter a question."
    cleaned = text.strip()
    if not cleaned:
        return False, "Please enter a non-empty question."
    if len(cleaned) > 4000:
        return False, "Your question is too long. Please shorten it (max 4000 characters)."
    return True, ""


def list_pdf_paths(folder: Path | None = None) -> list[Path]:
    """Return sorted PDF paths under `folder` (defaults to data/)."""
    root = folder or DATA_DIR
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.pdf") if p.is_file())


def ensure_placeholder_pdfs() -> int:
    """
    If there are no PDFs in data/, create a small sample learning PDF so the app can boot.
    Returns the number of PDFs created (0 or 1).
    """
    if list_pdf_paths():
        return 0
    try:
        from fpdf import FPDF
    except ImportError:
        # Optional dependency; README documents adding PDFs manually.
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "sample_data_science_basics.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(14)
    pdf.set_right_margin(14)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    lines = [
        "Sample Document: Data Science Basics (Placeholder)",
        "",
        "Pandas: Pandas is a Python library for tabular data. You can load CSV files with "
        "pd.read_csv(). Example: df = pd.read_csv('sales.csv'); print(df.head()).",
        "Real-world use: analysts use Pandas to clean retail sales data before forecasting.",
        "",
        "Machine Learning: ML learns patterns from data to make predictions. "
        "Example: train a model to predict house prices from features like size and location.",
        "Real-world use: fraud detection systems score transactions using ML models.",
        "",
        "SQL: SQL queries relational databases. Example: SELECT name, salary FROM employees "
        "WHERE department = 'Engineering';",
        "Real-world use: product dashboards pull metrics from a warehouse using SQL.",
    ]
    for line in lines:
        pdf.multi_cell(usable_w, 8, line)
    pdf.output(str(out))
    return 1
