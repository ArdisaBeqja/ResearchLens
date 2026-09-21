from pathlib import Path

# PROBLEM FORMULATION means defining the task before writing the ML pipeline.
# These settings document the scope; they do not enforce answer quality yet.
PROJECT_SPEC = {
    "input": "English, text-based research PDFs on one topic",
    "output": "Answers and paper comparisons with page citations",
    "evaluation": "Evidence retrieval, answer correctness, citation support",
    "limitation": "Scores from different experiments may not be comparable",
}

# __file__ is this Python file. resolve() gives its absolute location.
# This makes paths work even if you run Python from another directory.
# config.py is in src/, so parents[1] points to the project root.
BASE_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = BASE_DIR / "data" / "papers"
CATALOG_PATH = BASE_DIR / "data" / "catalog.json"

# METADATA describes a document: its title, original URL, and other details.
# Add an entry for EACH PDF you download. Keys must match the PDF filenames.
# Missing metadata will be recorded as unknown, never invented.
PAPER_METADATA = {
    # "paper1.pdf": {
    #     "title": "Copy the actual paper title here",
    #     "source_url": "Paste the original paper URL here",
    # },
}

# PRETRAINED MODELS have already learned language patterns from other data.
# Reusing them is transfer learning; loading them does not train new weights.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
DATABASE_PATH = BASE_DIR / "data" / "chunks.sqlite"
PAGES_PATH = BASE_DIR / "data" / "pages.json"