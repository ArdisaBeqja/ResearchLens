"""Step 2: document processing and NLP preprocessing.
PDF extraction converts visual documents into machine-readable text.
NLP preprocessing cleans extraction noise without discarding scientific details.
This starter extracts page-based Markdown, not a complete table/figure model.
"""
def clean_text(text):
    """Conservative preprocessing: retain case, punctuation, units and newlines."""
    return text.replace("\x00", "").replace("\r\n", "\n").strip()

def extract_pages(paper):
    import pymupdf4llm
    # page_chunks keeps PDF page boundaries, needed for citations.
    pages = pymupdf4llm.to_markdown(paper["path"], page_chunks=True)
    # enumerate(..., 1) uses PDF page positions starting at 1.
    # These may differ from the page labels printed inside the paper.
    return [{**paper, "page": i, "text": clean_text(page["text"])}
            for i, page in enumerate(pages, 1)]
