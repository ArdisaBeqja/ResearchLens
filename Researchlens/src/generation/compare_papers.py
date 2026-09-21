"""Steps 9–10: information extraction and comparative summarization.
Each field contains validated claim/citation JSON, evidence and generation stats.
The interface turns these records into a table with identical columns per paper.
Never treat scores from different datasets/settings as directly comparable.
"""
def compare_papers(retriever, llm, paper_ids, mode="hybrid"):
    """Retrieve each field per paper; build a comparison without a global top-k bias."""
    fields = {
        "method": "What model or method do the authors propose or use?",
        "dataset": "What datasets and sample sizes are used?",
        "evaluation": "What metrics and train/test or speaker splits are used?",
        "results": "What results are reported? Include metric, dataset and evaluation setting.",
        "limitations": "What limitations do the authors explicitly report?",
    }
    records = []
    # MULTI-DOCUMENT SYNTHESIS: collect comparable fields for EVERY paper.
    # Separate retrieval prevents one paper dominating a global top-k result.
    for pid in paper_ids:
        record = {"paper_id": pid, "fields": {}}
        # INFORMATION EXTRACTION: ask one focused question for each field.
        for field, question in fields.items():
            evidence = retriever.search(question, mode=mode, k=5, paper_id=pid)
            answer, sources, stats = llm.answer(question, evidence)
            record["fields"][field] = {"answer": answer, "sources": sources, "stats": stats}
        records.append(record)
    return records
