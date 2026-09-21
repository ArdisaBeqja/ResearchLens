"""Step 3: text chunking, tokenization and context management.
Chunks are small passages we can retrieve independently. This first version
respects page boundaries; section-aware chunking is a later improvement.
The embedding tokenizer creates chunks. The GENERATOR tokenizer budgets answers.
"""
def chunk_pages(pages, tokenizer, max_tokens=220, overlap=30):
    """Page-bounded token windows; advanced section-aware splitting is future work."""
    if not 0 <= overlap < max_tokens:
        raise ValueError("Require 0 <= overlap < max_tokens")
    chunks = []
    for page in pages:
        # TOKENIZATION converts text into the embedding model's subword IDs.
        ids = tokenizer.encode(page["text"], add_special_tokens=False)
        # Overlap repeats some tokens so boundary context is not entirely lost.
        for start in range(0, len(ids), max_tokens-overlap):
            text = tokenizer.decode(ids[start:start+max_tokens], skip_special_tokens=True)
            # Stable IDs combine the paper, page and token offset.
            chunks.append({"id": f'{page["paper_id"]}-p{page["page"]}-t{start}',
                           "paper_id": page["paper_id"], "title": page["title"],
                           "page": page["page"], "text": text})
            if start + max_tokens >= len(ids):
                break
    return chunks

def select_context(rows, token_count, budget=2200):
    """Budget evidence with the generator's tokenizer, including source labels."""
    selected, used = [], 0
    for row in rows:
        block = f'[{row["id"]}] {row["title"]}, PDF page {row["page"]}\n{row["text"]}'
        cost = token_count(block)
        if used + cost <= budget:
            selected.append({**row, "block": block})
            used += cost
    return selected
