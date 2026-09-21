"""Steps 6–7: vector search, semantic similarity, dense and hybrid retrieval.
Dense retrieval finds passages by embedding similarity. Hybrid retrieval fuses
lexical and dense rankings. Neural reranking scores query/passage pairs together.
"""
from collections import Counter
from src.config import EMBED_MODEL, RERANK_MODEL
from src.retrieval.keyword_search import BM25
from src.retrieval.create_embeddings import build_vector_index

def reciprocal_rank_fusion(*rankings, k=20):
    """Combine rank positions rather than incompatible BM25/cosine scores."""
    scores, records = Counter(), {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, 1):
            records[row["id"]] = row
            scores[row["id"]] += 1 / (60 + rank)
    return [{**records[cid], "score": score} for cid, score in scores.most_common(k)]

class Retriever:
    def __init__(self, chunks, embedding_model=EMBED_MODEL):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self.embedding_model = embedding_model
        self.encoder = self.index = self.reranker = None

    def prepare_dense(self):
        # Lazy initialization avoids loading a neural model for BM25-only use.
        if self.index is None:
            self.encoder, self.index = build_vector_index(self.chunks, self.embedding_model)

    def dense(self, question, k=20, paper_id=None):
        self.prepare_dense()
        # Encode questions with the SAME model used to encode passages.
        query = self.encoder.encode([question], normalize_embeddings=True).astype("float32")
        # Search all rows before paper filtering: sufficient for this small corpus.
        scores, indexes = self.index.search(query, len(self.chunks))
        rows = [{**self.chunks[int(i)], "score": float(s)} for s, i in zip(scores[0], indexes[0])
                if i >= 0 and (paper_id is None or self.chunks[int(i)]["paper_id"] == paper_id)]
        return rows[:k]

    def search(self, question, mode="hybrid", k=5, paper_id=None):
        if mode not in {"bm25", "dense", "hybrid", "rerank"}:
            raise ValueError("Unknown retrieval mode")
        n = max(20, k)
        lexical = self.bm25.search(question, n, paper_id) if mode != "dense" else []
        if mode == "bm25":
            return lexical[:k]
        dense = self.dense(question, n, paper_id)
        if mode == "dense":
            return dense[:k]
        # HYBRID RETRIEVAL combines two complementary search methods.
        candidates = reciprocal_rank_fusion(lexical, dense, k=n)
        if mode == "rerank" and candidates:
            from sentence_transformers import CrossEncoder
            if self.reranker is None:
                self.reranker = CrossEncoder(RERANK_MODEL)
            # A CROSS-ENCODER reads the question and passage together.
            # It reranks the candidate list; scores are not factual confidence.
            scores = self.reranker.predict([(question, c["text"]) for c in candidates])
            candidates = sorted([{**c, "score": float(s)} for c, s in zip(candidates, scores)],
                                key=lambda c: c["score"], reverse=True)
        return candidates[:k]
