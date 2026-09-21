"""Step 4: information retrieval, lexical search and BM25.
Information retrieval finds evidence rather than writing an answer.
Lexical search matches terms. BM25 weights rare terms and accounts for passage
length; it is a classical ranking method, not a trained neural network.
"""
import math
import re
from collections import Counter

def lexical_tokens(text):
    """BM25 tokens; NOT the neural model's subword tokenizer."""
    return re.findall(r"\w+(?:[-.]\w+)*", text.lower())

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        if not chunks:
            raise ValueError("No extractable chunks. Check the PDFs.")
        self.chunks, self.k1, self.b = chunks, k1, b
        # TF counts each term in each passage; DF counts passages containing it.
        self.tf = [Counter(lexical_tokens(c["text"])) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.lengths) / len(chunks) or 1
        self.df = Counter(term for terms in self.tf for term in terms)

    def search(self, question, k=20, paper_id=None):
        results = []
        for i, tf in enumerate(self.tf):
            c = self.chunks[i]
            if paper_id and c["paper_id"] != paper_id:
                continue
            score = 0.0
            for term in set(lexical_tokens(question)):
                f = tf[term]
                if f:
                    n = len(self.chunks)
                    # Rare terms receive more weight than common ones.
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    # Length normalization avoids favouring long passages.
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    score += idf * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                results.append({**c, "score": score})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:k]
