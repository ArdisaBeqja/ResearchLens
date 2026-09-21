"""Step 5: transformers, representation learning, embeddings and transfer learning.
A pretrained transformer maps each passage to a vector (a list of numbers).
The learned representation captures aspects of meaning; similarity is not truth.
This module performs inference, not model training.
"""
from src.config import EMBED_MODEL


def build_vector_index(chunks, model_name=EMBED_MODEL):
    import faiss
    from sentence_transformers import SentenceTransformer

    if not chunks:
        raise ValueError("No passages to embed")
    # Loading a pretrained encoder reuses its previously learned parameters.
    encoder = SentenceTransformer(model_name)
    lengths = [len(encoder.tokenizer.encode(c["text"])) for c in chunks]
    if max(lengths) > encoder.max_seq_length:
        raise ValueError("Passages exceed encoder context; rechunk with its tokenizer")

    # Each input passage becomes one row in the embedding matrix.
    # Normalize to unit length so inner product corresponds to cosine similarity.
    vectors = encoder.encode([c["text"] for c in chunks], normalize_embeddings=True)

    # FAISS stores and searches vectors. IndexFlatIP is exact inner-product search.
    # The index's vector row positions match the positions in the chunks list.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return encoder, index
