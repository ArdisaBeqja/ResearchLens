**ResearchLens**
ResearchLens is a Python application for exploring and comparing research papers. Upload PDFs, ask questions about their content, and inspect the passages behind each answer.
The application uses Retrieval-Augmented Generation (RAG): it retrieves relevant passages before asking a local language model to produce an answer with source citations.
Features
- Upload multiple English, text-based PDF papers.
- Detect duplicate papers using SHA-256 content hashes.
- Extract text by page and split it into overlapping token-based passages.
- Search papers with BM25, dense retrieval, hybrid search, or cross-encoder reranking.
- Generate answers with a local Ollama model, linking each claim to source passages.
- Inspect retrieved text, paper labels, PDF page numbers, and passage IDs.
- Compare selected papers across methods, datasets, evaluation settings, results, and author-reported limitations.
- Download extracted passages as a SQLite database and comparisons as JSON.
- Benchmark retrieval methods and optionally fine-tune the embedding model.
Tech Stack
- Python — application and processing logic
- Streamlit — web interface
- PyMuPDF4LLM — PDF text extraction
- Sentence Transformers / Hugging Face Transformers — embeddings, tokenization, and reranking
- FAISS — vector similarity search
- SQLite — extracted passage storage
- Ollama — local language model inference
- Docker — optional container deployment
Default Models
Purpose	Model
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Reranking	cross-encoder/ms-marco-MiniLM-L-6-v2
Answer generation	qwen2.5:3b through Ollama
Generator tokenizer	Qwen/Qwen2.5-3B-Instruct


