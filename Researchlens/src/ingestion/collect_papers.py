import hashlib
import json
from src.config import BASE_DIR, PAPERS_DIR, CATALOG_PATH, PAPER_METADATA


def file_hash(path):
    # A HASH is a fingerprint calculated from a file's bytes.
    # SHA-256 lets us detect identical PDFs even when filenames differ.
    # It detects byte-identical copies, not different versions of a paper.
    digest = hashlib.sha256()
    with path.open("rb") as file:
        # Read 1 MB at a time instead of loading a large PDF into memory.
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_papers():
    # DATASET CREATION means collecting and organising our input documents.
    # mkdir creates the input folder if it does not exist yet.
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in PAPERS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not files:
        print(f"Add your PDFs to: {PAPERS_DIR}")
        return []  # Do not overwrite an existing catalogue with an empty one.

    # Dictionary keys are hashes: one record per unique file content.
    unique = {}
    for path in files:
        digest = file_hash(path)
        metadata = PAPER_METADATA.get(path.name, {})

        # PROVENANCE means recording where information came from.
        # Relative paths remain usable if the project folder is moved.
        source = {
            "path": path.relative_to(BASE_DIR).as_posix(),
            "source_url": metadata.get("source_url"),
        }
        if not source["source_url"]:
            print(f"Missing source URL: add {path.name} to PAPER_METADATA.")

        if digest in unique:
            # DEDUPLICATION avoids indexing identical content twice.
            # Preserve both source records; do not delete the user's PDFs.
            unique[digest]["sources"].append(source)
            print(f"Duplicate content: {path.name}")
            continue

        unique[digest] = {
            "paper_id": digest,
            "sha256": digest,
            "title": metadata.get("title"),
            "filename": path.name,
            "sources": [source],
        }

    papers = list(unique.values())

    # JSON stores lists and dictionaries in a readable, portable text format.
    # This catalogue records metadata; PDF text extraction comes in Step 2.
    # Re-running refreshes the catalogue from the current PDFs and metadata.
    CATALOG_PATH.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Found {len(files)} PDFs; catalogued {len(papers)} unique PDFs.")
    print(f"Catalogue saved to: {CATALOG_PATH}")
    return papers


# Run collection when this file is executed directly.
# Importing its functions from another file will not start collection.
if __name__ == "__main__":
    collect_papers()
