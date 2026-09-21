"""Record the environment and dataset fingerprints with experiment settings."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

def record_run(database, settings, output="run_manifest.json"):
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    # REPRODUCIBILITY: record what data, packages and settings produced the result.
    # Add exact model revisions, prompt, hardware and seed to settings.
    manifest = {"python": platform.python_version(), "packages": packages,
                "dataset_sha256": hashlib.sha256(Path(database).read_bytes()).hexdigest(),
                "settings": settings}
    Path(output).write_text(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    # Snapshot installed versions AFTER successful setup, do not invent a tested lockfile.
    Path("requirements.lock.txt").write_bytes(subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"]))
    print("Saved installed versions to requirements.lock.txt")


def log_event(event, output="events.jsonl"):
    # MLOPS FOUNDATIONS: append experiment statistics for later inspection.
    # Each JSONL line is one event. This is not a production monitoring service.
    with Path(output).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")