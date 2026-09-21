"""Download only the public, pinned encoder assets; no dataset is transmitted."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if __name__ == "__main__":
    snapshot_download(config["model_id"], revision=config["revision"], local_dir=root / "local/model",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "vocab.json", "merges.txt"])
