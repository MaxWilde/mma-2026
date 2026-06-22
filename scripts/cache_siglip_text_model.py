#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SigLIP/SigLIP2 text-tower-only checkpoint for fast query-time retrieval.")
    parser.add_argument("--model", default="/scratch-shared/group_h/models/siglip2-so400m-patch16-512")
    parser.add_argument("--output", default="/scratch-shared/group_h/models/siglip2-so400m-patch16-512-text")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    model_type = local_model_type(args.model)
    local_files_only = not args.allow_download
    try:
        from transformers import AutoProcessor, Siglip2TextModel, SiglipTextModel
    except ImportError as exc:
        raise SystemExit("Missing transformers dependency.") from exc

    if model_type == "siglip2":
        model = Siglip2TextModel.from_pretrained(args.model, local_files_only=local_files_only)
    elif model_type == "siglip":
        model = SiglipTextModel.from_pretrained(args.model, local_files_only=local_files_only)
    else:
        raise SystemExit(f"Unsupported model_type for text export: {model_type!r}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=local_files_only)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    print(f"Saved text tower: {output}")
    print(f"Source model: {args.model}")
    print(f"Model type: {model_type}")


def local_model_type(model_name: str) -> str | None:
    config_path = Path(model_name).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("model_type")


if __name__ == "__main__":
    main()
