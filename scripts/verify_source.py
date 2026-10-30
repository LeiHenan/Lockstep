"""Verify SHA-256 hashes of the public release sources, without importing them."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/source-manifest.json").read_text())
    errors = []
    for item in manifest["copied_files"]:
        path = root / item["release"]
        if not path.is_file():
            errors.append(f"missing: {item['release']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"modified: {item['release']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['copied_files'])} release files.")


if __name__ == "__main__":
    main()

