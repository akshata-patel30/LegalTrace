"""Verification script for Pipeline A (Step 1: Document Intake) handoff contract."""

import hashlib
import json
from pathlib import Path
import sys

def verify_intake_contract(store_dir: Path) -> bool:
    manifest_path = store_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"FAIL: Manifest not found at {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Verifying {len(records)} records in manifest...")
    valid_count = 0
    errors = []

    for i, rec in enumerate(records, 1):
        doc_id = rec.get("doc_id")
        if not doc_id:
            errors.append(f"Record #{i}: Missing doc_id")
            continue

        full_text_path = store_dir / rec.get("full_text_path", "")
        metadata_path = store_dir / rec.get("metadata_path", "")

        if not full_text_path.exists():
            errors.append(f"Doc {doc_id}: full_text file missing at {full_text_path}")
            continue

        if not metadata_path.exists():
            errors.append(f"Doc {doc_id}: metadata file missing at {metadata_path}")
            continue

        # Canonical content hash verification
        from ingestion_source.normalizer import compute_sha256_hash
        text = full_text_path.read_text(encoding="utf-8")
        calculated_hash = compute_sha256_hash(text)
        manifest_hash = rec.get("content_hash")

        if calculated_hash != manifest_hash:
            errors.append(f"Doc {doc_id}: SHA-256 mismatch (expected {manifest_hash}, calculated {calculated_hash})")
            continue

        # Metadata integrity check
        with open(metadata_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)

        for req_field in ["doc_id", "court", "title", "source", "source_url"]:
            if not meta.get(req_field):
                errors.append(f"Doc {doc_id}: Metadata missing required field '{req_field}'")
                break

        valid_count += 1

    print("=" * 60)
    print(f"Total manifest entries: {len(records)}")
    print(f"Verified & intact records: {valid_count}")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\nError details (first 5):")
        for err in errors[:5]:
            print(f" - {err}")
        return False

    print("\n[OK] Pipeline A Intake Contract Verification: PASSED (100% Intact)")
    return True

if __name__ == "__main__":
    store_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("raw-store")
    success = verify_intake_contract(store_dir)
    sys.exit(0 if success else 1)
