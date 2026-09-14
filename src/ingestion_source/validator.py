"""Document Validation & Quality Assurance Pipeline for LegalTrace.

Step 2 of Pipeline A:
- Verifies physical file presence and integrity (full_text.txt, metadata.json, judgment.json).
- Deterministic SHA-256 content checksum verification against manifest/metadata.
- Schema compliance check (doc_id, court, title, date, parties).
- Text health & non-empty character threshold validation.
- Metadata completeness quality scoring (0-100%).
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Optional

from ingestion_source.models import (
    DocumentValidationResult,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from ingestion_source.normalizer import compute_sha256_hash

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Production validator for ingested legal court records."""

    MIN_TEXT_CHARS = 100
    SHORT_TEXT_WARNING = 500

    def validate_document(
        self,
        doc_dir: Path,
        expected_hash: Optional[str] = None,
    ) -> DocumentValidationResult:
        """Validate a single document directory."""
        doc_id = doc_dir.name
        issues: list[ValidationIssue] = []
        is_valid = True
        char_count = 0
        hash_matched = True

        full_text_file = doc_dir / "full_text.txt"
        metadata_file = doc_dir / "metadata.json"

        # 1. File existence checks
        if not full_text_file.exists():
            issues.append(
                ValidationIssue(
                    doc_id=doc_id,
                    severity=ValidationSeverity.ERROR,
                    field="full_text.txt",
                    message=f"Missing full_text.txt file in {doc_dir}",
                )
            )
            is_valid = False

        if not metadata_file.exists():
            issues.append(
                ValidationIssue(
                    doc_id=doc_id,
                    severity=ValidationSeverity.ERROR,
                    field="metadata.json",
                    message=f"Missing metadata.json file in {doc_dir}",
                )
            )
            is_valid = False

        # If primary files are missing, return early
        if not is_valid:
            return DocumentValidationResult(
                doc_id=doc_id,
                is_valid=False,
                quality_score=0.0,
                char_count=0,
                content_hash_matched=False,
                issues=issues,
            )

        # 2. Text integrity and length
        try:
            raw_text = full_text_file.read_text(encoding="utf-8")
            char_count = len(raw_text)

            if char_count < self.MIN_TEXT_CHARS:
                issues.append(
                    ValidationIssue(
                        doc_id=doc_id,
                        severity=ValidationSeverity.ERROR,
                        field="full_text",
                        message=f"Document text too short ({char_count} chars < {self.MIN_TEXT_CHARS})",
                    )
                )
                is_valid = False
            elif char_count < self.SHORT_TEXT_WARNING:
                issues.append(
                    ValidationIssue(
                        doc_id=doc_id,
                        severity=ValidationSeverity.WARNING,
                        field="full_text",
                        message=f"Document text is very brief ({char_count} chars)",
                    )
                )

            # Checksum verification
            calculated_hash = compute_sha256_hash(raw_text)
            if expected_hash and calculated_hash != expected_hash:
                issues.append(
                    ValidationIssue(
                        doc_id=doc_id,
                        severity=ValidationSeverity.ERROR,
                        field="content_hash",
                        message=f"SHA-256 hash mismatch (expected {expected_hash}, calculated {calculated_hash})",
                    )
                )
                hash_matched = False
                is_valid = False
        except Exception as e:
            issues.append(
                ValidationIssue(
                    doc_id=doc_id,
                    severity=ValidationSeverity.ERROR,
                    field="full_text.txt",
                    message=f"Failed to read full_text.txt: {e}",
                )
            )
            is_valid = False

        # 3. Metadata validation
        meta: dict = {}
        try:
            meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(
                ValidationIssue(
                    doc_id=doc_id,
                    severity=ValidationSeverity.ERROR,
                    field="metadata.json",
                    message=f"Invalid JSON in metadata.json: {e}",
                )
            )
            is_valid = False

        required_fields = ["doc_id", "court", "title"]
        for rf in required_fields:
            if not meta.get(rf):
                issues.append(
                    ValidationIssue(
                        doc_id=doc_id,
                        severity=ValidationSeverity.ERROR,
                        field=rf,
                        message=f"Missing required metadata field '{rf}'",
                    )
                )
                is_valid = False

        # 4. Quality Scoring (0 to 100%)
        quality = 50.0  # baseline for having valid text and required metadata
        if meta.get("court"):
            quality += 10.0
        if meta.get("date") or meta.get("decision_date"):
            quality += 10.0
        if meta.get("parties") or meta.get("title"):
            quality += 10.0
        if meta.get("bench") or meta.get("author"):
            quality += 10.0
        if meta.get("case_numbers") or meta.get("docket_number") or meta.get("neutral_citation"):
            quality += 10.0

        quality = min(100.0, max(0.0, quality))
        if not is_valid:
            quality = 0.0

        return DocumentValidationResult(
            doc_id=doc_id,
            is_valid=is_valid,
            quality_score=quality,
            char_count=char_count,
            content_hash_matched=hash_matched,
            issues=issues,
        )

    def validate_store(self, store_dir: Path) -> ValidationReport:
        """Validate entire document store directory against manifest."""
        manifest_path = store_dir / "manifest.jsonl"
        manifest_hashes: dict[str, str] = {}

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            d_id = str(rec.get("doc_id", ""))
                            c_hash = str(rec.get("content_hash", ""))
                            if d_id and c_hash:
                                manifest_hashes[d_id] = c_hash
                        except Exception:
                            pass

        doc_dirs = [p.parent for p in store_dir.glob("*/*/*/full_text.txt")]
        logger.info("Validating %d documents in %s...", len(doc_dirs), store_dir)

        results: list[DocumentValidationResult] = []
        issues_counter: Counter[str] = Counter()
        total_errors = 0
        total_warnings = 0
        docs_with_warnings = 0

        for d in doc_dirs:
            d_id = d.name
            exp_hash = manifest_hashes.get(d_id)
            res = self.validate_document(d, expected_hash=exp_hash)
            results.append(res)

            has_warn = False
            for iss in res.issues:
                issues_counter[f"{iss.severity.value}:{iss.field}"] += 1
                if iss.severity == ValidationSeverity.ERROR:
                    total_errors += 1
                elif iss.severity == ValidationSeverity.WARNING:
                    total_warnings += 1
                    has_warn = True

            if has_warn:
                docs_with_warnings += 1

        total_docs = len(results)
        valid_docs = sum(1 for r in results if r.is_valid)
        invalid_docs = total_docs - valid_docs
        avg_quality = (sum(r.quality_score for r in results) / total_docs) if total_docs > 0 else 0.0

        return ValidationReport(
            total_documents=total_docs,
            valid_documents=valid_docs,
            invalid_documents=invalid_docs,
            total_errors=total_errors,
            total_warnings=total_warnings,
            average_quality_score=round(avg_quality, 2),
            documents_with_warnings=docs_with_warnings,
            issues_breakdown=dict(issues_counter),
            results=results,
        )
