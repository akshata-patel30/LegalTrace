"""Legal-Aware Semantic Chunker for LegalTrace.

Step 6 of Pipeline A:
- Semantic chunking respecting legal rhetorical role boundaries (zero cross-role contamination).
- Context header prepending [Court, Docket, Year, Judge, Role, Title] for dense vector retrieval.
- Paragraph overlap within identical roles; hard boundary on role transitions.
- Intact statutory sections and precedent quote blocks.
- Sentence-boundary splitting for oversized paragraphs.
- Document-level and corpus-wide batch chunking with JSON/JSONL output.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Optional

from ingestion_source.models import (
    ChunkingReport,
    CleanedJudgment,
    CleanedParagraph,
    EnrichedLegalMetadata,
    LegalChunk,
    ParagraphType,
    RhetoricalRole,
)

logger = logging.getLogger(__name__)


class LegalSemanticChunker:
    """Chunks legal documents into semantically coherent, role-isolated vector units."""

    def __init__(
        self,
        min_words: int = 150,
        target_words: int = 400,
        max_words: int = 800,
        overlap_paras: int = 1,
    ) -> None:
        """Initialize chunker with token/word boundaries.

        Args:
            min_words: Minimum target word count before considering a split.
            target_words: Desired target word count per chunk.
            max_words: Hard maximum word count before forcing a sentence-level break.
            overlap_paras: Number of paragraphs to overlap between consecutive chunks in the same role.
        """
        self.min_words = min_words
        self.target_words = target_words
        self.max_words = max_words
        self.overlap_paras = overlap_paras

    def chunk_judgment(
        self,
        judgment: CleanedJudgment,
        enriched: Optional[EnrichedLegalMetadata] = None,
    ) -> list[LegalChunk]:
        """Chunk a CleanedJudgment into role-isolated LegalChunks."""
        if not judgment.paragraphs:
            return []

        doc_id = judgment.doc_id
        court = judgment.court
        court_slug = judgment.court_slug
        year = judgment.year
        case_title = judgment.title

        # Resolve docket number
        docket_number = judgment.header.docket_numbers[0] if judgment.header.docket_numbers else None

        # Resolve presiding judge
        presiding_judge = None
        if enriched and enriched.canonical_judges:
            presiding_judge = ", ".join(enriched.canonical_judges)
        elif judgment.header.bench:
            presiding_judge = ", ".join(judgment.header.bench)

        # Resolve legal domain & case type
        legal_domain = enriched.legal_domain.value if enriched else None
        case_type = enriched.case_type if enriched else None

        # 1. Group consecutive paragraphs by identical rhetorical role
        role_groups: list[list[CleanedParagraph]] = []
        current_group: list[CleanedParagraph] = []
        current_role: Optional[RhetoricalRole] = None

        for para in judgment.paragraphs:
            if current_role is None:
                current_role = para.rhetorical_role
                current_group.append(para)
            elif para.rhetorical_role == current_role:
                current_group.append(para)
            else:
                role_groups.append(current_group)
                current_group = [para]
                current_role = para.rhetorical_role

        if current_group:
            role_groups.append(current_group)

        # 2. Chunk each role group independently (preventing cross-role mixing)
        chunks: list[LegalChunk] = []
        chunk_idx = 1

        for group in role_groups:
            group_role = group[0].rhetorical_role
            group_chunks = self._chunk_role_group(
                group=group,
                role=group_role,
                doc_id=doc_id,
                start_index=chunk_idx,
                court=court,
                court_slug=court_slug,
                year=year,
                case_title=case_title,
                docket_number=docket_number,
                presiding_judge=presiding_judge,
                legal_domain=legal_domain,
                case_type=case_type,
                enriched=enriched,
            )
            chunks.extend(group_chunks)
            chunk_idx += len(group_chunks)

        return chunks

    def _chunk_role_group(
        self,
        group: list[CleanedParagraph],
        role: RhetoricalRole,
        doc_id: str,
        start_index: int,
        court: str,
        court_slug: str,
        year: int,
        case_title: str,
        docket_number: Optional[str],
        presiding_judge: Optional[str],
        legal_domain: Optional[str],
        case_type: Optional[str],
        enriched: Optional[EnrichedLegalMetadata],
    ) -> list[LegalChunk]:
        """Produce LegalChunks from a list of paragraphs sharing the same rhetorical role."""
        if not group:
            return []

        # First, flatten any individual oversized paragraphs by splitting them on sentences
        expanded_items: list[tuple[list[int], str, bool]] = []
        for p in group:
            words = p.text.split()
            is_quote = p.is_quote or p.para_type in (ParagraphType.STATUTORY_QUOTE, ParagraphType.CASE_QUOTE)
            if len(words) <= self.max_words:
                expanded_items.append(([p.para_id], p.text, is_quote))
            else:
                # Oversized paragraph: split into sentence blocks
                sub_parts = self._split_text_on_sentences(p.text, max_words=self.max_words)
                for part_text in sub_parts:
                    expanded_items.append(([p.para_id], part_text, is_quote))

        # Now group expanded items into chunks adhering to target_words and max_words
        assembled_chunks: list[LegalChunk] = []
        curr_paras: list[int] = []
        curr_texts: list[str] = []
        curr_words = 0
        curr_has_quote = False
        curr_index = start_index

        i = 0
        while i < len(expanded_items):
            para_ids, text, is_quote = expanded_items[i]
            item_words = len(text.split())

            # If adding this item exceeds max_words and we already have accumulated items
            if curr_words + item_words > self.max_words and curr_texts:
                # Emit current chunk
                assembled_chunks.append(
                    self._create_chunk(
                        doc_id=doc_id,
                        chunk_index=curr_index,
                        court=court,
                        court_slug=court_slug,
                        year=year,
                        case_title=case_title,
                        docket_number=docket_number,
                        presiding_judge=presiding_judge,
                        legal_domain=legal_domain,
                        case_type=case_type,
                        role=role,
                        para_ids=curr_paras,
                        text="\n\n".join(curr_texts),
                        is_quote_block=curr_has_quote,
                        enriched=enriched,
                    )
                )
                curr_index += 1

                # Apply overlap if configured and we had multiple items
                if self.overlap_paras > 0 and len(curr_texts) > 1:
                    overlap_item = expanded_items[i - 1]
                    curr_paras = list(overlap_item[0])
                    curr_texts = [overlap_item[1]]
                    curr_words = len(overlap_item[1].split())
                    curr_has_quote = overlap_item[2]
                else:
                    curr_paras = []
                    curr_texts = []
                    curr_words = 0
                    curr_has_quote = False

            # Add current item
            for pid in para_ids:
                if pid not in curr_paras:
                    curr_paras.append(pid)
            curr_texts.append(text)
            curr_words += item_words
            if is_quote:
                curr_has_quote = True

            # If we've reached or exceeded target_words, we can complete chunk if remaining items exist
            if curr_words >= self.target_words and i + 1 < len(expanded_items):
                assembled_chunks.append(
                    self._create_chunk(
                        doc_id=doc_id,
                        chunk_index=curr_index,
                        court=court,
                        court_slug=court_slug,
                        year=year,
                        case_title=case_title,
                        docket_number=docket_number,
                        presiding_judge=presiding_judge,
                        legal_domain=legal_domain,
                        case_type=case_type,
                        role=role,
                        para_ids=curr_paras,
                        text="\n\n".join(curr_texts),
                        is_quote_block=curr_has_quote,
                        enriched=enriched,
                    )
                )
                curr_index += 1

                if self.overlap_paras > 0 and len(curr_texts) > 1:
                    overlap_item = expanded_items[i]
                    curr_paras = list(overlap_item[0])
                    curr_texts = [overlap_item[1]]
                    curr_words = len(overlap_item[1].split())
                    curr_has_quote = overlap_item[2]
                else:
                    curr_paras = []
                    curr_texts = []
                    curr_words = 0
                    curr_has_quote = False

            i += 1

        # Emit any trailing accumulated text
        if curr_texts:
            assembled_chunks.append(
                self._create_chunk(
                    doc_id=doc_id,
                    chunk_index=curr_index,
                    court=court,
                    court_slug=court_slug,
                    year=year,
                    case_title=case_title,
                    docket_number=docket_number,
                    presiding_judge=presiding_judge,
                    legal_domain=legal_domain,
                    case_type=case_type,
                    role=role,
                    para_ids=curr_paras,
                    text="\n\n".join(curr_texts),
                    is_quote_block=curr_has_quote,
                    enriched=enriched,
                )
            )

        return assembled_chunks

    def _split_text_on_sentences(self, text: str, max_words: int) -> list[str]:
        """Split oversized text at sentence boundaries into segments of <= max_words."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments: list[str] = []
        curr_sentences: list[str] = []
        curr_word_count = 0

        for sentence in sentences:
            s_words = len(sentence.split())
            if curr_word_count + s_words > max_words and curr_sentences:
                segments.append(" ".join(curr_sentences))
                curr_sentences = []
                curr_word_count = 0

            curr_sentences.append(sentence)
            curr_word_count += s_words

        if curr_sentences:
            segments.append(" ".join(curr_sentences))

        return segments

    def _create_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        court: str,
        court_slug: str,
        year: int,
        case_title: str,
        docket_number: Optional[str],
        presiding_judge: Optional[str],
        legal_domain: Optional[str],
        case_type: Optional[str],
        role: RhetoricalRole,
        para_ids: list[int],
        text: str,
        is_quote_block: bool,
        enriched: Optional[EnrichedLegalMetadata],
    ) -> LegalChunk:
        """Construct a LegalChunk with context header and entity cross-linking."""
        chunk_id = f"{doc_id}_c{chunk_index:03d}"
        clean_text = text.strip()
        words = clean_text.split()
        word_count = len(words)
        char_count = len(clean_text)

        # Build context header
        context_header = self._build_context_header(
            court=court,
            docket=docket_number,
            year=year,
            judge=presiding_judge,
            role=role,
            case_title=case_title,
        )
        text_with_context = f"{context_header}\n{clean_text}"

        # Match relevant statutes and citations present in this chunk
        chunk_statutes: list[str] = []
        chunk_citations: list[str] = []

        if enriched:
            text_lower = clean_text.lower()
            for st in enriched.statutes:
                if st.act_code.lower() in text_lower or (len(st.act_name) > 4 and st.act_name.lower() in text_lower):
                    if st.sections:
                        sec_str = ", ".join(st.sections)
                        chunk_statutes.append(f"{st.act_code}: Section {sec_str}")
                    else:
                        chunk_statutes.append(st.act_code)

            for cit in enriched.citations:
                if cit.citation_text.lower() in text_lower:
                    chunk_citations.append(cit.citation_text)

        return LegalChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            court=court,
            court_slug=court_slug,
            year=year,
            case_title=case_title,
            docket_number=docket_number,
            presiding_judge=presiding_judge,
            legal_domain=legal_domain,
            case_type=case_type,
            rhetorical_role=role,
            para_ids=para_ids,
            text=clean_text,
            text_with_context=text_with_context,
            char_count=char_count,
            word_count=word_count,
            statute_mentions=sorted(list(set(chunk_statutes))),
            citations=sorted(list(set(chunk_citations))),
            is_quote_block=is_quote_block,
        )

    def _build_context_header(
        self,
        court: str,
        docket: Optional[str],
        year: int,
        judge: Optional[str],
        role: RhetoricalRole,
        case_title: str,
    ) -> str:
        """Generate structured metadata prefix for vector embedding."""
        parts = [f"Court: {court}"]
        if docket:
            parts.append(f"Docket: {docket}")
        parts.append(f"Year: {year}")
        if judge:
            parts.append(f"Judge: {judge}")
        parts.append(f"Role: {role.value.upper()}")
        parts.append(f"Title: {case_title}")
        return f"[{' | '.join(parts)}]"

    def chunk_document_dir(self, doc_dir: Path) -> list[LegalChunk]:
        """Chunk a single judgment folder and persist chunks.json."""
        cleaned_file = doc_dir / "cleaned.json"
        if not cleaned_file.exists():
            raise FileNotFoundError(f"Missing cleaned.json in {doc_dir}")

        cleaned_data = json.loads(cleaned_file.read_text(encoding="utf-8"))
        judgment = CleanedJudgment(**cleaned_data)

        # Check for optional enriched.json
        enriched: Optional[EnrichedLegalMetadata] = None
        enriched_file = doc_dir / "enriched.json"
        if enriched_file.exists():
            enriched_data = json.loads(enriched_file.read_text(encoding="utf-8"))
            enriched = EnrichedLegalMetadata(**enriched_data)

        chunks = self.chunk_judgment(judgment=judgment, enriched=enriched)

        # Write chunks.json
        chunks_file = doc_dir / "chunks.json"
        chunks_file.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return chunks

    def chunk_store(
        self,
        store_dir: Path,
        output_jsonl: Optional[Path] = None,
    ) -> list[LegalChunk]:
        """Chunk all documents in raw-store and optionally write consolidated chunks.jsonl."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Found %d judgments to chunk in %s", len(cleaned_files), store_dir)

        all_chunks: list[LegalChunk] = []
        for i, f in enumerate(cleaned_files, 1):
            try:
                chunks = self.chunk_document_dir(f.parent)
                all_chunks.extend(chunks)
                if i % 25 == 0 or i == len(cleaned_files):
                    logger.info("Chunked %d/%d judgments (%d total chunks)...", i, len(cleaned_files), len(all_chunks))
            except Exception as e:
                logger.error("Error chunking %s: %s", f.parent, e)

        if output_jsonl and all_chunks:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as out:
                for chunk in all_chunks:
                    out.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d chunks to %s", len(all_chunks), output_jsonl)

        return all_chunks

    def generate_report(self, chunks: list[LegalChunk], total_docs: int) -> ChunkingReport:
        """Compute summary metrics and rhetorical role distribution."""
        total_chunks = len(chunks)
        avg_chunks_per_doc = (total_chunks / total_docs) if total_docs > 0 else 0.0
        avg_words = (sum(c.word_count for c in chunks) / total_chunks) if total_chunks > 0 else 0.0
        avg_chars = (sum(c.char_count for c in chunks) / total_chunks) if total_chunks > 0 else 0.0

        role_dist = Counter(c.rhetorical_role.value for c in chunks)

        return ChunkingReport(
            total_documents=total_docs,
            total_chunks=total_chunks,
            average_chunks_per_doc=round(avg_chunks_per_doc, 2),
            average_words_per_chunk=round(avg_words, 1),
            average_chars_per_chunk=round(avg_chars, 1),
            role_distribution=dict(role_dist.most_common()),
        )
