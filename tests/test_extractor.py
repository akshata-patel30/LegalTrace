"""Unit tests for LegalMetadataExtractor (Step 4: Legal Metadata & Entity Extraction)."""

from pathlib import Path
import pytest

from ingestion_source.extractor import LegalMetadataExtractor
from ingestion_source.models import (
    BenchType,
    CleanedHeader,
    CleanedJudgment,
    CleaningMetrics,
    DispositionType,
    LegalDomain,
)


@pytest.fixture
def extractor() -> LegalMetadataExtractor:
    return LegalMetadataExtractor()


def test_classify_case_type_and_domain(extractor: LegalMetadataExtractor) -> None:
    # Bail Application
    ctype, code, domain = extractor._classify_case_type_and_domain(
        ["BAIL APPLN. 71/2023"], "Some text", "Quentin vs State"
    )
    assert code == "BAIL_APPLICATION"
    assert domain == LegalDomain.CRIMINAL

    # Commercial Suit
    ctype, code, domain = extractor._classify_case_type_and_domain(
        ["CS(COMM) 120/2023"], "Commercial dispute text", "Company A vs Company B"
    )
    assert code == "CS_COMM"
    assert domain == LegalDomain.COMMERCIAL

    # Arbitration
    ctype, code, domain = extractor._classify_case_type_and_domain(
        ["O.M.P.(COMM) 45/2023"], "Arbitration text", "X vs Y"
    )
    assert code == "OMP_COMM"
    assert domain == LegalDomain.ARBITRATION

    # Writ Petition Civil
    ctype, code, domain = extractor._classify_case_type_and_domain(
        ["W.P.(C) 100/2023"], "Constitutional petition", "Citizen vs Union of India"
    )
    assert code == "WP_C"
    assert domain == LegalDomain.CONSTITUTIONAL


def test_extract_judges_and_bench(extractor: LegalMetadataExtractor) -> None:
    # Single Judge
    judges, btype = extractor._extract_judges_and_bench(
        ["HON'BLE MR. JUSTICE AMIT SHARMA"], "CORAM: HON'BLE MR. JUSTICE AMIT SHARMA"
    )
    assert len(judges) == 1
    assert judges[0] == "Justice Amit Sharma"
    assert btype == BenchType.SINGLE_BENCH

    # Division Bench
    judges_db, btype_db = extractor._extract_judges_and_bench(
        ["HON'BLE MR. JUSTICE MANMOHAN", "HON'BLE MS. JUSTICE MINI PUSHKARNA"], "Some text"
    )
    assert len(judges_db) == 2
    assert btype_db == BenchType.DIVISION_BENCH
    assert "Justice Manmohan" in judges_db
    assert "Justice Mini Pushkarna" in judges_db


def test_extract_statutes(extractor: LegalMetadataExtractor) -> None:
    text = (
        "The applicant was arrested under Sections 21, 23 and 29 of the NDPS Act, 1985. "
        "The present petition under Section 439 of the Cr.P.C. seeks regular bail. "
        "Also Article 226 of the Constitution of India was invoked."
    )
    statutes = extractor._extract_statutes(text)
    codes = {s.act_code: s for s in statutes}

    assert "NDPS Act" in codes
    assert "21" in codes["NDPS Act"].sections
    assert "23" in codes["NDPS Act"].sections
    assert "29" in codes["NDPS Act"].sections

    assert "CrPC" in codes
    assert "439" in codes["CrPC"].sections

    assert "Constitution" in codes
    assert "226" in codes["Constitution"].sections


def test_extract_citations(extractor: LegalMetadataExtractor) -> None:
    text = (
        "In AIR 1980 SC 52 and 2022 SCC OnLine Del 4427, the Supreme Court and this Court held that "
        "precedent applies. Also cited was (2018) 1 SCC 1."
    )
    citations = extractor._extract_citations(text, neutral_citation="2023:DHC:1234")
    cit_texts = [c.citation_text for c in citations]

    assert "2023:DHC:1234" in cit_texts
    assert "AIR 1980 SC 52" in cit_texts
    assert "2022 SCC OnLine Del 4427" in cit_texts
    assert "(2018) 1 SCC 1" in cit_texts


def test_classify_counsel(extractor: LegalMetadataExtractor) -> None:
    counsels = [
        "Mr. Vikas Pahwa, Senior Advocate with Mr. X",
        "Mr. Rajesh Mahajan, Special Public Prosecutor",
        "Mr. Y, Advocate",
    ]
    senior, standing = extractor._classify_counsel(counsels)
    assert len(senior) == 1
    assert "Senior Advocate" in senior[0]
    assert len(standing) == 1
    assert "Special Public Prosecutor" in standing[0]
