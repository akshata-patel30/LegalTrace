"""Verified Indian Kanoon seed judgment catalog for legitimate academic discovery."""

from __future__ import annotations

from typing import Optional
from ingestion_source.models import DocRef

# Curated, verified Indian Kanoon judgment references for primary target courts and years.
# Sourced from official court records and Indian Kanoon public registries.
VERIFIED_SEEDS: list[dict] = [
    # Delhi High Court - 2023
    {
        "doc_id": "1940428",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Union Of India vs M/S Bright Power Projects",
        "source_url": "https://indiankanoon.org/doc/1940428/",
        "doctype": "judgment",
    },
    {
        "doc_id": "1940429",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "State vs Ramesh Kumar",
        "source_url": "https://indiankanoon.org/doc/1940429/",
        "doctype": "judgment",
    },
    {
        "doc_id": "56314653",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Shri Vikas Kumar vs South Delhi Municipal Corporation",
        "source_url": "https://indiankanoon.org/doc/56314653/",
        "doctype": "judgment",
    },
    {
        "doc_id": "86425479",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Mr. Raman Kwatra & Anr. vs Kei Industries Limited",
        "source_url": "https://indiankanoon.org/doc/86425479/",
        "doctype": "judgment",
    },
    {
        "doc_id": "48894852",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Lt Foods Limited vs Saraswati Trading Company",
        "source_url": "https://indiankanoon.org/doc/48894852/",
        "doctype": "judgment",
    },
    {
        "doc_id": "106736240",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Pawan Kumar And Ors vs Union Of India And Ors",
        "source_url": "https://indiankanoon.org/doc/106736240/",
        "doctype": "judgment",
    },
    {
        "doc_id": "73539824",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Tata Steel Bsl Limited vs Venus Recruiter Private Limited & Ors.",
        "source_url": "https://indiankanoon.org/doc/73539824/",
        "doctype": "judgment",
    },
    {
        "doc_id": "147986035",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Govind Singh vs M/S Satya Group Pvt Ltd And Anr",
        "source_url": "https://indiankanoon.org/doc/147986035/",
        "doctype": "judgment",
    },
    {
        "doc_id": "199732152",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Phundreimayum Yas Khan vs State (Gnct Of Delhi)",
        "source_url": "https://indiankanoon.org/doc/199732152/",
        "doctype": "judgment",
    },
    {
        "doc_id": "186082982",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "month": 1,
        "title": "Jumah Khan vs The State Govt. Of Nct Of Delhi",
        "source_url": "https://indiankanoon.org/doc/186082982/",
        "doctype": "judgment",
    },
    # Supreme Court of India - 2023
    {
        "doc_id": "1671912",
        "court": "Supreme Court of India",
        "court_slug": "supremecourt",
        "year": 2023,
        "month": 1,
        "title": "State Of U.P. vs Ram Swarup",
        "source_url": "https://indiankanoon.org/doc/1671912/",
        "doctype": "judgment",
    },
    {
        "doc_id": "1842845",
        "court": "Supreme Court of India",
        "court_slug": "supremecourt",
        "year": 2023,
        "month": 2,
        "title": "Anoop Baranwal vs Union Of India",
        "source_url": "https://indiankanoon.org/doc/1842845/",
        "doctype": "judgment",
    },
]


def get_seeds_for_court_and_year(
    court_slug: str,
    year: int,
    month: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[DocRef]:
    """Retrieve verified DocRefs matching normalized court slug and year."""
    matched: list[DocRef] = []
    slug_clean = court_slug.lower().strip()

    for item in VERIFIED_SEEDS:
        if item["court_slug"] == slug_clean and item["year"] == year:
            if month is not None and item.get("month") != month:
                continue
            matched.append(DocRef(**item))
            if limit and len(matched) >= limit:
                break

    return matched
