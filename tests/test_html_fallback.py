"""Unit tests for polite HTML fallback scraper."""

import pytest
import httpx
import respx

from ingestion_source.config import Settings
from ingestion_source.models import DocRef, FetchFailureType, SourceType
from ingestion_source.rate_limiter import RobotsTxtManager
from ingestion_source.sources.html_fallback import (
    IndianKanoonHTMLFallback,
    normalize_cookie_header,
    parse_cookie_dict,
)


@respx.mock
def test_html_fallback_fetch_success(
    test_settings: Settings,
    sample_doc_ref: DocRef,
    mock_judgment_html: str,
    mock_robots_txt: str,
):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    respx.get(f"https://indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(200, text=mock_judgment_html)
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    result = scraper.fetch(sample_doc_ref)

    assert result.success is True
    assert result.source == SourceType.SCRAPE
    assert result.status_code == 200
    assert result.raw_content == mock_judgment_html
    scraper.close()


@respx.mock
def test_html_fallback_challenge_detected(
    test_settings: Settings,
    sample_doc_ref: DocRef,
    mock_robots_txt: str,
):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    challenge_html = "<!DOCTYPE html><html><head><title>Just a moment...</title><script src='https://challenges.cloudflare.com/turnstile'></script></head><body></body></html>"
    respx.get(f"https://indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(403, text=challenge_html)
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    result = scraper.fetch(sample_doc_ref)

    assert result.success is False
    assert result.failure_type == FetchFailureType.CHALLENGE_DETECTED
    assert scraper.rate_limiter.is_cooling_down() is True
    scraper.close()


@respx.mock
def test_html_fallback_structural_drift(
    test_settings: Settings,
    sample_doc_ref: DocRef,
    mock_robots_txt: str,
):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    # Return HTML missing expected containers
    respx.get(f"https://indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(200, text="<html><body><div>Completely unexpected layout with no text</div></body></html>")
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    result = scraper.fetch(sample_doc_ref)

    assert result.success is False
    assert result.failure_type == FetchFailureType.STRUCTURAL_DRIFT
    scraper.close()


def test_html_fallback_robots_disallowed(test_settings: Settings, sample_doc_ref: DocRef):
    robots_mgr = RobotsTxtManager()
    robots_mgr.set_content("User-agent: *\nDisallow: /doc/\n")

    scraper = IndianKanoonHTMLFallback(settings=test_settings, robots_mgr=robots_mgr)
    result = scraper.fetch(sample_doc_ref)

    assert result.success is False
    assert result.failure_type == FetchFailureType.DISALLOWED_BY_ROBOTS
    scraper.close()


@respx.mock
def test_html_fallback_discovery(test_settings: Settings, mock_robots_txt: str):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    browse_html = """
    <html>
    <body>
        <div class="browselist">
            <a href="/doc/1001/">Judgment Alpha vs Beta</a>
            <a href="/doc/1002/">Order Gamma vs Delta</a>
        </div>
    </body>
    </html>
    """
    respx.get("https://indiankanoon.org/browse/delhi/2023/").mock(
        return_value=httpx.Response(200, text=browse_html)
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    refs = list(scraper.discover(court="Delhi High Court", year=2023, limit=2))

    assert len(refs) == 2
    assert refs[0].doc_id == "1001"
    assert refs[0].title == "Judgment Alpha vs Beta"
    assert refs[1].doc_id == "1002"
    scraper.close()


@respx.mock
def test_html_fallback_discovery_403_challenge(test_settings: Settings, mock_robots_txt: str):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    challenge_html = "<html><head><title>Just a moment...</title><script src='https://challenges.cloudflare.com'></script></head><body></body></html>"
    respx.get("https://indiankanoon.org/browse/delhi/2023/").mock(
        return_value=httpx.Response(403, text=challenge_html)
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    refs = list(scraper.discover(court="Delhi High Court", year=2023, limit=5))

    assert len(refs) == 0
    assert scraper.last_access_control_encountered is True
    scraper.close()


def test_parse_cookie_dict_and_normalization():
    # 1. Standard string
    raw = "csrftoken=abc12345; sessionid=sess999; cf_clearance=clr888"
    d = parse_cookie_dict(raw)
    assert d == {"csrftoken": "abc12345", "sessionid": "sess999", "cf_clearance": "clr888"}
    assert normalize_cookie_header(raw) == "csrftoken=abc12345; sessionid=sess999; cf_clearance=clr888"

    # 2. Netscape tab format
    netscape = ".indiankanoon.org\tTRUE\t/\tFALSE\t1790000000\tcf_clearance\ttoken123\n.indiankanoon.org\tTRUE\t/\tFALSE\t1790000000\tcsrftoken\tcsrf456"
    d2 = parse_cookie_dict(netscape)
    assert d2 == {"cf_clearance": "token123", "csrftoken": "csrf456"}
    assert "cf_clearance=token123" in normalize_cookie_header(netscape)
    assert "csrftoken=csrf456" in normalize_cookie_header(netscape)

    # 3. None or empty
    assert parse_cookie_dict(None) == {}
    assert normalize_cookie_header(None) is None
    assert parse_cookie_dict("") == {}
    assert normalize_cookie_header("") is None


def test_html_fallback_cookie_jar_init(test_settings: Settings):
    test_settings.INDIANKANOON_COOKIES = "csrftoken=xyz; cf_clearance=clr_xyz"
    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    assert scraper._client.cookies.get("csrftoken") == "xyz"
    assert scraper._client.cookies.get("cf_clearance") == "clr_xyz"
    scraper.close()


@respx.mock
def test_html_fallback_discover_by_search(test_settings: Settings, mock_robots_txt: str):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )
    search_html = """
    <html>
    <body>
      <div class="result">
        <div class="result_title"><a href="/doc/998877/">State vs Ramesh Kumar (Section 37 NDPS)</a></div>
        <div class="headline">Commercial quantity recovery was alleged under Section 37...</div>
      </div>
      <div class="result">
        <div class="result_title"><a href="/doc/998878/">Suresh vs Union of India</a></div>
        <div class="headline">Bail granted under Section 37 NDPS Act...</div>
      </div>
    </body>
    </html>
    """
    respx.get(url__regex=r"https://indiankanoon\.org/search/.*").mock(
        return_value=httpx.Response(200, text=search_html)
    )

    scraper = IndianKanoonHTMLFallback(settings=test_settings)
    refs = list(scraper.discover(court="Delhi High Court", year=2023, limit=2, query="Section 37 NDPS bail"))

    assert len(refs) == 2
    assert refs[0].doc_id == "998877"
    assert refs[0].title == "State vs Ramesh Kumar (Section 37 NDPS)"
    assert "Section 37" in (refs[0].headline or "")
    assert refs[1].doc_id == "998878"
    scraper.close()



