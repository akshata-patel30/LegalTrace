"""Unit tests for Indian Kanoon official API client."""

import pytest
import httpx
import respx

from ingestion_source.config import Settings
from ingestion_source.models import DocRef, FetchFailureType, SourceType
from ingestion_source.sources.api_client import IndianKanoonAPIClient


@respx.mock
def test_api_client_fetch_success(test_settings: Settings, sample_doc_ref: DocRef, mock_api_doc_json: dict):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    result = client.fetch(sample_doc_ref)

    assert result.success is True
    assert result.source == SourceType.API
    assert result.status_code == 200
    assert result.raw_json == mock_api_doc_json
    assert client.api_calls_count == 1
    client.close()


@respx.mock
def test_api_client_fetch_not_found(test_settings: Settings, sample_doc_ref: DocRef):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(404, text="Document not found")
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    result = client.fetch(sample_doc_ref)

    assert result.success is False
    assert result.status_code == 404
    assert result.failure_type == FetchFailureType.NOT_FOUND
    client.close()


@respx.mock
def test_api_client_fetch_rate_limited(test_settings: Settings, sample_doc_ref: DocRef):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(429, text="Rate limit exceeded")
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    result = client.fetch(sample_doc_ref)

    assert result.success is False
    assert result.status_code == 429
    assert result.failure_type == FetchFailureType.RATE_LIMITED
    client.close()


@respx.mock
def test_api_client_fetch_forbidden(test_settings: Settings, sample_doc_ref: DocRef):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(403, text="Invalid API key")
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    result = client.fetch(sample_doc_ref)

    assert result.success is False
    assert result.status_code == 403
    assert result.failure_type == FetchFailureType.FORBIDDEN
    client.close()


def test_api_client_budget_limit(test_settings: Settings, sample_doc_ref: DocRef):
    test_settings.MAX_API_CALLS_PER_RUN = 1
    client = IndianKanoonAPIClient(settings=test_settings)
    client.api_calls_count = 1  # Cap reached

    result = client.fetch(sample_doc_ref)
    assert result.success is False
    assert result.failure_type == FetchFailureType.BUDGET_EXCEEDED
    client.close()


@respx.mock
def test_api_client_discovery(test_settings: Settings, mock_api_search_json: dict):
    respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=mock_api_search_json)
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    refs = list(client.discover(court="Delhi High Court", year=2023, limit=2))

    assert len(refs) == 2
    assert refs[0].doc_id == "1940428"
    assert refs[1].doc_id == "1940429"
    client.close()


@respx.mock
def test_api_client_discover_with_query(test_settings: Settings, mock_api_search_json: dict):
    route = respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=mock_api_search_json)
    )

    client = IndianKanoonAPIClient(settings=test_settings)
    refs = list(client.discover(court="Delhi High Court", year=2023, limit=2, query="bail Section 37 NDPS"))

    assert len(refs) == 2
    assert route.called
    req_url = str(route.calls.last.request.url)
    assert "formInput" in req_url
    assert "Section" in req_url
    client.close()

