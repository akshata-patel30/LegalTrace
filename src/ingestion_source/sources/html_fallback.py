"""High-performance polite HTML scraper with robots.txt enforcement, Cloudflare challenge detection, and drift detection."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Iterator, Optional
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup
import httpx

from ingestion_source.config import Settings, get_settings
from ingestion_source.models import (
    DocRef,
    FetchFailureType,
    FetchResult,
    SourceType,
)
from ingestion_source.rate_limiter import RobotsTxtManager, TokenBucketLimiter
import asyncio
import concurrent.futures
import json

logger = logging.getLogger(__name__)


def _is_in_asyncio_loop() -> bool:
    """Check if current execution context has an active or running asyncio event loop."""
    try:
        loop = asyncio.get_running_loop()
        return loop is not None and loop.is_running()
    except RuntimeError:
        return False



def parse_cookie_dict(raw_cookies: Optional[str]) -> dict[str, str]:
    """Extract a clean key-value dictionary of cookies from any format (Netscape tabs, JSON, multiline RFC)."""
    if not raw_cookies or not str(raw_cookies).strip():
        return {}

    raw = str(raw_cookies).strip().strip('"').strip("'")
    parsed_pairs: list[tuple[str, str]] = []

    # 1. Try parsing as JSON list: [{"name": "...", "value": "..."}, ...]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        parsed_pairs.append((str(item["name"]).strip(), str(item["value"]).strip()))
        except Exception:
            pass

    # 2. Process lines (handles Netscape format, curl format, or multiline)
    if not parsed_pairs:
        lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]
        for line in lines:
            # Check Netscape format (tab-separated: domain, flag, path, secure, expiration, name, value)
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 7:
                    name = parts[5].strip()
                    val = parts[6].strip()
                    if name:
                        parsed_pairs.append((name, val))
                elif len(parts) == 2:
                    parsed_pairs.append((parts[0].strip(), parts[1].strip()))
            elif "=" in line:
                for sp in line.split(";"):
                    if "=" in sp:
                        k, v = sp.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k and k.lower() not in ("path", "domain", "expires", "samesite", "httponly", "secure"):
                            parsed_pairs.append((k, v))

    cookie_dict: dict[str, str] = {}
    for k, v in parsed_pairs:
        clean_k = re.sub(r"[\r\n\t]+", "", k).strip()
        clean_v = re.sub(r"[\r\n\t]+", "", v).strip()
        if clean_k:
            cookie_dict[clean_k] = clean_v

    return cookie_dict


def normalize_cookie_header(raw_cookies: Optional[str]) -> Optional[str]:
    """Convert any cookie format (Netscape tabs, JSON, multiline, or standard) into valid HTTP Cookie header."""
    cookie_dict = parse_cookie_dict(raw_cookies)
    if not cookie_dict:
        return None
    return "; ".join(f"{k}={v}" for k, v in cookie_dict.items())


class IndianKanoonHTMLFallback:
    """Production-quality ethical scraper for Indian Kanoon public legal documents."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.Client] = None,
        rate_limiter: Optional[TokenBucketLimiter] = None,
        robots_mgr: Optional[RobotsTxtManager] = None,
        engine: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.INDIANKANOON_BASE_URL.rstrip("/")
        if engine is not None:
            self.engine = engine
        elif http_client is not None:
            self.engine = "httpx"
        else:
            self.engine = getattr(self.settings, "SCRAPER_ENGINE", "scrapling")

        self._scrapling_session = None
        self._scrapling_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._session_lock = threading.Lock()

        # Robots.txt compliance manager
        self.robots_mgr = robots_mgr or RobotsTxtManager(
            base_url=self.base_url,
            user_agent=self.settings.USER_AGENT,
            cache_dir=self.settings.INTAKE_OUTPUT_PATH / "job_state",
        )

        # Determine effective delay adhering strictly to robots.txt crawl-delay
        crawl_delay = self.robots_mgr.get_crawl_delay() or self.settings.SCRAPE_MIN_DELAY_SECONDS
        effective_min = max(crawl_delay, self.settings.SCRAPE_MIN_DELAY_SECONDS)
        effective_max = max(effective_min + 2.0, self.settings.SCRAPE_MAX_DELAY_SECONDS)

        # Shared rate limiter
        self.rate_limiter = rate_limiter or TokenBucketLimiter(
            rate_rps=1.0 / max(0.5, effective_min),
            capacity=1.0,
            min_delay_seconds=effective_min,
            max_delay_seconds=effective_max,
            backoff_penalty_seconds=self.settings.ADAPTIVE_BACKOFF_SECONDS,
        )

        # Build HTTP headers with dynamic Chrome version match and stateful CookieJar
        ua = self.settings.USER_AGENT.strip("'\"").strip()
        chrome_ver_match = re.search(r"Chrome/(\d+)", ua)
        chrome_major = chrome_ver_match.group(1) if chrome_ver_match else "128"

        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": f'"Chromium";v="{chrome_major}", "Not(A:Brand";v="24", "Google Chrome";v="{chrome_major}"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Referer": f"{self.base_url}/",
        }

        cookie_dict = parse_cookie_dict(self.settings.INDIANKANOON_COOKIES)

        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            cookies=cookie_dict if cookie_dict else None,
            timeout=self.settings.HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._owns_client = http_client is None
        self.last_access_control_encountered = False

    def _run_in_scrapling_thread(self, func, *args, **kwargs):
        """Execute a callable inside a dedicated single-threaded worker with an isolated event loop."""
        def _target():
            try:
                import asyncio
                asyncio.set_event_loop(None)
            except Exception:
                pass
            return func(*args, **kwargs)

        with self._session_lock:
            if self._scrapling_executor is None:
                self._scrapling_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="scrapling_worker",
                )
        return self._scrapling_executor.submit(_target).result()

    def _init_scrapling_session(self):
        """Must be executed inside the isolated Scrapling worker thread."""
        if self._scrapling_session is None:
            from scrapling.fetchers import StealthySession

            logger.info("Initializing Scrapling dynamic stealth browser session in isolated worker thread...")
            cookie_dict = parse_cookie_dict(self.settings.INDIANKANOON_COOKIES)
            cookies = [
                {"name": k, "value": v, "domain": ".indiankanoon.org", "path": "/"}
                for k, v in cookie_dict.items()
            ] if cookie_dict else None

            session_kwargs = {
                "headless": True,
                "solve_cloudflare": True,
            }
            if cookies:
                session_kwargs["cookies"] = cookies

            self._scrapling_session = StealthySession(**session_kwargs)
            self._scrapling_session.start()
        return self._scrapling_session

    def _scrapling_fetch(self, url: str) -> tuple[int, str]:
        """Fetch URL using Scrapling pinned to the isolated worker thread."""
        def _fetch():
            session = self._init_scrapling_session()
            resp = session.fetch(url)
            return resp.status, str(resp.html_content)

        return self._run_in_scrapling_thread(_fetch)

    def _fetch_page(self, path_or_url: str) -> tuple[int, str]:
        """Fetch URL or relative path using the configured engine (scrapling or httpx)."""
        full_url = urljoin(self.base_url + "/", path_or_url.lstrip("/"))
        rel_path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"

        if self.engine == "scrapling":
            try:
                return self._scrapling_fetch(full_url)
            except Exception as e:
                logger.warning("Scrapling fetch failed (%s). Falling back to HTTPX.", e)
                resp = self._client.get(rel_path)
                return resp.status_code, resp.text
        else:
            resp = self._client.get(rel_path)
            return resp.status_code, resp.text


    def get_source_type(self) -> SourceType:
        return SourceType.SCRAPE

    def discover(
        self,
        court: str,
        year: int,
        month: Optional[int] = None,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Iterator[DocRef]:
        """Discover documents by targeted keyword search or by traversing the /browse/ hierarchy."""
        if query and query.strip():
            yield from self._discover_by_search(
                query=query.strip(),
                court=court,
                year=year,
                limit=limit,
            )
            return

        court_slug = self._court_to_slug(court)
        base_browse_path = f"/browse/{court_slug}/{year}/"
        if month:
            base_browse_path += f"{month}/"

        browse_url = urljoin(self.base_url + "/", base_browse_path.lstrip("/"))

        if not self.robots_mgr.is_allowed(browse_url):
            logger.warning("Discovery path %s disallowed by robots.txt", browse_url)
            return

        self.rate_limiter.acquire()
        yielded = 0
        self.last_access_control_encountered = False

        try:
            status_code, text = self._fetch_page(base_browse_path)
            if status_code in (401, 403) or "challenges.cloudflare.com" in text or "<title>Just a moment...</title>" in text:
                logger.warning(
                    "Browse root %s returned status %d with Cloudflare challenge. Obeying access controls without bypass.",
                    base_browse_path,
                    status_code,
                )
                self.rate_limiter.trigger_backoff()
                self.last_access_control_encountered = True
                return

            if status_code != 200:
                logger.warning("Browse root %s returned status %d", base_browse_path, status_code)
                return

            soup = BeautifulSoup(text, "html.parser")

            # Check if this page contains sub-month links (e.g. /browse/delhi/2023/1/)
            month_links = soup.find_all("a", href=re.compile(rf"^/browse/{re.escape(court_slug)}/{year}/\d+/"))
            sub_paths: list[str] = []
            if month_links and not month:
                for ml in month_links:
                    href = ml.get("href", "")
                    if href and href not in sub_paths:
                        sub_paths.append(href)
            else:
                sub_paths = [base_browse_path]

            # Iterate through available browse buckets
            for path in sub_paths:
                current_page = 0
                while True:
                    page_path = path
                    if current_page > 0:
                        page_path = f"{path}?pagenum={current_page}"

                    page_url = urljoin(self.base_url + "/", page_path.lstrip("/"))
                    if not self.robots_mgr.is_allowed(page_url):
                        break

                    self.rate_limiter.acquire()
                    page_status, page_text = self._fetch_page(page_path)
                    if page_status != 200:
                        break

                    page_soup = BeautifulSoup(page_text, "html.parser")
                    doc_links = page_soup.find_all("a", href=re.compile(r"^/(?:doc|docfragment)/(\d+)/"))
                    if not doc_links:
                        break

                    batch_yielded = 0
                    for link in doc_links:
                        href = link.get("href", "")
                        m = re.search(r"/(?:doc|docfragment)/(\d+)/", href)
                        if not m:
                            continue

                        doc_id = m.group(1)
                        title = link.get_text(strip=True) or f"Document {doc_id}"
                        full_doc_url = f"{self.base_url}/doc/{doc_id}/"

                        ref = DocRef(
                            doc_id=doc_id,
                            court=court,
                            court_slug=court_slug,
                            year=year,
                            month=month,
                            title=title,
                            source_url=full_doc_url,
                            doctype="judgment",
                        )
                        yield ref
                        yielded += 1
                        batch_yielded += 1

                        if limit and yielded >= limit:
                            return

                    # Stop if no new documents or standard page size exhausted
                    if batch_yielded == 0 or len(doc_links) < 10:
                        break
                    current_page += 1
                    if current_page >= 100:  # Safety boundary
                        break

        except Exception as e:
            logger.error("Error discovering documents on %s: %s", browse_url, e)

    def _discover_by_search(
        self,
        query: str,
        court: str,
        year: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterator[DocRef]:
        """Discover documents matching a keyword query via Indian Kanoon /search/?formInput=... endpoint."""
        court_slug = self._court_to_slug(court)

        # Build query string with native Indian Kanoon Lucene filters if not explicitly included
        query_clean = query.strip()
        parts = [query_clean]
        if court_slug and "doctypes:" not in query_clean.lower():
            parts.append(f"doctypes:{court_slug}")
        if year and "fromdate:" not in query_clean.lower() and "todate:" not in query_clean.lower():
            parts.append(f"fromdate:1-1-{year} todate:31-12-{year}")

        full_query = " ".join(parts)
        yielded = 0
        current_page = 0
        max_pages = min(50, ((limit // 10) + 2) if limit else 50)
        self.last_access_control_encountered = False

        while current_page < max_pages:
            search_path = f"/search/?formInput={quote_plus(full_query)}"
            if current_page > 0:
                search_path += f"&pagenum={current_page}"

            self.rate_limiter.acquire()
            try:
                status_code, text = self._fetch_page(search_path)
            except Exception as e:
                logger.error("Error fetching search page %s: %s", search_path, e)
                break

            if status_code in (401, 403) or "challenges.cloudflare.com" in text or "<title>Just a moment...</title>" in text:
                logger.warning(
                    "Search query '%s' returned status %d / Cloudflare challenge. Obeying access controls.",
                    full_query,
                    status_code,
                )
                self.rate_limiter.trigger_backoff()
                self.last_access_control_encountered = True
                return

            if status_code != 200:
                logger.warning("Search path %s returned status %d", search_path, status_code)
                break

            soup = BeautifulSoup(text, "html.parser")
            batch_yielded = 0

            # 1. Primary selector: <article class="result"> or <div class="result"> containers
            results = soup.find_all(["article", "div"], class_="result")
            if results:
                for res in results:
                    title_elem = res.find(class_="result_title")
                    a_tag = title_elem.find("a") if title_elem else None
                    if not a_tag:
                        a_tag = res.find("a", href=re.compile(r"^/(?:doc|docfragment)/(\d+)/"))
                    if not a_tag:
                        continue
                    m = re.search(r"^/(?:doc|docfragment)/(\d+)/", a_tag.get("href", ""))
                    if not m:
                        continue
                    doc_id = m.group(1)
                    title = a_tag.get_text(strip=True) or f"Document {doc_id}"

                    headline_div = res.find(class_="headline")
                    headline = headline_div.get_text(" ", strip=True) if headline_div else None

                    yield DocRef(
                        doc_id=doc_id,
                        court=court,
                        court_slug=court_slug,
                        year=year or 2023,
                        title=title,
                        source_url=f"{self.base_url}/doc/{doc_id}/",
                        doctype="judgment",
                        headline=headline,
                    )
                    yielded += 1
                    batch_yielded += 1
                    if limit and yielded >= limit:
                        return
            else:
                # 2. Resilient fallback: Any /doc/(\d+)/ links on page
                doc_links = soup.find_all("a", href=re.compile(r"^/doc/(\d+)/"))
                seen_ids = set()
                for link in doc_links:
                    href = link.get("href", "")
                    m = re.search(r"^/doc/(\d+)/", href)
                    if not m:
                        continue
                    doc_id = m.group(1)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    title = link.get_text(strip=True) or f"Document {doc_id}"

                    yield DocRef(
                        doc_id=doc_id,
                        court=court,
                        court_slug=court_slug,
                        year=year or 2023,
                        title=title,
                        source_url=f"{self.base_url}/doc/{doc_id}/",
                        doctype="judgment",
                    )
                    yielded += 1
                    batch_yielded += 1
                    if limit and yielded >= limit:
                        return

            # Indian Kanoon serves 10 results per page. If 0 or <10 returned, we reached end of results.
            if batch_yielded == 0 or batch_yielded < 10:
                break

            current_page += 1

    def fetch(self, ref: DocRef) -> FetchResult:
        """Fetch raw HTML for a judgment with robots compliance, challenge detection, and drift guards."""
        doc_path = f"/doc/{ref.doc_id}/"
        doc_url = urljoin(self.base_url + "/", doc_path.lstrip("/"))

        # 1. Obey robots.txt
        if not self.robots_mgr.is_allowed(doc_url):
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.SCRAPE,
                failure_type=FetchFailureType.DISALLOWED_BY_ROBOTS,
                error_message=f"Fetching {doc_url} is disallowed by Indian Kanoon robots.txt",
            )

        # 2. Acquire token-bucket slot
        self.rate_limiter.acquire()

        start_time = time.perf_counter()
        try:
            if self.engine == "scrapling":
                try:
                    status_code, html_content = self._scrapling_fetch(doc_url)
                except Exception as e:
                    logger.warning("Scrapling fetch failed for doc %s (%s). Falling back to HTTPX.", ref.doc_id, e)
                    resp = self._client.get(doc_path)
                    status_code = resp.status_code
                    html_content = resp.text
            else:
                resp = self._client.get(doc_path)
                status_code = resp.status_code
                html_content = resp.text

            latency = (time.perf_counter() - start_time) * 1000.0

            # 3. Detect Cloudflare challenge / Turnstile bot gate
            if "challenges.cloudflare.com" in html_content or "<title>Just a moment...</title>" in html_content:
                self.rate_limiter.trigger_backoff()
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=status_code,
                    failure_type=FetchFailureType.CHALLENGE_DETECTED,
                    error_message="Cloudflare bot protection challenge detected. Configure INDIANKANOON_COOKIES or residential proxy.",
                    latency_ms=latency,
                )

            # 4. Detect non-existent documents (Indian Kanoon returns 200 with err_msg)
            if "Act/Judgment not found" in html_content or 'class="err_msg"' in html_content:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=404,
                    failure_type=FetchFailureType.NOT_FOUND,
                    error_message=f"Document {ref.doc_id} not found on Indian Kanoon.",
                    latency_ms=latency,
                )

            if status_code == 200:
                drift_detected, reason = self._detect_structural_drift(html_content)
                if drift_detected:
                    logger.warning("Structural drift for doc %s: %s", ref.doc_id, reason)
                    return FetchResult(
                        ref=ref,
                        success=False,
                        source=SourceType.SCRAPE,
                        raw_content=html_content,
                        status_code=200,
                        failure_type=FetchFailureType.STRUCTURAL_DRIFT,
                        error_message=f"DOM structural drift: {reason}",
                        latency_ms=latency,
                    )

                self.rate_limiter.report_success()
                return FetchResult(
                    ref=ref,
                    success=True,
                    source=SourceType.SCRAPE,
                    raw_content=html_content,
                    status_code=200,
                    latency_ms=latency,
                )

            elif status_code == 404:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=404,
                    failure_type=FetchFailureType.NOT_FOUND,
                    error_message=f"Document {ref.doc_id} not found on Indian Kanoon.",
                    latency_ms=latency,
                )

            elif status_code == 429:
                self.rate_limiter.trigger_backoff()
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=429,
                    failure_type=FetchFailureType.RATE_LIMITED,
                    error_message="HTTP 429 Too Many Requests received from Indian Kanoon.",
                    latency_ms=latency,
                )

            elif status_code in (401, 403):
                self.rate_limiter.trigger_backoff()
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=status_code,
                    failure_type=FetchFailureType.FORBIDDEN,
                    error_message=f"HTTP {status_code} Forbidden/Access Denied.",
                    latency_ms=latency,
                )

            else:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.SCRAPE,
                    status_code=status_code,
                    failure_type=FetchFailureType.TRANSIENT,
                    error_message=f"Unexpected status code {status_code}",
                    latency_ms=latency,
                )

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.SCRAPE,
                failure_type=FetchFailureType.TRANSIENT,
                error_message=f"Network error during scrape: {type(e).__name__} - {e}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.SCRAPE,
                failure_type=FetchFailureType.UNKNOWN,
                error_message=f"Unexpected error scraping {ref.doc_id}: {e}",
                latency_ms=latency,
            )

    def _detect_structural_drift(self, html: str) -> tuple[bool, str]:
        """Verify presence of key Indian Kanoon DOM selectors to avoid parsing corrupted markup."""
        if not html or len(html.strip()) < 100:
            return True, "HTML response is empty or unreasonably short."

        soup = BeautifulSoup(html, "html.parser")

        # Typical Indian Kanoon judgment elements
        has_judgments = soup.find(id="judgments") or soup.find(class_="judgments")
        has_doc_content = soup.find("div", class_="doc_content") or soup.find("div", class_="judgments")
        has_doc_title = soup.find("div", class_="doc_title") or soup.find("h2", class_="doc_title") or soup.find("h1")
        has_pre_or_p = bool(soup.find_all(["pre", "p"]))

        if not (has_judgments or has_doc_content or (has_doc_title and has_pre_or_p)):
            return True, "Missing primary content containers (.judgments, .doc_content, or title with text blocks)."

        return False, ""

    def _court_to_slug(self, court: str) -> str:
        name = court.lower().strip()
        slug_map = {
            "delhi high court": "delhi",
            "delhi": "delhi",
            "supreme court of india": "supremecourt",
            "supreme court": "supremecourt",
            "bombay high court": "bombay",
            "bombay": "bombay",
            "calcutta high court": "kolkata",
            "madras high court": "chennai",
            "karnataka high court": "karnataka",
            "allahabad high court": "allahabad",
            "kerala high court": "kerala",
            "punjab and haryana high court": "punjab",
        }
        return slug_map.get(name, name.replace(" ", "_"))

    def close(self) -> None:
        if self._owns_client and self._client:
            try:
                self._client.close()
            except Exception:
                pass
        with self._session_lock:
            if self._scrapling_executor is not None:
                def _close_session():
                    if self._scrapling_session is not None:
                        try:
                            self._scrapling_session.close()
                        except Exception as e:
                            logger.debug("Error closing Scrapling session: %s", e)
                        finally:
                            self._scrapling_session = None

                try:
                    self._scrapling_executor.submit(_close_session).result(timeout=10)
                except Exception:
                    pass
                try:
                    self._scrapling_executor.shutdown(wait=False)
                except Exception:
                    pass
                self._scrapling_executor = None
