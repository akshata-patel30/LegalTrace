"""Command Line Interface for LegalTrace Indian Kanoon Acquisition Service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from ingestion_source.config import Settings, get_settings
from ingestion_source.discovery import DiscoveryEngine, normalize_court_slug
from ingestion_source.job import JobOrchestrator
from ingestion_source.models import JobReport
from ingestion_source.rate_limiter import RobotsTxtManager

app = typer.Typer(
    name="ingest",
    help="LegalTrace Indian Kanoon Data Acquisition & Ingestion CLI (Multithreaded Scraper & API)",
    add_completion=False,
)
console = Console()


def _render_report(report: JobReport) -> None:
    """Render a JobReport as a Rich summary table."""
    table = Table(title=f"Ingestion Job Report: [bold cyan]{report.job_id}[/bold cyan]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Status", f"[bold]{report.status.upper()}[/bold]")
    table.add_row("Mode", report.mode)
    table.add_row("Started At", report.started_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    table.add_row("Duration", f"{report.duration_seconds:.2f} seconds")
    table.add_row("Total Discovered", str(report.total_discovered))
    table.add_row("Successfully Stored", f"[bold green]{report.stored_count}[/bold green]")
    table.add_row("Skipped (Deduped)", str(report.skipped_count))
    table.add_row("Failed", f"[bold red]{report.failed_count}[/bold red]")
    table.add_row("API Calls Consumed", str(report.api_calls_count))

    if report.error_summary:
        err_str = ", ".join(f"{k}: {v}" for k, v in report.error_summary.items())
        table.add_row("Error Breakdown", f"[yellow]{err_str}[/yellow]")

    console.print(table)


@app.command()
def discover(
    court: str = typer.Option("Delhi High Court", "--court", "-c", help="Court name (e.g. 'Delhi High Court')"),
    year: int = typer.Option(2023, "--year", "-y", help="Target year"),
    month: Optional[int] = typer.Option(None, "--month", "-m", help="Optional target month (1-12)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max document references to discover"),
    output: Path = typer.Option(Path("discovered.jsonl"), "--output", "-o", help="Output JSONL path"),
    source: str = typer.Option("auto", "--source", "-s", help="Discovery source strategy (auto | scrape | api | seed)"),
    seed_file: Optional[Path] = typer.Option(None, "--seed-file", help="Optional path to custom seed manifest file"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="Optional custom cookies (e.g. session / clearance)"),
) -> None:
    """Enumerate available judgments without downloading full text."""
    console.print(Panel(f"Discovering judgments for [bold]{court}[/bold] (Year: {year})", style="blue"))
    settings = get_settings()
    if cookies:
        settings.INDIANKANOON_COOKIES = cookies

    engine = DiscoveryEngine(settings=settings)

    with console.status("[bold green]Discovering document references..."):
        count = engine.save_discovered_to_jsonl(
            output_path=output,
            courts=[court],
            years=[year],
            limit=limit,
            source_mode=source,
            seed_file=seed_file,
        )

    console.print(f"[bold green][OK] Successfully discovered {count} documents saved to {output}[/bold green]")


@app.command()
def fetch(
    input_file: Path = typer.Option(Path("discovered.jsonl"), "--input", "-i", help="Path to discovered.jsonl"),
    mode: str = typer.Option("scrape_only", "--mode", help="Acquisition mode (scrape_only | hybrid | api)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max documents to fetch in this run"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of concurrent worker threads"),
    delay: Optional[float] = typer.Option(None, "--delay", "-d", help="Delay in seconds between requests (e.g. 5.0 to 10.0)"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="Optional custom cookies (e.g. session / clearance)"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Scraper engine: scrapling (dynamic stealth browser) or httpx"),
) -> None:
    """Fetch and ingest full judgments from a discovered.jsonl file with multithreading and deduplication."""
    if not input_file.exists():
        console.print(f"[bold red]Error: Input file {input_file} does not exist. Run 'ingest discover' first.[/bold red]")
        raise typer.Exit(code=1)

    refs = DiscoveryEngine.load_discovered_from_jsonl(input_file)
    if not refs:
        console.print("[yellow]Input file contains no document references.[/yellow]")
        return

    settings = get_settings()
    settings.MODE = mode  # type: ignore
    settings.MAX_WORKERS = workers
    if engine:
        settings.SCRAPER_ENGINE = engine  # type: ignore
    if delay is not None:
        settings.SCRAPE_MIN_DELAY_SECONDS = max(0.5, delay)
        settings.SCRAPE_MAX_DELAY_SECONDS = max(delay + 1.0, delay * 1.5)
    if cookies:
        settings.INDIANKANOON_COOKIES = cookies

    orchestrator = JobOrchestrator(settings=settings)
    job_id = orchestrator.create_job(refs, mode=settings.MODE)

    console.print(Panel(
        f"Starting Job [bold cyan]{job_id}[/bold cyan]\n"
        f"Mode: [bold]{settings.MODE}[/bold] | Engine: [bold]{settings.SCRAPER_ENGINE}[/bold] | Workers: [bold]{workers}[/bold] | Delay: [bold]{settings.SCRAPE_MIN_DELAY_SECONDS:.1f}s[/bold] | Queue: [bold]{len(refs)} items[/bold]",
        style="green",
    ))

    def on_progress(ref, status: str) -> None:
        if status == "stored":
            console.print(f" [green]STORED[/green]   doc_id={ref.doc_id} -> judgment.json | {ref.court} | {ref.title[:45]}")
        elif status == "skipped":
            console.print(f" [yellow]SKIPPED[/yellow]  doc_id={ref.doc_id} (deduped)")
        elif status == "retrying":
            console.print(f" [cyan]WAITING[/cyan]  doc_id={ref.doc_id} (rate limit / challenge, pausing before retry...)")
        else:
            console.print(f" [red]FAILED[/red]   doc_id={ref.doc_id}")

    report = orchestrator.execute_job(job_id, limit=limit, workers=workers, on_progress=on_progress)
    _render_report(report)


@app.command()
def run(
    court: str = typer.Option("Delhi High Court", "--court", "-c", help="Court name"),
    year: int = typer.Option(2023, "--year", "-y", help="Target year"),
    mode: str = typer.Option("scrape_only", "--mode", help="Mode: scrape_only | hybrid | api"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of documents to ingest"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of concurrent worker threads"),
    delay: Optional[float] = typer.Option(None, "--delay", "-d", help="Delay in seconds between requests"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="Optional custom cookies (e.g. session / clearance)"),
) -> None:
    """Convenience command: discover and fetch in a single multithreaded run."""
    settings = get_settings()
    settings.MODE = mode  # type: ignore
    settings.MAX_WORKERS = workers
    if delay is not None:
        settings.SCRAPE_MIN_DELAY_SECONDS = max(0.5, delay)
        settings.SCRAPE_MAX_DELAY_SECONDS = max(delay + 1.0, delay * 1.5)
    if cookies:
        settings.INDIANKANOON_COOKIES = cookies

    engine = DiscoveryEngine(settings=settings)

    console.print(Panel(
        f"Automated Ingestion: [bold]{court}[/bold] ({year})\n"
        f"Mode: [bold]{mode}[/bold] | Workers: [bold]{workers}[/bold] | Target Limit: [bold]{limit}[/bold]",
        style="blue",
    ))

    with console.status("[bold green]Discovering document references..."):
        refs = list(engine.discover_refs(courts=[court], years=[year], total_limit=limit))

    if not refs:
        console.print("[yellow]No document references found matching query.[/yellow]")
        return

    orchestrator = JobOrchestrator(settings=settings)
    job_id = orchestrator.create_job(refs, mode=mode)

    def on_progress(ref, status: str) -> None:
        if status == "stored":
            console.print(f" [green]STORED[/green]  doc_id={ref.doc_id} | {ref.court} | {ref.title[:50]}")
        elif status == "skipped":
            console.print(f" [yellow]SKIPPED[/yellow] doc_id={ref.doc_id} (deduped)")
        else:
            console.print(f" [red]FAILED[/red]  doc_id={ref.doc_id}")

    report = orchestrator.execute_job(job_id, limit=limit, workers=workers, on_progress=on_progress)
    _render_report(report)


@app.command()
def resume(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Job ID to resume"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Optional limit on items to process"),
    workers: int = typer.Option(3, "--workers", "-w", help="Number of concurrent worker threads"),
) -> None:
    """Resume an interrupted or paused ingestion job with multithreading."""
    settings = get_settings()
    orchestrator = JobOrchestrator(settings=settings)

    console.print(Panel(f"Resuming Job [bold cyan]{job_id}[/bold cyan] (Workers: {workers})", style="yellow"))
    report = orchestrator.execute_job(job_id, limit=limit, workers=workers)
    _render_report(report)


@app.command()
def report(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Job ID to inspect"),
) -> None:
    """Display metrics and error summary for a job run."""
    settings = get_settings()
    orchestrator = JobOrchestrator(settings=settings)
    report_data = orchestrator.get_report(job_id)
    _render_report(report_data)


@app.command(name="check-robots")
def check_robots(
    url: str = typer.Option("https://indiankanoon.org/doc/12345/", "--url", help="Target URL to check"),
) -> None:
    """Check live/cached robots.txt compliance for Indian Kanoon."""
    settings = get_settings()
    manager = RobotsTxtManager(
        base_url=settings.INDIANKANOON_BASE_URL,
        user_agent=settings.USER_AGENT,
        cache_dir=settings.INTAKE_OUTPUT_PATH / "job_state",
    )
    with console.status("[bold green]Fetching & verifying robots.txt..."):
        manager.fetch_rules()
        allowed = manager.is_allowed(url)
        delay = manager.get_crawl_delay()

    status_str = "[bold green]ALLOWED[/bold green]" if allowed else "[bold red]DISALLOWED[/bold red]"
    console.print(f"URL: [cyan]{url}[/cyan]")
    console.print(f"Status: {status_str}")
    console.print(f"User-Agent: [white]{settings.USER_AGENT}[/white]")
    if delay is not None:
        console.print(f"Crawl-Delay: [yellow]{delay} seconds[/yellow]")


@app.command()
def restructure(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
) -> None:
    """Restructure previously acquired judgments into the new canonical judgment.json schema."""
    from ingestion_source.storage import StorageWriter

    console.print(Panel(f"Restructuring stored documents in [bold cyan]{store_dir}[/bold cyan]", style="blue"))
    writer = StorageWriter(store_dir)
    count = writer.restructure_stored_documents()
    console.print(f"[bold green][OK] Successfully upgraded {count} documents with canonical judgment.json[/bold green]")


@app.command()
def export(
    output: Path = typer.Option(Path("all_judgments.jsonl"), "--output", "-o", help="Destination JSONL path"),
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
) -> None:
    """Consolidate all downloaded judgment.json files into a single unified JSONL dataset."""
    from ingestion_source.storage import StorageWriter

    writer = StorageWriter(store_dir)
    count = writer.export_consolidated_jsonl(output)
    console.print(f"[bold green][OK] Successfully consolidated {count} judgments into {output}[/bold green]")


@app.command()
def validate(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to validate"),
) -> None:
    """Validate document integrity, checksums, and schema completeness (Step 2 of Pipeline A)."""
    from ingestion_source.validator import DocumentValidator

    validator = DocumentValidator()

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        res = validator.validate_document(doc_dir)
        status_color = "green" if res.is_valid else "red"
        console.print(Panel(f"Validation for Doc [bold cyan]{doc_id}[/bold cyan]: [{status_color}]{'PASSED' if res.is_valid else 'FAILED'}[/{status_color}]", style="blue"))
        console.print(f"  Valid: [{status_color}]{res.is_valid}[/{status_color}]")
        console.print(f"  Quality Score: [cyan]{res.quality_score:.1f}%[/cyan]")
        console.print(f"  Characters: [yellow]{res.char_count:,}[/yellow]")
        console.print(f"  Checksum Verified: [green]{res.content_hash_matched}[/green]")
        if res.issues:
            console.print("  Issues:")
            for iss in res.issues:
                col = "red" if iss.severity.value == "error" else "yellow"
                console.print(f"   - [{col}]{iss.severity.value.upper()}[/{col}] ({iss.field}): {iss.message}")
        return

    console.print(Panel(
        f"Validating Document Store in [bold cyan]{store_dir}[/bold cyan] (Step 2: Integrity & QA)",
        style="blue",
    ))

    with console.status("[bold green]Running integrity and quality checks..."):
        report = validator.validate_store(store_dir)

    table = Table(title="[bold green]LegalTrace Document Validation Report (Step 2)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Documents Checked", str(report.total_documents))
    table.add_row("Valid & Intact Documents", f"[bold green]{report.valid_documents}[/bold green]")
    table.add_row("Invalid Documents", f"[bold red]{report.invalid_documents}[/bold red]" if report.invalid_documents else "[green]0[/green]")
    table.add_row("Total Validation Errors", f"[bold red]{report.total_errors}[/bold red]" if report.total_errors else "[green]0[/green]")
    table.add_row("Total Warnings", f"[yellow]{report.total_warnings}[/yellow]")
    table.add_row("Average Quality Score", f"[bold green]{report.average_quality_score:.1f}%[/bold green]")
    table.add_row("Integrity Status", "[bold green]100% PASSING[/bold green]" if report.invalid_documents == 0 else "[bold red]FAILURES DETECTED[/bold red]")

    console.print(table)

    if report.issues_breakdown:
        issues_table = Table(title="[bold yellow]Validation Issues Breakdown[/bold yellow]")
        issues_table.add_column("Issue Type", style="bold white")
        issues_table.add_column("Count", style="yellow")
        for k, v in report.issues_breakdown.items():
            issues_table.add_row(k, str(v))
        console.print(issues_table)


@app.command()
def clean(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output: Optional[Path] = typer.Option(Path("all_200_cleaned_judgments.jsonl"), "--output", "-o", help="Destination JSONL path for cleaned records"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to clean"),
) -> None:
    """Clean, normalize, and segment legal judgments removing Kanoon noise and classifying dispositions."""
    from collections import Counter
    from ingestion_source.cleaner import LegalDocumentCleaner

    cleaner = LegalDocumentCleaner()

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        console.print(Panel(f"Cleaning single document: [bold cyan]{doc_id}[/bold cyan] ({doc_dir})", style="blue"))
        cleaned = cleaner.clean_document_dir(doc_dir)
        console.print(f"[bold green][OK] Successfully cleaned doc_id={doc_id}[/bold green]")
        console.print(f"  Paragraphs: [cyan]{len(cleaned.paragraphs)}[/cyan]")
        console.print(f"  Disposition: [magenta]{cleaned.disposition.value.upper()}[/magenta]")
        console.print(f"  Noise removed: [yellow]{cleaned.metrics.noise_removed_chars} chars ({cleaned.metrics.reduction_percentage}%)[/yellow]")
        return

    console.print(Panel(
        f"Batch Cleaning Judgments in [bold cyan]{store_dir}[/bold cyan]\n"
        f"Writing cleaned JSONL to [bold]{output}[/bold]",
        style="blue",
    ))

    with console.status("[bold green]Cleaning and segmenting judgments..."):
        cleaned_list = cleaner.clean_store(store_dir=store_dir, output_jsonl=output)

    if not cleaned_list:
        console.print("[yellow]No judgments found to clean.[/yellow]")
        return

    total_raw = sum(c.metrics.raw_char_count for c in cleaned_list)
    total_clean = sum(c.metrics.cleaned_char_count for c in cleaned_list)
    total_removed = total_raw - total_clean
    total_reduction_pct = (total_removed / total_raw * 100.0) if total_raw > 0 else 0.0
    total_paras = sum(c.metrics.paragraphs_count for c in cleaned_list)

    disp_counts = Counter(c.disposition.value for c in cleaned_list)

    table = Table(title="[bold green]LegalTrace Data Cleaning Summary (Step 3)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Judgments Processed", str(len(cleaned_list)))
    table.add_row("Total Raw Characters", f"{total_raw:,}")
    table.add_row("Total Cleaned Characters", f"{total_clean:,}")
    table.add_row("Noise Characters Removed", f"[yellow]{total_removed:,} ({total_reduction_pct:.2f}%)[/yellow]")
    table.add_row("Total Segmented Paragraphs", f"{total_paras:,}")
    table.add_row("Output Consolidated JSONL", str(output))

    console.print(table)

    disp_table = Table(title="[bold magenta]Judicial Disposition Distribution[/bold magenta]")
    disp_table.add_column("Disposition", style="bold white")
    disp_table.add_column("Count", style="green")
    disp_table.add_column("Percentage", style="yellow")

    for disp, cnt in disp_counts.most_common():
        pct = (cnt / len(cleaned_list)) * 100.0
        disp_table.add_row(disp.upper(), str(cnt), f"{pct:.1f}%")

    console.print(disp_table)
    console.print(f"[bold green][OK] Cleaning complete! Generated cleaned_text.txt & cleaned.json in each document directory.[/bold green]")


@app.command()
def enrich(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output: Optional[Path] = typer.Option(Path("all_200_enriched_judgments.jsonl"), "--output", "-o", help="Destination JSONL path for enriched metadata"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to enrich"),
) -> None:
    """Extract legal metadata, statutory provisions, citations, and counsel (Step 4 of Pipeline A)."""
    from collections import Counter
    from ingestion_source.extractor import LegalMetadataExtractor

    extractor = LegalMetadataExtractor()

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        console.print(Panel(f"Enriching single document: [bold cyan]{doc_id}[/bold cyan] ({doc_dir})", style="blue"))
        enriched = extractor.enrich_document_dir(doc_dir)
        console.print(f"[bold green][OK] Successfully enriched doc_id={doc_id}[/bold green]")
        console.print(f"  Case Type: [cyan]{enriched.case_type} ({enriched.case_type_code})[/cyan]")
        console.print(f"  Domain: [magenta]{enriched.legal_domain.value.upper()}[/magenta]")
        console.print(f"  Bench Type: [yellow]{enriched.bench_type.value.upper()}[/yellow]")
        console.print(f"  Judges: [white]{', '.join(enriched.canonical_judges) if enriched.canonical_judges else 'Unspecified'}[/white]")
        console.print(f"  Statutes Referenced: [green]{enriched.total_statutes_count}[/green]")
        for st in enriched.statutes:
            console.print(f"    • [bold]{st.act_code}[/bold] ({st.act_name}): Sections {', '.join(st.sections)}")
        console.print(f"  Citations Discovered: [cyan]{enriched.total_citations_count}[/cyan]")
        for c in enriched.citations:
            console.print(f"    • [{c.reporter}] [white]{c.citation_text}[/white]")
        return

    console.print(Panel(
        f"Batch Extracting Legal Metadata in [bold cyan]{store_dir}[/bold cyan] (Step 4)\n"
        f"Writing enriched JSONL to [bold]{output}[/bold]",
        style="blue",
    ))

    with console.status("[bold green]Extracting statutes, case types, and citations..."):
        enriched_list = extractor.enrich_store(store_dir=store_dir, output_jsonl=output)

    if not enriched_list:
        console.print("[yellow]No cleaned judgments found to enrich. Run 'ingest clean' first.[/yellow]")
        return

    total_statutes = sum(e.total_statutes_count for e in enriched_list)
    total_citations = sum(e.total_citations_count for e in enriched_list)

    domain_counts = Counter(e.legal_domain.value for e in enriched_list)
    act_counts: Counter[str] = Counter()
    judge_counts: Counter[str] = Counter()

    for e in enriched_list:
        for st in e.statutes:
            act_counts[st.act_code] += 1
        for j in e.canonical_judges:
            judge_counts[j] += 1

    table = Table(title="[bold green]LegalTrace Metadata & Entity Extraction Summary (Step 4)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Judgments Enriched", str(len(enriched_list)))
    table.add_row("Total Statute Mentions Extracted", f"{total_statutes:,}")
    table.add_row("Total Citations Discovered", f"{total_citations:,}")
    table.add_row("Unique Acts Referenced", str(len(act_counts)))
    table.add_row("Output Consolidated JSONL", str(output))

    console.print(table)

    domain_table = Table(title="[bold magenta]Legal Domain Distribution[/bold magenta]")
    domain_table.add_column("Domain", style="bold white")
    domain_table.add_column("Count", style="green")
    domain_table.add_column("Percentage", style="yellow")

    for dom, cnt in domain_counts.most_common():
        pct = (cnt / len(enriched_list)) * 100.0
        domain_table.add_row(dom.upper(), str(cnt), f"{pct:.1f}%")

    console.print(domain_table)

    acts_table = Table(title="[bold cyan]Top 10 Most Frequently Referenced Acts[/bold cyan]")
    acts_table.add_column("Statute / Code", style="bold white")
    acts_table.add_column("Document Frequency", style="green")

    for act_code, cnt in act_counts.most_common(10):
        acts_table.add_row(act_code, str(cnt))

    console.print(acts_table)
    console.print(f"[bold green][OK] Enrichment complete! Generated enriched.json in each document directory.[/bold green]")


@app.command()
def structure(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output: Optional[Path] = typer.Option(Path("all_200_structured_judgments.jsonl"), "--output", "-o", help="Destination JSONL path for structured judgments"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to structure"),
) -> None:
    """Classify judgment paragraphs into legal rhetorical roles (Step 5 of Pipeline A)."""
    from collections import Counter
    from ingestion_source.structure import LegalStructureClassifier

    classifier = LegalStructureClassifier()

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        console.print(Panel(f"Structuring paragraphs for Doc: [bold cyan]{doc_id}[/bold cyan]", style="blue"))
        judgment = classifier.structure_document_dir(doc_dir)
        console.print(f"[bold green][OK] Successfully labeled {len(judgment.paragraphs)} paragraphs for doc_id={doc_id}[/bold green]")

        role_counts = Counter(p.rhetorical_role.value for p in judgment.paragraphs)
        table = Table(title=f"Paragraph Roles for {doc_id}")
        table.add_column("Role", style="bold white")
        table.add_column("Paragraph Count", style="green")
        for r, cnt in role_counts.most_common():
            table.add_row(r.upper(), str(cnt))
        console.print(table)
        return

    console.print(Panel(
        f"Batch Labeling Rhetorical Roles in [bold cyan]{store_dir}[/bold cyan] (Step 5)\n"
        f"Writing structured JSONL to [bold]{output}[/bold]",
        style="blue",
    ))

    with console.status("[bold green]Classifying legal rhetorical roles across all paragraphs..."):
        structured_list = classifier.structure_store(store_dir=store_dir, output_jsonl=output)

    if not structured_list:
        console.print("[yellow]No cleaned judgments found to structure. Run 'ingest clean' first.[/yellow]")
        return

    total_paras = sum(len(j.paragraphs) for j in structured_list)
    all_roles: Counter[str] = Counter()
    for j in structured_list:
        for p in j.paragraphs:
            all_roles[p.rhetorical_role.value] += 1

    table = Table(title="[bold green]LegalTrace Rhetorical Role Labeling (RRL) Summary (Step 5)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Judgments Structured", str(len(structured_list)))
    table.add_row("Total Paragraphs Classified", f"{total_paras:,}")
    table.add_row("Distinct Rhetorical Roles", str(len(all_roles)))
    table.add_row("Output Consolidated JSONL", str(output))

    console.print(table)

    roles_table = Table(title="[bold magenta]Rhetorical Role Distribution across Corpus[/bold magenta]")
    roles_table.add_column("Rhetorical Role", style="bold white")
    roles_table.add_column("Paragraphs Count", style="green")
    roles_table.add_column("Corpus Percentage", style="yellow")

    for role, cnt in all_roles.most_common():
        pct = (cnt / total_paras * 100.0) if total_paras > 0 else 0.0
        roles_table.add_row(role.upper(), f"{cnt:,}", f"{pct:.1f}%")

    console.print(roles_table)
    console.print(f"[bold green][OK] Structure detection complete! Updated cleaned.json in each document directory.[/bold green]")


@app.command()
def chunk(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output: Optional[Path] = typer.Option(Path("all_200_chunks.jsonl"), "--output", "-o", help="Destination JSONL path for chunks"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to chunk"),
    min_words: int = typer.Option(150, "--min-words", help="Minimum words per chunk"),
    target_words: int = typer.Option(400, "--target-words", help="Target words per chunk"),
    max_words: int = typer.Option(800, "--max-words", help="Maximum words per chunk"),
    overlap_paras: int = typer.Option(1, "--overlap-paras", help="Paragraph overlap within identical roles"),
) -> None:
    """Segment legal judgments into role-isolated semantic chunks with context headers (Step 6 of Pipeline A)."""
    from ingestion_source.chunker import LegalSemanticChunker

    chunker = LegalSemanticChunker(
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
        overlap_paras=overlap_paras,
    )

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        console.print(Panel(f"Chunking single document: [bold cyan]{doc_id}[/bold cyan] ({doc_dir})", style="blue"))
        chunks = chunker.chunk_document_dir(doc_dir)
        console.print(f"[bold green][OK] Generated {len(chunks)} chunks for doc_id={doc_id}[/bold green]")

        table = Table(title=f"Sample Chunks for {doc_id}")
        table.add_column("Chunk ID", style="bold cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Words", style="green")
        table.add_column("Paras", style="yellow")
        table.add_column("Statutes / Citations", style="white")

        for c in chunks[:10]:
            entities = []
            if c.statute_mentions:
                entities.append(f"Statutes: {len(c.statute_mentions)}")
            if c.citations:
                entities.append(f"Cites: {len(c.citations)}")
            table.add_row(
                c.chunk_id,
                c.rhetorical_role.value.upper(),
                str(c.word_count),
                f"#{','.join(str(p) for p in c.para_ids[:3])}{'...' if len(c.para_ids) > 3 else ''}",
                ", ".join(entities) or "None",
            )
        console.print(table)
        return

    console.print(Panel(
        f"Batch Semantic Chunking in [bold cyan]{store_dir}[/bold cyan] (Step 6: Legal RAG Chunking)\n"
        f"Output JSONL: [bold]{output}[/bold] | Target: [bold]{target_words} words[/bold] (Max: {max_words})",
        style="blue",
    ))

    with console.status("[bold green]Generating role-isolated semantic chunks..."):
        chunks = chunker.chunk_store(store_dir=store_dir, output_jsonl=output)

    if not chunks:
        console.print("[yellow]No judgments found to chunk. Run 'ingest structure' first.[/yellow]")
        return

    # Count distinct docs
    unique_docs = len(set(c.doc_id for c in chunks))
    report = chunker.generate_report(chunks, total_docs=unique_docs)

    statute_chunks_cnt = sum(1 for c in chunks if c.statute_mentions)
    citation_chunks_cnt = sum(1 for c in chunks if c.citations)
    quote_chunks_cnt = sum(1 for c in chunks if c.is_quote_block)

    table = Table(title="[bold green]LegalTrace Semantic Chunking Summary (Step 6)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Judgments Processed", str(report.total_documents))
    table.add_row("Total Semantic Chunks Generated", f"[bold green]{report.total_chunks:,}[/bold green]")
    table.add_row("Average Chunks per Judgment", f"{report.average_chunks_per_doc:.1f}")
    table.add_row("Average Words per Chunk", f"{report.average_words_per_chunk:.1f}")
    table.add_row("Average Characters per Chunk", f"{report.average_chars_per_chunk:,.1f}")
    table.add_row("Chunks with Statutory Anchors", f"{statute_chunks_cnt:,} ({(statute_chunks_cnt / report.total_chunks * 100):.1f}%)")
    table.add_row("Chunks with Citation Anchors", f"{citation_chunks_cnt:,} ({(citation_chunks_cnt / report.total_chunks * 100):.1f}%)")
    table.add_row("Chunks with Quoted Passages", f"{quote_chunks_cnt:,} ({(quote_chunks_cnt / report.total_chunks * 100):.1f}%)")
    table.add_row("Output Consolidated JSONL", str(output))

    console.print(table)

    role_table = Table(title="[bold magenta]Chunk Distribution by Rhetorical Role[/bold magenta]")
    role_table.add_column("Rhetorical Role", style="bold white")
    role_table.add_column("Chunk Count", style="green")
    role_table.add_column("Corpus Percentage", style="yellow")

    for r_role, cnt in report.role_distribution.items():
        pct = (cnt / report.total_chunks * 100.0) if report.total_chunks > 0 else 0.0
        role_table.add_row(r_role.upper(), f"{cnt:,}", f"{pct:.1f}%")

    console.print(role_table)
    console.print(f"[bold green][OK] Chunking complete! Generated chunks.json in each document directory and {output}[/bold green]")


@app.command()
def graph(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output_edges: Optional[Path] = typer.Option(Path("citation_edges.jsonl"), "--output-edges", help="Destination JSONL path for edges"),
    output_nodes: Optional[Path] = typer.Option(Path("precedent_nodes.jsonl"), "--output-nodes", help="Destination JSONL path for precedent nodes"),
    output_graph_json: Optional[Path] = typer.Option(Path("citation_graph.json"), "--output-graph", help="Destination D3 JSON graph path"),
    output_graphml: Optional[Path] = typer.Option(Path("citation_graph.graphml"), "--output-graphml", help="Destination GraphML XML path"),
) -> None:
    """Build directed precedent citation graph, cross-document links, and authority rankings (Step 7 of Pipeline A)."""
    from ingestion_source.graph import CitationGraphBuilder

    console.print(Panel(
        f"Building Precedent Citation Graph from [bold cyan]{store_dir}[/bold cyan] (Step 7: Network & Linker)",
        style="blue",
    ))

    builder = CitationGraphBuilder(store_dir=store_dir)

    with console.status("[bold green]Constructing citation graph & resolving cross-document links..."):
        edges, nodes, summary = builder.build_graph_from_store(
            store_dir=store_dir,
            output_edges_jsonl=output_edges,
            output_nodes_jsonl=output_nodes,
            output_graph_json=output_graph_json,
            output_graphml=output_graphml,
        )

    table = Table(title="[bold green]LegalTrace Precedent Citation Network Summary (Step 7)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Citing Judgments in Network", str(summary.total_citing_documents))
    table.add_row("Total Directed Citation Edges", f"[bold green]{summary.total_edges:,}[/bold green]")
    table.add_row("Distinct Cited Precedent Authorities", f"[bold green]{summary.total_precedent_nodes:,}[/bold green]")
    table.add_row("Internal Cross-Corpus Matches", f"[yellow]{summary.internal_edges_count:,}[/yellow]")
    table.add_row("External Authorities Cited", f"{summary.external_edges_count:,}")
    table.add_row("Edges JSONL", str(output_edges))
    table.add_row("Nodes JSONL", str(output_nodes))
    table.add_row("D3.js Network JSON", str(output_graph_json))
    table.add_row("GraphML XML", str(output_graphml))

    console.print(table)

    top_table = Table(title="[bold cyan]Top 15 Most Cited Landmark Precedents (Authority In-Degree)[/bold cyan]")
    top_table.add_column("Rank", style="bold yellow")
    top_table.add_column("Precedent Citation", style="bold white")
    top_table.add_column("Reporter", style="green")
    top_table.add_column("Case Title / Landmark Name", style="magenta")
    top_table.add_column("In-Degree (Citations)", style="cyan")

    for i, prec in enumerate(summary.top_precedents[:15], 1):
        top_table.add_row(
            str(i),
            prec["citation"],
            prec["reporter"],
            prec["title"],
            str(prec["in_degree"]),
        )
    console.print(top_table)

    treat_table = Table(title="[bold magenta]Judicial Treatment Distribution of Precedents[/bold magenta]")
    treat_table.add_column("Treatment", style="bold white")
    treat_table.add_column("Count", style="green")
    treat_table.add_column("Percentage", style="yellow")

    for treat, cnt in summary.treatment_breakdown.items():
        pct = (cnt / summary.total_edges * 100.0) if summary.total_edges > 0 else 0.0
        treat_table.add_row(treat.upper(), f"{cnt:,}", f"{pct:.1f}%")
    console.print(treat_table)

    console.print(f"[bold green][OK] Precedent network generated! Saved edges, nodes, D3 graph, and GraphML.[/bold green]")


@app.command()
def headnote(
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
    output: Optional[Path] = typer.Option(Path("all_200_headnotes.jsonl"), "--output", "-o", help="Destination JSONL path for headnotes"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Optional single document ID to generate and display headnote"),
) -> None:
    """Generate structured publication-grade legal headnotes and executive summaries (Step 8 of Pipeline A)."""
    from ingestion_source.summarizer import LegalHeadnoteGenerator

    generator = LegalHeadnoteGenerator()

    if doc_id:
        target_dirs = [d for d in store_dir.glob(f"**/{doc_id}") if d.is_dir()]
        if not target_dirs:
            console.print(f"[bold red]Document {doc_id} not found in {store_dir}[/bold red]")
            raise typer.Exit(code=1)
        doc_dir = target_dirs[0]
        hn = generator.generate_for_document_dir(doc_dir)
        console.print(f"[bold green][OK] Generated headnote for doc_id={doc_id}[/bold green]\n")
        console.print(hn.formatted_headnote)
        return

    console.print(Panel(
        f"Batch Generating Legal Headnotes in [bold cyan]{store_dir}[/bold cyan] (Step 8: Automated Summarization)\n"
        f"Writing consolidated JSONL to [bold]{output}[/bold]",
        style="blue",
    ))

    with console.status("[bold green]Synthesizing facts, legal issues, ratio decidendi, and holdings..."):
        headnotes = generator.generate_for_store(store_dir=store_dir, output_jsonl=output)

    if not headnotes:
        console.print("[yellow]No judgments found to summarize. Run 'ingest clean' first.[/yellow]")
        return

    report = generator.generate_report(headnotes)

    table = Table(title="[bold green]LegalTrace Automated Headnote Generation Summary (Step 8)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Judgments Processed", str(report.total_documents))
    table.add_row("Headnotes Generated", f"[bold green]{report.headnotes_generated}[/bold green]")
    table.add_row("Average Words per Headnote", f"{report.average_headnote_words:.1f}")
    table.add_row("Judgments with Synthesized Facts", f"{report.documents_with_facts} ({(report.documents_with_facts / report.total_documents * 100):.1f}%)")
    table.add_row("Judgments with Formulated Issues", f"{report.documents_with_issues} ({(report.documents_with_issues / report.total_documents * 100):.1f}%)")
    table.add_row("Judgments with Ratio Decidendi", f"{report.documents_with_ratio} ({(report.documents_with_ratio / report.total_documents * 100):.1f}%)")
    table.add_row("Judgments with Operative Holdings", f"{report.documents_with_holding} ({(report.documents_with_holding / report.total_documents * 100):.1f}%)")
    table.add_row("Output Consolidated JSONL", str(output))

    console.print(table)
    console.print(f"[bold green][OK] Headnotes complete! Created headnote.json & headnote.txt in each document directory.[/bold green]")


@app.command(name="index")
def index_cmd(
    input_chunks: Path = typer.Option(Path("all_200_chunks.jsonl"), "--input", "-i", help="Path to chunks JSONL"),
    index_dir: Path = typer.Option(Path("search_index"), "--index-dir", help="Directory to save persistent index"),
    embedder_type: str = typer.Option("dense", "--embedder", "-e", help="Embedder type: 'dense' or 'transformer'"),
) -> None:
    """Build persistent hybrid vector (dense + BM25) search index across legal chunks (Step 9 of Pipeline A)."""
    from ingestion_source.models import LegalChunk
    from ingestion_source.search import HybridLegalSearchEngine

    if not input_chunks.exists():
        console.print(f"[bold red]Input chunks file {input_chunks} does not exist. Run 'ingest chunk' first.[/bold red]")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"Building Hybrid Search Index (Dense Vectors + BM25 Lexical)\n"
        f"Source: [bold cyan]{input_chunks}[/bold cyan] -> Index Dir: [bold]{index_dir}[/bold]\n"
        f"Embedder Model: [bold yellow]{embedder_type}[/bold yellow]",
        style="blue",
    ))

    with console.status(f"[bold green]Loading chunks and building {embedder_type} vector + BM25 indices..."):
        chunks = []
        with open(input_chunks, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(LegalChunk(**json.loads(line)))

        engine = HybridLegalSearchEngine(dim=384, embedder_type=embedder_type)
        engine.index_chunks(chunks)
        engine.save_index(index_dir)

    table = Table(title="[bold green]LegalTrace Search Index Created (Step 9)[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total Indexed Chunks", f"[bold green]{len(chunks):,}[/bold green]")
    table.add_row("Embedding Engine", f"[bold yellow]{embedder_type.upper()}[/bold yellow]")
    table.add_row("Dense Embedding Dimension", "384")
    table.add_row("BM25 Unique Vocabulary Size", f"{len(engine.bm25.idf):,}")
    table.add_row("Average Document Token Length", f"{engine.bm25.avgdl:.1f} tokens")
    table.add_row("Index Directory", str(index_dir))

    console.print(table)
    console.print(f"[bold green][OK] Index ready for high-speed hybrid search![/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (e.g. 'commercial quantity NDPS bail conditions')"),
    index_dir: Path = typer.Option(Path("search_index"), "--index-dir", help="Path to search index directory"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid | dense | bm25"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Optional filter by rhetorical role (e.g. operative, analysis, facts)"),
    court: Optional[str] = typer.Option(None, "--court", "-c", help="Optional filter by court name"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Optional filter by decision year"),
) -> None:
    """Query the legal corpus using dense vector embeddings, BM25 lexical, or RRF hybrid search."""
    from ingestion_source.search import HybridLegalSearchEngine

    if not index_dir.exists():
        console.print(f"[bold red]Index directory {index_dir} not found. Run 'ingest index' first.[/bold red]")
        raise typer.Exit(code=1)

    engine = HybridLegalSearchEngine(dim=384)
    engine.load_index(index_dir)

    resp = engine.search(
        query=query,
        mode=mode,
        top_k=top_k,
        filter_role=role,
        filter_court=court,
        filter_year=year,
    )

    console.print(Panel(
        f"Query: [bold cyan]\"{query}\"[/bold cyan]\n"
        f"Mode: [bold]{mode.upper()}[/bold] | Latency: [bold green]{resp.latency_ms} ms[/bold green] | Matches: [bold]{len(resp.results)}[/bold] (Indexed: {resp.total_indexed:,})",
        style="blue",
    ))

    if not resp.results:
        console.print("[yellow]No matching chunks found for query.[/yellow]")
        return

    table = Table(title=f"Search Results (Top {len(resp.results)})")
    table.add_column("#", style="bold yellow", width=3)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Role", style="magenta", width=18)
    table.add_column("Case & Citation", style="bold white", width=30)
    table.add_column("Matching Text Snippet", style="white")

    for item in resp.results:
        entities = []
        if item.statute_mentions:
            entities.append(f"Statutes: {len(item.statute_mentions)}")
        if item.citations:
            entities.append(f"Cites: {len(item.citations)}")
        ent_str = f" ({', '.join(entities)})" if entities else ""

        table.add_row(
            str(item.rank),
            f"{item.score:.4f}",
            item.rhetorical_role.value.upper(),
            f"{item.case_title[:28]}...\nDoc: {item.doc_id}{ent_str}",
            item.text_snippet,
        )

    console.print(table)


@app.command(name="benchmark")
def benchmark_cmd(
    index_dir: Path = typer.Option(Path("search_index"), "--index-dir", help="Directory containing the hybrid search index"),
    output: Path = typer.Option(Path("evaluation_report.json"), "--output", "-o", help="Destination JSON report path"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Evaluation ranking depth K (default 10)"),
) -> None:
    """Run empirical legal retrieval benchmark comparing BM25, Dense Vector, and Hybrid RRF (Step 10 of Pipeline A)."""
    from ingestion_source.evaluation import LegalBenchmarkEvaluator, build_default_benchmark

    if not index_dir.exists():
        console.print(f"[bold red]Index directory {index_dir} not found. Run 'ingest index' first.[/bold red]")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"Evaluating Legal Retrieval Quality across [bold cyan]20 Benchmark Queries[/bold cyan] (Step 10: Legal Evaluation Benchmark)\n"
        f"Comparing [bold]BM25 (Sparse)[/bold], [bold]Dense Embeddings (384-d)[/bold], and [bold]Hybrid RRF[/bold] (Depth K={top_k})",
        style="blue",
    ))

    with console.status("[bold green]Executing evaluation queries across retrieval modes..."):
        evaluator = LegalBenchmarkEvaluator(index_dir=index_dir)
        benchmark = build_default_benchmark()
        report = evaluator.run_benchmark(benchmark=benchmark, top_k=top_k)

        # Also collect per-query comparison rows for detailed analysis
        per_query_rows = []
        for q in benchmark:
            for mode in ["bm25", "dense", "hybrid"]:
                m = evaluator.evaluate_query(q, mode=mode, top_k=top_k)
                per_query_rows.append({
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "category": q.category.value,
                    "mode": mode,
                    "first_relevant_rank": int(m.first_relevant_rank) if m.first_relevant_rank is not None else -1,
                    "mrr": float(m.mrr),
                    "precision_at_1": float(m.precision_at_k.get(1, 0.0)),
                    "precision_at_3": float(m.precision_at_k.get(3, 0.0)),
                    "precision_at_5": float(m.precision_at_k.get(5, 0.0)),
                    "precision_at_10": float(m.precision_at_k.get(10, 0.0)),
                    "recall_at_1": float(m.recall_at_k.get(1, 0.0)),
                    "recall_at_3": float(m.recall_at_k.get(3, 0.0)),
                    "recall_at_5": float(m.recall_at_k.get(5, 0.0)),
                    "recall_at_10": float(m.recall_at_k.get(10, 0.0)),
                    "hit_at_1": float(m.hit_at_k.get(1, 0.0)),
                    "hit_at_3": float(m.hit_at_k.get(3, 0.0)),
                    "hit_at_5": float(m.hit_at_k.get(5, 0.0)),
                    "hit_at_10": float(m.hit_at_k.get(10, 0.0)),
                    "ndcg_at_5": float(m.ndcg_at_k.get(5, 0.0)),
                    "ndcg_at_10": float(m.ndcg_at_k.get(10, 0.0)),
                    "latency_ms": float(m.latency_ms),
                })

    # Save to JSON
    report_dict = report.model_dump()
    report_dict["per_query_comparison"] = per_query_rows
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    # Display comparison table
    table = Table(title="[bold green]LegalTrace Retrieval Benchmark Evaluation (Step 10)[/bold green]")
    table.add_column("Evaluation Metric", style="bold white", width=26)
    table.add_column("BM25 (Sparse)", style="cyan", justify="right")
    table.add_column("Dense Vector", style="magenta", justify="right")
    table.add_column("Hybrid (RRF)", style="bold green", justify="right")
    table.add_column("Hybrid vs BM25 Lift", style="yellow", justify="right")

    bm = report.bm25_metrics
    dm = report.dense_metrics
    hm = report.hybrid_metrics

    metrics = [
        ("Mean Reciprocal Rank (MRR)", bm.mrr, dm.mrr, hm.mrr),
        ("Hit Rate @ 1", bm.hit_at_1, dm.hit_at_1, hm.hit_at_1),
        ("Hit Rate @ 3", bm.hit_at_3, dm.hit_at_3, hm.hit_at_3),
        ("Hit Rate @ 5", bm.hit_at_5, dm.hit_at_5, hm.hit_at_5),
        ("Hit Rate @ 10", bm.hit_at_10, dm.hit_at_10, hm.hit_at_10),
        ("Precision @ 1", bm.precision_at_1, dm.precision_at_1, hm.precision_at_1),
        ("Precision @ 3", bm.precision_at_3, dm.precision_at_3, hm.precision_at_3),
        ("Precision @ 5", bm.precision_at_5, dm.precision_at_5, hm.precision_at_5),
        ("Recall @ 5", bm.recall_at_5, dm.recall_at_5, hm.recall_at_5),
        ("Recall @ 10", bm.recall_at_10, dm.recall_at_10, hm.recall_at_10),
        ("NDCG @ 5", bm.ndcg_at_5, dm.ndcg_at_5, hm.ndcg_at_5),
        ("NDCG @ 10", bm.ndcg_at_10, dm.ndcg_at_10, hm.ndcg_at_10),
        ("Avg Query Latency", bm.average_latency_ms, dm.average_latency_ms, hm.average_latency_ms),
    ]

    for name, b_val, d_val, h_val in metrics:
        if "Latency" in name:
            lift_str = f"{(h_val - b_val):+.1f} ms"
            table.add_row(name, f"{b_val:.2f} ms", f"{d_val:.2f} ms", f"{h_val:.2f} ms", lift_str)
        else:
            diff = (h_val - b_val)
            pct = (diff / b_val * 100) if b_val > 0 else 0.0
            lift_str = f"+{diff:.3f} ({pct:+.1f}%)" if diff >= 0 else f"{diff:.3f} ({pct:+.1f}%)"
            table.add_row(name, f"{b_val:.3f}", f"{d_val:.3f}", f"[bold green]{h_val:.3f}[/bold green]", f"[bold yellow]{lift_str}[/bold yellow]")

    console.print(table)
    console.print(f"[bold green][OK] Benchmark complete! Evaluation report saved to {output}[/bold green]")


@app.command(name="serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Launch the LegalTrace production REST API and interactive Web UI (Step 11 of Pipeline A)."""
    import uvicorn
    from ingestion_source.api import app as fastapi_app

    console.print(Panel(
        f"Starting [bold cyan]LegalTrace REST API & Interactive Research Explorer[/bold cyan] (Step 11)\n"
        f"Web UI: [bold green]http://{host}:{port}/[/bold green]\n"
        f"Swagger API Docs: [bold blue]http://{host}:{port}/docs[/bold blue]\n"
        f"ReDoc: [bold]http://{host}:{port}/redoc[/bold]",
        style="blue",
    ))

    uvicorn.run(fastapi_app, host=host, port=port)


@app.command(name="db-migrate")
def db_migrate_cmd(
    target: str = typer.Argument("all", help="Target database: all | postgres | qdrant | falkordb"),
    db_url: Optional[str] = typer.Option(None, "--db-url", help="PostgreSQL connection string override"),
    qdrant_url: Optional[str] = typer.Option(None, "--qdrant-url", help="Qdrant service URL override"),
    falkor_host: Optional[str] = typer.Option(None, "--falkor-host", help="FalkorDB host override"),
    falkor_port: Optional[int] = typer.Option(None, "--falkor-port", help="FalkorDB port override"),
) -> None:
    """Migrate Pipeline A artifacts into enterprise databases (PostgreSQL, Qdrant, FalkorDB)."""
    import asyncio
    from ingestion_source.config import get_settings
    from ingestion_source.db.migrate import (
        migrate_all,
        migrate_falkordb,
        migrate_postgres,
        migrate_qdrant,
    )

    cfg = get_settings()
    target_lower = target.lower().strip()

    console.print(Panel(
        f"[bold cyan]LegalTrace Database Infrastructure Migration (Phase 2)[/bold cyan]\n"
        f"Target: [bold yellow]{target_lower.upper()}[/bold yellow]\n"
        f"PostgreSQL: [dim]{db_url or cfg.DATABASE_URL}[/dim]\n"
        f"Qdrant: [dim]{qdrant_url or cfg.QDRANT_URL}[/dim]\n"
        f"FalkorDB: [dim]{falkor_host or cfg.FALKORDB_HOST}:{falkor_port or cfg.FALKORDB_PORT}[/dim]",
        style="cyan",
    ))

    if target_lower == "all":
        results = asyncio.run(migrate_all(cfg))
        console.print("[bold green]Migration Summary:[/bold green]")
        console.print_json(data=results)
    elif target_lower in ("postgres", "pg"):
        url = db_url or cfg.DATABASE_URL
        stats = asyncio.run(migrate_postgres(db_url=url))
        console.print(f"[bold green][OK] PostgreSQL populated:[/bold green] {stats}")
    elif target_lower in ("qdrant", "vector"):
        url = qdrant_url or cfg.QDRANT_URL
        stats = migrate_qdrant(qdrant_url=url, collection_name=cfg.QDRANT_COLLECTION)
        console.print(f"[bold green][OK] Qdrant populated:[/bold green] {stats}")
    elif target_lower in ("falkordb", "falkor", "graph"):
        host = falkor_host or cfg.FALKORDB_HOST
        port = falkor_port or cfg.FALKORDB_PORT
        stats = migrate_falkordb(host=host, port=port, graph_name=cfg.FALKORDB_GRAPH)
        console.print(f"[bold green][OK] FalkorDB populated:[/bold green] {stats}")
    else:
        console.print(f"[bold red]Unknown migration target '{target}'. Choose from: all, postgres, qdrant, falkordb[/bold red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Phase 1: Online Legal RAG & Agentic Reasoning CLI Commands
# ---------------------------------------------------------------------------

@app.command(name="rag-plan")
def rag_plan_cmd(
    query: str = typer.Argument(..., help="Legal query to decompose into sub-tasks"),
) -> None:
    """Decompose complex legal questions into retrieval sub-tasks and target rhetorical roles."""
    from ingestion_source.rag.planner import LegalQueryPlanner

    planner = LegalQueryPlanner()
    plan = planner.plan(query)

    console.print(Panel(
        f"Original Query: [bold cyan]\"{query}\"[/bold cyan]\n"
        f"Detected Domain: [bold yellow]{plan.detected_domain.upper()}[/bold yellow] | Strategy: [bold green]{plan.synthesis_strategy}[/bold green]",
        style="blue",
        title="[bold green]LegalTrace Query Decomposition Plan[/bold green]",
    ))

    table = Table(title=f"Retrieval Sub-Tasks ({len(plan.sub_queries)} tasks)")
    table.add_column("#", style="bold yellow", width=3)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Sub-Query Text", style="white")
    table.add_column("Target Roles", style="cyan", width=25)
    table.add_column("Top K", style="green", width=6)

    for i, sq in enumerate(plan.sub_queries, 1):
        roles_str = ", ".join(sq.target_roles) if sq.target_roles else "Any role"
        table.add_row(
            str(i),
            sq.query_type.value.upper(),
            sq.query_text,
            roles_str,
            str(sq.top_k),
        )

    console.print(table)


@app.command(name="rag-query")
def rag_query_cmd(
    query: str = typer.Argument(..., help="Legal research question"),
    max_chunks: int = typer.Option(8, "--max-chunks", "-k", help="Maximum evidence chunks to evaluate"),
    index_dir: Path = typer.Option(Path("search_index"), "--index-dir", help="Path to search index directory"),
) -> None:
    """Execute online legal RAG with query planning, multi-hop retrieval, Shepardization, and citation-grounded synthesis."""
    from ingestion_source.rag.synthesizer import LegalRAGSynthesizer
    from ingestion_source.search import HybridLegalSearchEngine

    if not index_dir.exists():
        console.print(f"[bold red]Index directory {index_dir} not found. Run 'ingest index' first.[/bold red]")
        raise typer.Exit(code=1)

    with console.status("[bold green]Loading search index..."):
        engine = HybridLegalSearchEngine(dim=384)
        engine.load_index(index_dir)

    synthesizer = LegalRAGSynthesizer(search_engine=engine)

    with console.status("[bold cyan]Planning query, retrieving evidence, and synthesizing opinion..."):
        res = synthesizer.query(user_query=query, engine=engine, max_evidence_chunks=max_chunks)

    # Display Query Plan Summary
    console.print(Panel(
        f"Query: [bold cyan]\"{res.query}\"[/bold cyan]\n"
        f"Domain: [bold yellow]{res.query_plan.detected_domain.upper()}[/bold yellow] | Model: [bold]{res.model_used}[/bold] | Latency: [bold green]{res.synthesis_latency_ms:.1f} ms[/bold green] | Confidence: [bold]{res.confidence * 100:.1f}%[/bold]",
        style="blue",
        title="[bold green]LegalTrace RAG Execution Telemetry[/bold green]",
    ))

    # Display Shepard Warnings if any
    if res.shepard_warnings:
        warn_panel = "\n".join(f"[bold red]•[/bold red] {w}" for w in res.shepard_warnings)
        console.print(Panel(warn_panel, style="red", title="[bold red]Shepard's Treatment Warnings[/bold red]"))

    # Synthesized Answer
    console.print(Panel(
        res.answer_text,
        style="green",
        title="[bold green]Synthesized Legal Opinion (Grounded in Evidence)[/bold green]",
    ))

    # Evidence Table
    if res.citations:
        table = Table(title=f"Table of Authorities & Retrieved Evidence ({len(res.citations)} passages)")
        table.add_column("Cite", style="bold yellow", width=5)
        table.add_column("Case Title & Court", style="bold white", width=32)
        table.add_column("Role", style="magenta", width=14)
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Evidence Snippet", style="white")

        for c in res.citations:
            table.add_row(
                f"[{c.citation_index}]",
                f"{c.case_title[:30]}...\n{c.court} ({c.year})",
                c.rhetorical_role.upper(),
                f"{c.relevance_score:.3f}",
                c.text[:180] + "...",
            )
        console.print(table)


@app.command(name="shepardize")
def shepardize_cmd(
    citation: str = typer.Argument(..., help="Precedent citation string (e.g. 'AIR 1978 SC 597')"),
) -> None:
    """Audit precedent treatment history across the corpus and determine whether it remains good law."""
    from ingestion_source.rag.shepardizer import LegalShepardizer

    shepardizer = LegalShepardizer()
    res = shepardizer.check(citation)

    status_color = "green" if res.is_good_law else "red" if res.status.value in ("overruled", "questioned") else "yellow"

    console.print(Panel(
        f"Precedent: [bold white]{res.precedent_citation}[/bold white] (Canonical Key: [dim]{res.canonical_id}[/dim])\n"
        f"Status: [{status_color}][bold]{res.status.value.upper()}[/bold][/{status_color}] | Good Law: [{status_color}]{res.is_good_law}[/{status_color}] | Total Citations: [bold cyan]{res.total_citations}[/bold cyan]",
        style="blue",
        title="[bold green]Precedent Shepardization & Good-Law Report[/bold green]",
    ))

    if res.warning_message:
        console.print(Panel(f"[bold yellow]{res.warning_message}[/bold yellow]", style="yellow", title="Warning Notice"))

    if res.treatment_summary:
        summary_table = Table(title="Treatment Breakdown")
        summary_table.add_column("Treatment", style="bold white")
        summary_table.add_column("Count", style="cyan")
        for k, v in res.treatment_summary.items():
            summary_table.add_row(k.upper().replace("_", " "), str(v))
        console.print(summary_table)

    if res.recent_treatments:
        treat_table = Table(title=f"Subsequent Citing Judgments ({len(res.recent_treatments)} instances)")
        treat_table.add_column("Citing Case Title", style="bold white", width=35)
        treat_table.add_column("Court / Year", style="dim", width=18)
        treat_table.add_column("Treatment", style="bold yellow", width=16)
        treat_table.add_column("Context Snippet", style="white")

        for t in res.recent_treatments[:10]:
            treat_table.add_row(
                t.citing_title[:33] + "...",
                f"{t.citing_court} ({t.citing_year})",
                t.treatment.upper().replace("_", " "),
                t.context[:160] + "...",
            )
        console.print(treat_table)


@app.command(name="memo")
def memo_cmd(
    query_or_doc_id: str = typer.Argument(..., help="Query for memo or doc_id for brief"),
    doc_type: str = typer.Option("memo", "--type", "-t", help="Document type: memo | brief | bail"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output text file path"),
    index_dir: Path = typer.Option(Path("search_index"), "--index-dir", help="Path to search index directory"),
) -> None:
    """Generate court-ready legal memorandum, case brief, or bail application draft."""
    from ingestion_source.rag.memo_generator import LegalMemoGenerator
    from ingestion_source.search import HybridLegalSearchEngine

    generator = LegalMemoGenerator()
    doc_type_lower = doc_type.lower().strip()

    if doc_type_lower in ("brief", "case_brief"):
        doc = generator.generate_case_brief(doc_id=query_or_doc_id)
    else:
        if not index_dir.exists():
            console.print(f"[bold red]Index directory {index_dir} not found. Run 'ingest index' first.[/bold red]")
            raise typer.Exit(code=1)
        engine = HybridLegalSearchEngine(dim=384)
        engine.load_index(index_dir)

        if doc_type_lower in ("bail", "bail_application"):
            doc = generator.generate_bail_application(
                applicant_name="Applicant",
                fir_no="000/2023",
                police_station="Special Cell",
                sections_invoked=["Section 439 CrPC"],
                custody_date="01.01.2023",
                facts_summary=query_or_doc_id,
                engine=engine,
            )
        else:
            doc = generator.generate_legal_memorandum(query=query_or_doc_id, engine=engine)

    console.print(Panel(
        f"Title: [bold white]{doc.title}[/bold white]\n"
        f"Type: [bold yellow]{doc.doc_type.upper()}[/bold yellow] | Generated At: [dim]{doc.generated_at}[/dim]",
        style="blue",
        title="[bold green]LegalTrace Document Generation[/bold green]",
    ))

    if output:
        output.write_text(doc.content, encoding="utf-8")
        console.print(f"[bold green][OK] Document saved to {output}[/bold green]")
    else:
        console.print(doc.content)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Multi-Court Corpus Expansion Commands
# ─────────────────────────────────────────────────────────────────────────────

@app.command(name="batch-ingest")
def batch_ingest(
    courts: Optional[list[str]] = typer.Option(None, "--court", "-c", help="Target court(s) to ingest"),
    years: Optional[list[int]] = typer.Option(None, "--year", "-y", help="Target decision year(s)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max documents per court-year batch"),
    workers: int = typer.Option(2, "--workers", "-w", help="Number of concurrent worker threads"),
    batch_size: int = typer.Option(50, "--batch-size", help="Batch size for database upserts"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion of completed batches"),
    mode: str = typer.Option("api", "--mode", "-m", help="Fetch mode: api, hybrid, scrape_only"),
) -> None:
    """Run multi-court, multi-year corpus expansion directly into PostgreSQL, Qdrant, and FalkorDB (Phase 3)."""
    import asyncio
    from ingestion_source.batch_ingest import BatchIngestionOrchestrator

    settings = get_settings()
    target_courts = courts or settings.TARGET_COURTS
    target_years = years or settings.TARGET_YEARS
    target_limit = limit if limit is not None else settings.SAMPLE_LIMIT

    console.print(Panel(
        f"[bold white]LegalTrace Multi-Court Corpus Expansion (Phase 3)[/bold white]\n"
        f"Courts ({len(target_courts)}): [cyan]{', '.join(target_courts)}[/cyan]\n"
        f"Years ({len(target_years)}): [cyan]{', '.join(str(y) for y in target_years)}[/cyan]\n"
        f"Batch Limit: [yellow]{target_limit}[/yellow] | Workers: [yellow]{workers}[/yellow] | Force: [yellow]{force}[/yellow]",
        style="blue",
    ))

    orchestrator = BatchIngestionOrchestrator(settings=settings)
    report = asyncio.run(
        orchestrator.run_batch(
            courts=target_courts,
            years=target_years,
            sample_limit=target_limit,
            batch_size=batch_size,
            workers=workers,
            force=force,
            mode=mode,
        )
    )

    table = Table(title="[bold green]Batch Corpus Expansion Summary[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")
    table.add_row("Total Batches", str(report.total_batches))
    table.add_row("Completed Batches", f"[green]{report.completed_batches}[/green]")
    table.add_row("Failed Batches", f"[red]{report.failed_batches}[/red]" if report.failed_batches else "0")
    table.add_row("Skipped Batches", f"[yellow]{report.skipped_batches}[/yellow]" if report.skipped_batches else "0")
    table.add_row("Total Docs Discovered", f"{report.total_docs_discovered:,}")
    table.add_row("Total Docs Fetched", f"{report.total_docs_fetched:,}")
    table.add_row("Total Docs Indexed (PostgreSQL)", f"[green]{report.total_docs_indexed:,}[/green]")
    table.add_row("Total Semantic Chunks (Qdrant)", f"[green]{report.total_chunks_indexed:,}[/green]")
    table.add_row("Total Citation Edges (FalkorDB)", f"[green]{report.total_edges_indexed:,}[/green]")
    table.add_row("Total Execution Time", f"{report.elapsed_seconds:.2f}s")
    console.print(table)


def _run_async_command(coro):
    """Run an async coroutine safely whether or not an event loop is currently active."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


@app.command(name="batch-status")
def batch_status() -> None:
    """Display progress and telemetry of all batch ingestion jobs stored in PostgreSQL."""
    from ingestion_source.batch_ingest import BatchIngestionOrchestrator

    orchestrator = BatchIngestionOrchestrator()
    jobs = _run_async_command(orchestrator.get_all_batch_status())

    if not jobs:
        console.print("[yellow]No batch ingestion jobs found in PostgreSQL database.[/yellow]")
        return

    table = Table(title="[bold green]PostgreSQL Batch Ingestion Telemetry (Phase 3)[/bold green]")
    table.add_column("Batch ID", style="bold white")
    table.add_column("Court", style="cyan")
    table.add_column("Year", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Discovered", style="white")
    table.add_column("Indexed", style="green")
    table.add_column("Chunks", style="green")
    table.add_column("Edges", style="green")
    table.add_column("Error", style="red")

    for j in jobs:
        status_color = "green" if j["status"] == "completed" else ("red" if j["status"] == "failed" else "yellow")
        table.add_row(
            j["batch_id"],
            j["court"],
            str(j["year"]),
            f"[{status_color}]{j['status'].upper()}[/{status_color}]",
            str(j["docs_discovered"]),
            str(j["docs_indexed"]),
            str(j["chunks_created"]),
            str(j["edges_created"]),
            (j["error_message"][:30] + "...") if j.get("error_message") else "-",
        )

    console.print(table)


@app.command(name="statute-lookup")
def statute_lookup(
    act: str = typer.Option(..., "--act", "-a", help="Act name or acronym (e.g. 'CrPC', 'IPC', 'Arbitration')"),
    section: str = typer.Option(..., "--section", "-s", help="Section number (e.g. '482', '438', '34')"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max ratio passages to return"),
    store_dir: Path = typer.Option(Path("raw-store"), "--store-dir", "-d", help="Path to raw-store directory"),
) -> None:
    """Reverse lookup of judicial interpretations and ratio decidendi for a statute provision (Phase 4)."""
    from ingestion_source.rag import StatuteRatioLookup

    console.print(Panel(
        f"Statute-to-Ratio Reverse Lookup\n"
        f"Act: [bold cyan]{act}[/bold cyan] | Section: [bold yellow]{section}[/bold yellow]",
        style="blue",
    ))

    lookup_engine = StatuteRatioLookup(raw_store_dir=store_dir)
    with console.status(f"[bold green]Searching judicial decisions interpreting {act} Section {section}..."):
        resp = _run_async_command(lookup_engine.lookup(act_name=act, section=section, top_k=limit))

    console.print(f"[bold]Canonical Act:[/bold] [cyan]{resp.canonical_act}[/cyan]")
    console.print(f"[bold]Total Judicial Rulings Found:[/bold] [green]{resp.total_results}[/green]\n")

    if resp.key_principles:
        console.print("[bold magenta]Synthesized Judicial Principles:[/bold magenta]")
        for i, p in enumerate(resp.key_principles, 1):
            console.print(f"  [bold yellow]{i}.[/bold yellow] {p}")
        console.print()

    if not resp.results:
        console.print("[yellow]No judicial rulings found interpreting this specific section in corpus.[/yellow]")
        return

    table = Table(title=f"Judicial Decisions Interpreting {resp.canonical_act} Section {section}")
    table.add_column("Doc ID", style="bold cyan")
    table.add_column("Case Title", style="bold white")
    table.add_column("Court / Year", style="yellow")
    table.add_column("Role", style="magenta")
    table.add_column("Ratio / Analysis Excerpt", style="white")

    for r in resp.results:
        excerpt = (r.ratio_text[:120] + "...") if len(r.ratio_text) > 120 else r.ratio_text
        table.add_row(
            r.doc_id,
            r.case_title[:35],
            f"{r.court} ({r.year})",
            r.rhetorical_role.upper(),
            excerpt,
        )

    console.print(table)


if __name__ == "__main__":
    app()


