"""SEC filing documents with auditable point-in-time clocks.

The submissions API is discovery metadata. The filing's raw SGML header is the
acceptance clock: ACCEPTANCE-DATETIME is explicitly interpreted in
America/New_York, including DST. The API timestamp is retained for comparison;
it never silently replaces a missing header. Acceptance plus configured latency
is a *historical availability assumption*, not a measured dissemination time.
Live availability additionally requires local retrieval of metadata and bytes.

Only SEC HTTPS endpoints are constructed. Static headers/documents reuse cached
bytes; submissions metadata is refreshed. Cache objects and retrieval receipts
are content-addressed and never overwritten. This collector is serial and rate
limited; callers running multiple processes must also enforce SEC's aggregate
10 requests/second per user limit.

References: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
https://www.sec.gov/about/developer-resources
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable
from zoneinfo import ZoneInfo

from data.pit import iso_utc, parse_utc

SEC_INDEX = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/"
PARSER_VERSION = "sec-primary-text-v1"


def _date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"Expected YYYY-MM-DD date, got {value!r}")
    return date.fromisoformat(value)


def _finite(value, name: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _cik(value) -> str:
    if not re.fullmatch(r"\d{1,10}", str(value)) or int(value) < 1:
        raise ValueError(f"Invalid SEC CIK: {value!r}")
    return f"{int(value):010d}"


def _safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or ".." in value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def acceptance_from_header(raw: bytes) -> datetime:
    """Normalize SEC's header wall clock in America/New_York, rejecting DST gaps/folds."""
    match = re.search(rb"<ACCEPTANCE-DATETIME>\s*(\d{14})(?!\d)", raw, re.IGNORECASE)
    if not match:
        raise ValueError("SEC header missing ACCEPTANCE-DATETIME; filing cannot be timed")
    naive = datetime.strptime(match.group(1).decode("ascii"), "%Y%m%d%H%M%S")
    eastern = ZoneInfo("America/New_York")
    local = naive.replace(tzinfo=eastern)
    if local.utcoffset() != naive.replace(tzinfo=eastern, fold=1).utcoffset():
        raise ValueError("SEC acceptance wall clock is ambiguous or nonexistent at DST transition")
    utc = local.astimezone(timezone.utc)
    if utc.astimezone(eastern).replace(tzinfo=None) != naive:
        raise ValueError("SEC acceptance wall clock is nonexistent")
    return utc


def _header_values(raw: bytes, label: str, tag: str) -> list[str]:
    """Return distinct values for a named SEC-header field.

    EDGAR headers have appeared both as line-oriented ``LABEL: value`` records
    and tag-like records.  Restrict parsing to the SEC header when it is
    delimited, and require whole-line labels so document prose cannot become
    issuer metadata by accident.
    """
    text = raw.decode("latin-1", errors="replace")
    block = re.search(r"<SEC-HEADER>(.*?)(?:</SEC-HEADER>|$)", text, re.I | re.S)
    if block:
        text = block.group(1)
    values = re.findall(rf"(?im)^\s*{re.escape(label)}\s*:\s*([^\r\n<]+?)\s*$", text)
    values.extend(re.findall(rf"(?is)<{re.escape(tag)}>\s*([^<\r\n]+?)\s*(?:</{re.escape(tag)}>|(?=<))", text))
    return sorted({" ".join(value.split()) for value in values if value.strip()})


def parse_header_identity(raw: bytes) -> dict:
    """Parse filing identity embedded in a raw SEC SGML header.

    The acceptance clock alone is not enough to bind a document to an event.
    Collection and replay share this parser and require one accession/form plus
    at least one filer CIK.  Headers may list other CIKs (for example reporting
    owners), so callers check that their expected CIK is among ``ciks``.
    """
    accessions = _header_values(raw, "ACCESSION NUMBER", "ACCESSION-NUMBER")
    if len(accessions) != 1 or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accessions[0]):
        raise ValueError("SEC header must contain exactly one valid ACCESSION NUMBER")
    forms = _header_values(raw, "CONFORMED SUBMISSION TYPE", "CONFORMED-SUBMISSION-TYPE")
    if len(forms) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./-]*", forms[0]):
        raise ValueError("SEC header must contain exactly one valid CONFORMED SUBMISSION TYPE")
    raw_ciks = _header_values(raw, "CENTRAL INDEX KEY", "CENTRAL-INDEX-KEY")
    try:
        ciks = sorted({_cik(value) for value in raw_ciks})
    except ValueError as exc:
        raise ValueError("SEC header contains an invalid CENTRAL INDEX KEY") from exc
    if not ciks:
        raise ValueError("SEC header is missing CENTRAL INDEX KEY")
    return {"accession": accessions[0], "form": forms[0], "ciks": ciks}


class _VisibleText(HTMLParser):
    OMIT = {"script", "style", "noscript", "head", "ix:hidden"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        hidden = tag in self.OMIT or "hidden" in values or bool(re.search(
            r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", values.get("style") or "", re.I
        ))
        suppressed = hidden or bool(self.stack and self.stack[-1][1])
        if tag not in self.VOID:
            self.stack.append((tag, suppressed))
        if not suppressed:
            self.parts.append(" ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data)


def extract_document_text(raw: bytes, filename: str, content_type: str = "") -> str:
    """Extract a primary HTML/text document; never synthesize a filing summary."""
    if filename.lower().endswith(".pdf") or raw.startswith(b"%PDF"):
        raise ValueError("PDF primary documents require a separate verified parser")
    encoding_match = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    if not encoding_match:
        encoding_match = re.search(r"charset\s*=\s*[\"']?([\w-]+)", raw[:8192].decode("ascii", errors="ignore"), re.I)
    encoding = encoding_match.group(1) if encoding_match else "utf-8"
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise ValueError(f"Primary document encoding could not be decoded ({encoding})") from exc
    if filename.lower().endswith((".htm", ".html", ".xhtml")) or "html" in content_type.lower():
        parser = _VisibleText()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    elif not filename.lower().endswith(".txt"):
        raise ValueError(f"Unsupported primary document format: {filename}")
    result = " ".join(text.split())
    if not result:
        raise ValueError("SEC primary document contains no visible text")
    return result


class _Cache:
    def __init__(self, root: Path):
        self.root = root

    def _immutable_write(self, path: Path, data: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Immutable SEC cache object was corrupted: {path}")

    def save(self, url: str, raw: bytes, retrieved_at: str, content_type: str) -> dict:
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / "objects" / digest[:2] / digest
        self._immutable_write(path, raw)
        receipt = {"url": url, "sha256": digest, "raw_path": str(path.resolve()),
                   "retrieved_at": retrieved_at, "content_type": content_type}
        encoded = json.dumps(receipt, sort_keys=True).encode("utf-8")
        receipt_hash = hashlib.sha256(encoded).hexdigest()
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        self._immutable_write(self.root / "receipts" / url_hash / f"{receipt_hash}.json", encoded)
        return receipt

    def get(self, url: str):
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        receipts = self.root / "receipts" / url_hash
        if not receipts.exists():
            return None
        records = []
        for path in receipts.glob("*.json"):
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != path.stem:
                raise ValueError("SEC cache retrieval receipt checksum mismatch")
            record = json.loads(data)
            if record.get("url") != url:
                raise ValueError("SEC cache URL mismatch")
            parse_utc(record["retrieved_at"])
            records.append(record)
        if not records:
            return None
        record = min(records, key=lambda row: parse_utc(row["retrieved_at"]))
        digest = record["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("SEC cache digest is invalid")
        raw = (self.root / "objects" / digest[:2] / digest).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("SEC cache object checksum mismatch")
        return raw, record


class _SecClient:
    def __init__(self, cfg: dict, root: Path, http_get=None, clock=None, sleep=None, monotonic=None):
        user_agent = cfg.get("user_agent") or os.environ.get("SEC_USER_AGENT", "")
        if not isinstance(user_agent, str) or not re.search(r"\S+@\S+\.\S+", user_agent) or "\n" in user_agent or "\r" in user_agent:
            raise ValueError("Set research.edgar.user_agent or SEC_USER_AGENT to an identified name and contact email")
        self.user_agent = user_agent
        self.timeout = _finite(cfg.get("timeout_seconds", 30), "timeout_seconds", 0.1, 120)
        self.delay = _finite(cfg.get("rate_limit_seconds", 0.11), "rate_limit_seconds", 0.1, 60)
        retries = cfg.get("max_retries", 2)
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5")
        self.retries = retries
        self.max_bytes = int(_finite(cfg.get("max_response_bytes", 25_000_000), "max_response_bytes", 1, 100_000_000))
        cache_path = Path(cfg.get("cache_dir", "results/research/sec_cache"))
        self.cache = _Cache(cache_path if cache_path.is_absolute() else root / cache_path)
        if http_get is None:
            import requests
            http_get = requests.get
        self.http_get = http_get
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.monotonic = monotonic or time.monotonic
        self.last_start = None

    def get(self, url: str, *, immutable=False):
        if immutable:
            cached = self.cache.get(url)
            if cached is not None:
                return cached
        for attempt in range(self.retries + 1):
            if self.last_start is not None:
                self.sleep(max(0.0, self.delay - (self.monotonic() - self.last_start)))
            self.last_start = self.monotonic()
            response = None
            try:
                response = self.http_get(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                                         timeout=self.timeout, stream=True, allow_redirects=False)
                status = response.status_code
                if status in (429, 500, 502, 503, 504):
                    raise ConnectionError(f"SEC HTTP {status}")
                if status != 200:
                    raise ValueError(f"SEC HTTP {status} for {url}; no fallback data was produced")
                chunks, total = [], 0
                for chunk in response.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise ValueError(f"SEC response exceeds max_response_bytes for {url}")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                if not raw:
                    raise ValueError(f"Empty SEC response for {url}")
                receipt = self.cache.save(url, raw, iso_utc(self.clock()), response.headers.get("Content-Type", ""))
                return raw, receipt
            except ValueError:
                raise
            except Exception as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"SEC fetch failed after {attempt + 1} attempts for {url}: {exc}") from exc
                self.sleep(min(8.0, 0.5 * 2 ** attempt))
            finally:
                if response is not None:
                    response.close()
        raise AssertionError("Unreachable SEC request state")

    def json(self, url: str):
        raw, receipt = self.get(url)
        return json.loads(raw.decode("utf-8")), receipt


def _rows(table: dict):
    if not isinstance(table, dict):
        raise ValueError("SEC submissions table must be an object of arrays")
    keys = ("accessionNumber", "form", "filingDate", "primaryDocument")
    if not table:
        return []
    if any(not isinstance(table.get(key), list) for key in keys):
        raise ValueError("SEC submissions missing required column arrays")
    count = len(table["accessionNumber"])
    if any(len(table[key]) != count for key in keys):
        raise ValueError("SEC submissions columns have unequal lengths")
    accepted = table.get("acceptanceDateTime", [None] * count)
    if not isinstance(accepted, list) or len(accepted) != count:
        raise ValueError("SEC acceptanceDateTime column has invalid length")
    return [{**{key: table[key][i] for key in keys}, "acceptanceDateTime": accepted[i]} for i in range(count)]


def collect_filings(cfg: dict, *, http_get: Callable | None = None, clock=None, sleep=None, monotonic=None) -> list[dict]:
    """Collect actual SEC primary documents, never substitute synthetic records.

    Configure research.edgar.{tickers,ciks,start_date,end_date,forms,user_agent,
    latency_seconds,availability_mode,cache_dir,timeout_seconds,rate_limit_seconds}.
    The inclusive range is SEC filingDate, not fiscal period or UTC calendar date.
    Optional injected HTTP/clocks keep deterministic tests entirely offline.
    """
    options = cfg.get("research", {}).get("edgar", {})
    start, end = _date(options.get("start_date")), _date(options.get("end_date"))
    if start > end:
        raise ValueError("research.edgar.start_date must be <= end_date")
    mode = options.get("availability_mode", "historical")
    if mode not in ("historical", "live"):
        raise ValueError("availability_mode must be historical or live")
    latency = _finite(options.get("latency_seconds", 60), "latency_seconds", 0, 86400)
    tickers = options.get("tickers", [])
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("research.edgar.tickers must be a nonempty list")
    tickers = [str(ticker).upper() for ticker in tickers]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", ticker) for ticker in tickers) or len(set(tickers)) != len(tickers):
        raise ValueError("research.edgar.tickers must be unique valid ticker symbols")
    forms = options.get("forms", ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"])
    if not isinstance(forms, list) or not forms or any(not isinstance(form, str) or not form for form in forms):
        raise ValueError("research.edgar.forms must be a nonempty list")
    forms = set(forms)
    root = Path(cfg.get("paths", {}).get("root", Path(__file__).resolve().parents[1]))
    client = _SecClient(options, root, http_get, clock, sleep, monotonic)
    configured_ciks = options.get("ciks", {})
    if not isinstance(configured_ciks, dict):
        raise ValueError("research.edgar.ciks must be an explicit ticker-to-CIK mapping")
    ciks = {str(key).upper(): _cik(value) for key, value in configured_ciks.items()}
    missing = [ticker for ticker in tickers if ticker not in ciks]
    if missing:
        # The SEC's current ticker endpoint cannot prove a historical identity:
        # symbols may be renamed, reused, or otherwise fail to describe the
        # issuer at the study date.  Make the researcher supply that durable
        # security-master decision rather than silently binding a live lookup.
        raise ValueError(f"Explicit historical ticker/CIK mappings are required for {missing}")
    output = []
    for ticker in tickers:
        cik = ciks[ticker]
        payload, current_receipt = client.json(f"{SUBMISSIONS}CIK{cik}.json")
        if not isinstance(payload, dict) or not isinstance(payload.get("filings"), dict):
            raise ValueError(f"Malformed SEC submissions metadata for {ticker}")
        if _cik(payload.get("cik")) != cik:
            raise ValueError(f"SEC current submissions CIK does not match configured issuer for {ticker}")
        archives = payload["filings"].get("files", [])
        if not isinstance(archives, list) or any(not isinstance(archive, dict) for archive in archives):
            raise ValueError(f"Malformed SEC archived-submissions index for {ticker}")
        tables = [(payload["filings"].get("recent", {}), current_receipt, "current", None)]
        for archive in archives:
            # Missing bounds require loading the file, not silently losing history.
            if archive.get("filingFrom") and _date(archive["filingFrom"]) > end:
                continue
            if archive.get("filingTo") and _date(archive["filingTo"]) < start:
                continue
            name = _safe_name(archive.get("name"), "historical submissions filename")
            if not re.fullmatch(rf"CIK{cik}-submissions-\d+\.json", name):
                raise ValueError("Historical submissions filename does not match issuer CIK")
            historical, receipt = client.json(f"{SUBMISSIONS}{name}")
            tables.append((historical, receipt, "archive", name))
        seen = {}
        for table, metadata_receipt, submissions_kind, archive_name in tables:
            for row in _rows(table):
                if row["form"] not in forms or not start <= _date(row["filingDate"]) <= end:
                    continue
                accession = row["accessionNumber"]
                if not isinstance(accession, str) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
                    raise ValueError(f"Invalid SEC accession: {accession!r}")
                if accession in seen:
                    if seen[accession] != row:
                        raise ValueError(f"Conflicting SEC metadata for accession {accession}")
                    continue
                seen[accession] = row
                document = _safe_name(row["primaryDocument"], "primary document filename")
                directory = f"{ARCHIVES}{int(cik)}/{accession.replace('-', '')}/"
                header_raw, header_receipt = client.get(f"{directory}{accession}.hdr.sgml", immutable=True)
                accepted = acceptance_from_header(header_raw)
                header_identity = parse_header_identity(header_raw)
                if (header_identity["accession"] != accession or header_identity["form"] != row["form"]
                        or cik not in header_identity["ciks"]):
                    raise ValueError("SEC SGML header identity differs from selected submissions metadata")
                raw, document_receipt = client.get(f"{directory}{document}", immutable=True)
                document_text = extract_document_text(raw, document, document_receipt["content_type"])
                fetched = max(parse_utc(receipt["retrieved_at"]) for receipt in
                              (current_receipt, metadata_receipt, header_receipt, document_receipt))
                public_available = accepted + timedelta(seconds=latency)
                available = max(public_available, fetched) if mode == "live" else public_available
                api_value = row.get("acceptanceDateTime")
                try:
                    api_delta = (parse_utc(api_value) - accepted).total_seconds() if api_value else None
                except ValueError:
                    api_delta = None
                output.append({
                    "event_id": f"sec_{ticker}_{accession}", "ticker": ticker, "source": "sec_edgar",
                    "form": row["form"], "event_time": iso_utc(accepted), "available_at": iso_utc(available),
                    "retrieved_at": iso_utc(fetched), "document_text": document_text,
                    "document_url": document_receipt["url"], "document_sha256": document_receipt["sha256"],
                    "accession": accession, "cik": cik, "filing_date": row["filingDate"],
                    "primary_document": document,
                    "availability_mode": mode,
                    "provenance": {
                        "provider": "SEC EDGAR", "data_kind": "historical" if mode == "historical" else "live",
                        "parser_version": PARSER_VERSION, "availability_mode": mode,
                        "availability_assumption": "acceptance_plus_latency" if mode == "historical" else "max_acceptance_plus_latency_and_local_retrieval",
                        "latency_seconds": latency, "public_available_at_assumed": iso_utc(public_available),
                        "acceptance_clock": "SGML ACCEPTANCE-DATETIME interpreted in America/New_York",
                        "api_acceptance_raw": api_value, "api_minus_header_seconds": api_delta,
                        "header": header_receipt, "header_identity": header_identity,
                        "document": document_receipt,
                        "submissions": metadata_receipt, "current_submissions": current_receipt,
                        "submissions_kind": submissions_kind, "submissions_archive_name": archive_name,
                        "ticker_mapping": {"kind": "explicit_config", "ticker": ticker, "cik": cik},
                        "range_basis": "SEC filingDate inclusive", "is_amendment": row["form"].endswith("/A"),
                    },
                })
    return sorted(output, key=lambda record: (parse_utc(record["available_at"]), record["event_id"]))
