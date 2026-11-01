"""Network-free boundary tests for actual SEC document ingestion."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data.edgar_documents import (
    ARCHIVES, SUBMISSIONS, acceptance_from_header, collect_filings,
    extract_document_text, parse_header_identity,
)
from data.pit import iso_utc, parse_utc


ACCESSION = "0000320193-25-000001"
DIRECTORY = f"{ARCHIVES}320193/{ACCESSION.replace('-', '')}/"
NOW = datetime(2025, 2, 1, 12, tzinfo=timezone.utc)


class Response:
    def __init__(self, body, status=200, content_type="text/html; charset=utf-8"):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def iter_content(self, chunk_size):
        for position in range(0, len(self.body), chunk_size):
            yield self.body[position:position + chunk_size]

    def close(self):
        self.closed = True


class HTTP:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.routes[url]
        return response.pop(0) if isinstance(response, list) else response


def table(accessions=None, forms=None, date="2025-01-15"):
    accessions = accessions or [ACCESSION]
    count = len(accessions)
    return {"accessionNumber": accessions, "form": forms or ["8-K"] * count,
            "filingDate": [date] * count, "primaryDocument": ["report.htm"] * count,
            "acceptanceDateTime": ["2025-01-15T21:30:00.000Z"] * count}


def header(accession=ACCESSION, form="8-K", cik="0000320193", accepted="20250115163000"):
    return ("<SEC-HEADER>\n"
            f"<ACCEPTANCE-DATETIME>{accepted}\n"
            f"ACCESSION NUMBER: {accession}\n"
            f"CONFORMED SUBMISSION TYPE: {form}\n"
            f"CENTRAL INDEX KEY: {cik}\n"
            "</SEC-HEADER>").encode()


def routes(recent=None, archives=None):
    return {
        f"{SUBMISSIONS}CIK0000320193.json": Response({"cik": "0000320193", "filings": {"recent": table() if recent is None else recent, "files": archives or []}}),
        f"{DIRECTORY}{ACCESSION}.hdr.sgml": Response(header()),
        f"{DIRECTORY}report.htm": Response(b"<html><head><style>bad style</style></head><body><script>bad script</script><p>Revenue &amp; profit grew.</p><ix:hidden>hidden facts</ix:hidden></body></html>"),
    }


class TimestampTests(unittest.TestCase):
    def test_offsets_roundtrip_and_naive_clocks_rejected(self):
        self.assertEqual(iso_utc(parse_utc("2025-01-15T16:30:00-05:00")), "2025-01-15T21:30:00Z")
        for timestamp in ("2025-01-15", "2025-01-15T16:30:00", "D0015", "2025-01-15T16:30Z", "2025-01-15T16:30:00+02:90"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                parse_utc(timestamp)
        with self.assertRaises(ValueError):
            iso_utc(datetime(2025, 1, 15))

    def test_header_uses_new_york_dst_and_rejects_ambiguous_time(self):
        self.assertEqual(iso_utc(acceptance_from_header(b"<ACCEPTANCE-DATETIME>20250715163000")), "2025-07-15T20:30:00Z")
        self.assertEqual(iso_utc(acceptance_from_header(b"<ACCEPTANCE-DATETIME>20250115163000")), "2025-01-15T21:30:00Z")
        for header in (b"<ACCEPTANCE-DATETIME>20251102013000", b"<ACCEPTANCE-DATETIME>20250309023000", b"<FILING-DATE>20250115"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                acceptance_from_header(header)

    def test_header_identity_requires_accession_form_and_filer_cik(self):
        parsed = parse_header_identity(header())
        self.assertEqual(parsed, {"accession": ACCESSION, "form": "8-K", "ciks": ["0000320193"]})
        with self.assertRaisesRegex(ValueError, "ACCESSION"):
            parse_header_identity(b"<SEC-HEADER>\nCONFORMED SUBMISSION TYPE: 8-K\nCENTRAL INDEX KEY: 1\n")


class EdgarDocumentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = {"paths": {"root": self.temporary.name}, "research": {"edgar": {
            "tickers": ["AAPL"], "ciks": {"AAPL": 320193}, "start_date": "2025-01-01",
            "end_date": "2025-01-31", "forms": ["8-K", "8-K/A"],
            "user_agent": "SignalGraph research contact@example.com", "latency_seconds": 60,
            "availability_mode": "historical", "cache_dir": "raw", "max_retries": 0,
        }}}

    def collect(self, http):
        return collect_filings(self.cfg, http_get=http, clock=lambda: NOW, sleep=lambda _: None)

    def test_actual_document_and_acceptance_are_preserved_in_immutable_cache(self):
        http = HTTP(routes())
        result = self.collect(http)[0]
        self.assertEqual(result["event_time"], "2025-01-15T21:30:00Z")
        self.assertEqual(result["available_at"], "2025-01-15T21:31:00Z")
        self.assertEqual(result["retrieved_at"], "2025-02-01T12:00:00Z")
        self.assertEqual(result["document_text"], "Revenue & profit grew.")
        self.assertEqual(result["provenance"]["api_minus_header_seconds"], 0)
        raw = Path(result["provenance"]["document"]["raw_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["document_sha256"])
        self.assertEqual(len(http.calls), 3)
        for _, kwargs in http.calls:
            self.assertEqual(kwargs["timeout"], 30)
            self.assertTrue(kwargs["stream"])
            self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(self.collect(http)[0], result)
        self.assertEqual(len(http.calls), 4)  # only submissions refreshes

    def test_live_clock_requires_local_retrieval_and_keeps_mismatched_api_clock(self):
        self.cfg["research"]["edgar"]["availability_mode"] = "live"
        recent = table()
        recent["acceptanceDateTime"] = ["2025-01-15T16:30:00Z"]
        result = self.collect(HTTP(routes(recent)))[0]
        self.assertEqual(result["available_at"], "2025-02-01T12:00:00Z")
        self.assertEqual(result["provenance"]["api_minus_header_seconds"], -18000)
        self.assertEqual(result["event_time"], "2025-01-15T21:30:00Z")

    def test_historical_pages_missing_api_clock_use_actual_header_and_keep_amendments(self):
        amended = "0000320193-25-000002"
        amendment_dir = f"{ARCHIVES}320193/{amended.replace('-', '')}/"
        archived = table([ACCESSION, amended], ["8-K", "8-K/A"])
        del archived["acceptanceDateTime"]
        history_name = "CIK0000320193-submissions-001.json"
        source = routes({}, [{"name": history_name, "filingFrom": "2025-01-01", "filingTo": "2025-01-31"}])
        source[f"{SUBMISSIONS}{history_name}"] = Response(archived)
        source[f"{amendment_dir}{amended}.hdr.sgml"] = Response(header(amended, "8-K/A", accepted="20250116163000"))
        source[f"{amendment_dir}report.htm"] = Response(b"<p>Amended profit grew.</p>")
        results = self.collect(HTTP(source))
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0]["event_id"], results[1]["event_id"])
        self.assertTrue(results[1]["provenance"]["is_amendment"])
        self.assertIsNone(results[0]["provenance"]["api_acceptance_raw"])

    def test_missing_acceptance_header_fails_without_filing_date_fallback(self):
        source = routes()
        source[f"{DIRECTORY}{ACCESSION}.hdr.sgml"] = Response(b"<FILING-DATE>20250115")
        with self.assertRaisesRegex(ValueError, "missing ACCEPTANCE"):
            self.collect(HTTP(source))

    def test_form_date_and_archive_bounds_filter_before_document_fetch(self):
        history = {"name": "CIK0000320193-submissions-001.json", "filingFrom": "2024-01-01", "filingTo": "2024-12-31"}
        http = HTTP(routes(table(forms=["10-K"]), [history]))
        self.assertEqual(self.collect(http), [])
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(self.collect(HTTP(routes(table(date="2024-12-31")))), [])

    def test_explicit_historical_ticker_mapping_is_required(self):
        self.cfg["research"]["edgar"]["ciks"] = {}
        # A current ticker index can be true today but wrong for the historical
        # study date (rename/reuse/class change), so the collector must not
        # silently substitute it for a reviewed security-master decision.
        with self.assertRaisesRegex(ValueError, "Explicit historical ticker/CIK"):
            self.collect(HTTP(routes()))

    def test_malformed_or_external_document_paths_and_metadata_fail_closed(self):
        for filename in ("https://example.com/doc.html", "../doc.htm", "subdir/doc.htm"):
            recent = table()
            recent["primaryDocument"] = [filename]
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "Invalid primary"):
                self.collect(HTTP(routes(recent)))
        recent = table()
        recent["filingDate"] = []
        with self.assertRaisesRegex(ValueError, "unequal lengths"):
            self.collect(HTTP(routes(recent)))

    def test_cache_corruption_is_detected(self):
        http = HTTP(routes())
        result = self.collect(http)[0]
        # Tampering is deliberately performed here to exercise hash verification.
        Path(result["provenance"]["document"]["raw_path"]).write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.collect(http)

    def test_retries_and_response_byte_limit_are_bounded(self):
        self.cfg["research"]["edgar"]["max_retries"] = 1
        source = routes()
        url = f"{SUBMISSIONS}CIK0000320193.json"
        source[url] = [Response(b"throttled", 429), source[url]]
        http = HTTP(source)
        self.assertEqual(len(self.collect(http)), 1)
        self.assertEqual(len([call for call in http.calls if call[0] == url]), 2)
        self.cfg["research"]["edgar"]["max_response_bytes"] = 5
        with self.assertRaisesRegex(ValueError, "exceeds max_response_bytes"):
            self.collect(HTTP(routes()))

    def test_request_rate_floor_and_identifier_required(self):
        self.cfg["research"]["edgar"]["rate_limit_seconds"] = 0.01
        with self.assertRaisesRegex(ValueError, "rate_limit_seconds"):
            self.collect(HTTP(routes()))
        self.cfg["research"]["edgar"]["rate_limit_seconds"] = 0.1
        self.cfg["research"]["edgar"]["user_agent"] = "anonymous"
        with self.assertRaisesRegex(ValueError, "identified name"):
            self.collect(HTTP(routes()))

    def test_requests_are_paced_including_transient_retry(self):
        self.cfg["research"]["edgar"]["max_retries"] = 1
        source = routes()
        url = f"{SUBMISSIONS}CIK0000320193.json"
        source[url] = [Response(b"temporary failure", 503), source[url]]
        http = HTTP(source)
        elapsed = [0.0]
        starts = []

        def sleep(seconds):
            elapsed[0] += seconds

        def fetch(url, **kwargs):
            starts.append(elapsed[0])
            return http(url, **kwargs)

        collect_filings(self.cfg, http_get=fetch, clock=lambda: NOW,
                        sleep=sleep, monotonic=lambda: elapsed[0])
        self.assertTrue(all(right - left >= 0.1 for left, right in zip(starts, starts[1:])))

    def test_visible_text_parser_rejects_unsupported_and_empty_documents(self):
        self.assertEqual(extract_document_text(b"Plain text filing", "report.txt"), "Plain text filing")
        self.assertEqual(extract_document_text(b'<p hidden>secret</p><p style="display:none">hidden</p><p>Visible</p>', "report.htm"), "Visible")
        for raw, filename in ((b"%PDF document", "report.pdf"), (b"<script>hidden</script>", "report.htm")):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                extract_document_text(raw, filename)


if __name__ == "__main__":
    unittest.main()
