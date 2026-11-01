from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from common import load_config, write_json
from data.market_data import (ALPHA_VANTAGE_ADJUSTED_ADAPTER, ALPHA_VANTAGE_PARSER_VERSION,
                              FAMA_FRENCH_BENCHMARK_DEFINITION, FAMA_FRENCH_DAILY_URL,
                              FAMA_FRENCH_FACTOR_LINEAGE_ID, FAMA_FRENCH_USAGE,
                              _canonical_json_bytes, _persist_factor_snapshot,
                              _persist_raw_snapshot, parse_fama_french_daily,
                              assemble_market_bundle, validate_market_bundle)
from data.pit import iso_utc, parse_utc
from evaluation.walk_forward import _event_row, _market, _settings, evaluate_research
from extraction.retrieve import retrieve_subgraph
from research.features import build_graph, prepare_events, snapshot_at, validate_events
from research.fixture import create_fixture
from research.paper import mark_positions, record_decisions, verify_ledger
from research.pipeline import (_current_submissions, _sec_submission_row,
                               DEFAULT_RESEARCH_CONFIG, load_inputs, verify_sec_snapshots)


def sec_snapshot_fixture(cache: Path, *, latency=60, fetched="2025-01-01T00:00:00Z") -> dict:
    """Create a full content-addressed SEC replay fixture, including receipts."""
    cik, ticker, accession = "0000000001", "TEST", "0000000001-25-000001"
    directory = f"https://www.sec.gov/Archives/edgar/data/1/{accession.replace('-', '')}/"
    current_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    submission = {"cik": cik, "filings": {"recent": {
        "accessionNumber": [accession], "form": ["8-K"],
        "filingDate": ["2024-01-02"], "primaryDocument": ["report.htm"],
        "acceptanceDateTime": ["2024-01-02T21:00:00Z"],
    }, "files": []}}
    header = ("<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20240102160000\n"
              f"ACCESSION NUMBER: {accession}\nCONFORMED SUBMISSION TYPE: 8-K\n"
              f"CENTRAL INDEX KEY: {cik}\n</SEC-HEADER>").encode()
    raw_by_key = {"header": header, "document": b"<html><p>Actual filing text.</p></html>",
                  "submissions": json.dumps(submission).encode(),
                  "current_submissions": json.dumps(submission).encode()}
    urls = {"header": f"{directory}{accession}.hdr.sgml", "document": f"{directory}report.htm",
            "submissions": current_url, "current_submissions": current_url}
    receipts = {}
    for key, raw in raw_by_key.items():
        digest = hashlib.sha256(raw).hexdigest()
        object_path = cache / "objects" / digest[:2] / digest
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(raw)
        receipt = {"sha256": digest, "retrieved_at": fetched, "url": urls[key],
                   "content_type": "text/html" if key == "document" else "application/json",
                   "raw_path": str(object_path.resolve())}
        encoded = json.dumps(receipt, sort_keys=True).encode("utf-8")
        sidecar = cache / "receipts" / hashlib.sha256(urls[key].encode()).hexdigest() / f"{hashlib.sha256(encoded).hexdigest()}.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(encoded)
        receipts[key] = receipt
    return {"event_id": f"sec_{ticker}_{accession}", "ticker": ticker, "cik": cik,
            "accession": accession, "form": "8-K", "filing_date": "2024-01-02",
            "primary_document": "report.htm", "source": "sec_edgar", "event_time": "2024-01-02T21:00:00Z",
            "available_at": "2024-01-02T21:00:00Z" if latency == 0 else
                            "2024-01-02T21:01:00Z" if latency == 60 else "2024-01-03T21:00:00Z",
            "retrieved_at": fetched, "availability_mode": "historical", "document_text": "Actual filing text.",
            "document_url": receipts["document"]["url"], "document_sha256": receipts["document"]["sha256"],
            "provenance": {**receipts, "parser_version": "sec-primary-text-v1", "latency_seconds": latency,
                           "availability_mode": "historical", "submissions_kind": "current",
                           "submissions_archive_name": None,
                           "header_identity": {"accession": accession, "form": "8-K", "ciks": [cik]},
                           "ticker_mapping": {"kind": "explicit_config", "ticker": ticker, "cik": cik}}}


class HistoricalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(DEFAULT_RESEARCH_CONFIG)
        envelope, self.market = create_fixture()
        self.raw = envelope["events"]

    def test_fixture_schema_and_no_future_novelty(self):
        validate_market_bundle(self.market)
        early = prepare_events(self.raw[:1], self.market, self.cfg)[0]
        whole = prepare_events(self.raw, self.market, self.cfg)
        same = next(e for e in whole if e["event_id"] == early["event_id"])
        self.assertEqual(early["features"], same["features"])
        self.assertEqual(early["feature_provenance"]["prior_documents"], 0)

    def test_sec_raw_json_replay_rejects_duplicate_keys(self):
        duplicate_submissions = (
            b'{"filings":{"recent":{"accessionNumber":["0000000001-25-000001"],'
            b'"form":["8-K"],"filingDate":["2024-01-02"],'
            b'"primaryDocument":["one.htm"]}},"filings":{}}'
        )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            _sec_submission_row(duplicate_submissions, "0000000001-25-000001")
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            _current_submissions(duplicate_submissions, "0000000001")

    def test_imported_scores_cannot_override_frozen_rule(self):
        self.raw[0]["score"] = 1.0
        self.raw[0]["document_text"] = "declined loss"
        row = prepare_events(self.raw[:1], self.market, self.cfg)[0]
        self.assertNotIn("score", row)
        self.assertEqual(row["features"]["direction"], -1)

    def test_late_sector_classification_stays_unknown(self):
        self.market["memberships"][0]["available_at"] = "2025-01-01T00:00:00Z"
        event = prepare_events(self.raw[:1], self.market, self.cfg)[0]
        self.assertEqual(event["sector"], "unknown")

    def test_live_clock_cannot_predate_collection(self):
        self.raw[0]["availability_mode"] = "live"
        with self.assertRaisesRegex(ValueError, "local retrieval"):
            validate_events(self.raw[:1])

    def test_historical_graph_has_only_observed_evidence(self):
        events = prepare_events(self.raw, self.market, self.cfg)
        first = events[0]
        graph = snapshot_at(events, self.market, self.cfg, first["available_at"])
        actual = {a["signal_id"] for _, a in graph.nodes(data=True) if a.get("kind") == "signal"}
        expected = {e["event_id"] for e in events if e["available_at"] <= first["available_at"]}
        self.assertEqual(actual, expected)
        self.assertFalse(graph.nodes[f"company:{first['ticker']}"]["price_history"])
        self.assertEqual({a["role"] for _, a in graph.nodes(data=True) if a.get("kind") == "player"}, {"issuer"})

    def test_utc_retrieval_filters_future_evidence_and_issuers(self):
        events = prepare_events(self.raw, self.market, self.cfg)
        graph = build_graph(events, self.market, self.cfg)
        first = events[0]
        before = (parse_utc(first["available_at"]) - timedelta(seconds=1)).isoformat()
        sub = retrieve_subgraph(graph, first["ticker"], before)
        self.assertEqual(sub["signals"], [])
        self.assertEqual(sub["players"], {})
        at = retrieve_subgraph(graph, first["ticker"], first["available_at"])
        self.assertEqual(len(at["signals"]), 1)

    def test_mixed_fixture_and_historical_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "events.json", {"metadata": {"data_kind": "historical"}, "events": self.raw})
            write_json(root / "market.json", self.market)
            self.cfg["research"].update(events_path=str(root / "events.json"), market_path=str(root / "market.json"))
            with self.assertRaisesRegex(ValueError, "cannot mix"):
                load_inputs(self.cfg)

    def test_historical_envelope_cannot_opt_out_of_the_sec_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market = copy.deepcopy(self.market)
            market["metadata"]["data_kind"] = "historical"
            write_json(root / "market.json", market)
            self.cfg["research"].update(events_path=str(root / "events.json"),
                                        market_path=str(root / "market.json"))

            # A mutable envelope label cannot choose an unverified adapter.
            write_json(root / "events.json", {"metadata": {"data_kind": "historical",
                                                "source": "made_up_adapter"}, "events": self.raw})
            with self.assertRaisesRegex(ValueError, "must declare the sec_edgar"):
                load_inputs(self.cfg)

            # Nor can the envelope declare SEC while individual events use a
            # different source and would previously have been skipped.
            write_json(root / "events.json", {"metadata": {"data_kind": "historical",
                                                "source": "sec_edgar"}, "events": self.raw})
            with self.assertRaisesRegex(ValueError, "must all use the sec_edgar"):
                load_inputs(self.cfg)

    def test_direct_historical_evaluator_replays_sec_receipts_not_mutable_marker(self):
        """A public evaluator call cannot promote a self-labelled SEC event."""
        market = copy.deepcopy(self.market)
        market["metadata"]["data_kind"] = "historical"
        events = prepare_events(self.raw[:1], self.market, self.cfg)
        events[0]["source"] = "sec_edgar"
        events[0]["provenance"] = {"source_integrity": "sec_edgar_replay_verified"}
        events[0].pop("cik", None)
        with self.assertRaisesRegex(ValueError, "historical_source_config"):
            evaluate_research(events, market, self.cfg["research"]["evaluation"])
        with self.assertRaisesRegex(ValueError, "SEC event is missing"):
            evaluate_research(events, market, self.cfg["research"]["evaluation"],
                              historical_source_config=self.cfg)

    def test_historical_load_inputs_replays_sec_price_and_factor_artifacts_together(self):
        """The official admission path binds all supported raw sources in one run."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sec_cache, market_cache = root / "raw-sec", root / "raw-market"
            stamp = parse_utc("2025-01-01T00:00:00Z")
            session = {"session": "2024-01-02", "open_at": "2024-01-02T14:30:00Z",
                       "close_at": "2024-01-02T21:00:00Z"}
            price_payload = {"Meta Data": {"2. Symbol": "TEST"}, "Time Series (Daily)": {
                "2024-01-02": {"5. adjusted close": "100", "6. volume": "1000"}}}
            price_receipt, price_sidecar = _persist_raw_snapshot(
                market_cache, _canonical_json_bytes(price_payload), provider="Alpha Vantage",
                endpoint="https://www.alphavantage.co/query",
                identity={"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": "TEST"},
                sessions=[session], adapter=ALPHA_VANTAGE_ADJUSTED_ADAPTER,
                parser_version=ALPHA_VANTAGE_PARSER_VERSION, retrieved_at=stamp)
            price = {"metadata": {
                "data_kind": "historical", "source": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
                "price_lineage_id": "alpha_vantage:TIME_SERIES_DAILY_ADJUSTED:TEST",
                "adjustment": "split_and_dividend", "retrieved_at": iso_utc(stamp),
                "input_sha256": price_receipt["sha256"], "raw_receipt": price_receipt,
                "raw_receipt_path": price_sidecar,
                "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(price_receipt)).hexdigest(),
                "usage": "Adjusted-price ratios only; levels are restated as of retrieval",
            }, "bars": [{**session, "ticker": "TEST", "available_at": iso_utc(stamp),
                           "adjusted_close": 100.0, "volume": 1000.0}]}
            factor_raw = b",Mkt-RF,SMB,HML,RF\n20240102,1,0,0,0.01\n"
            factor_receipt, factor_sidecar = _persist_factor_snapshot(
                market_cache, factor_raw, retrieved_at=stamp)
            factor_rows = parse_fama_french_daily(factor_raw)
            factors = {"metadata": {
                "data_kind": "historical", "source": FAMA_FRENCH_DAILY_URL,
                "retrieved_at": iso_utc(stamp), "input_sha256": factor_receipt["sha256"],
                "raw_receipt": factor_receipt, "raw_receipt_path": factor_sidecar,
                "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(factor_receipt)).hexdigest(),
                "factor_lineage_id": FAMA_FRENCH_FACTOR_LINEAGE_ID, "return_unit": "decimal",
                "factor_vintage": "retrieved_snapshot",
                "benchmark_definition": FAMA_FRENCH_BENCHMARK_DEFINITION,
                "usage": FAMA_FRENCH_USAGE,
            }, "factors": factor_rows,
                "benchmark": [{"session": row["session"], "return": row["mkt_rf"] + row["rf"]}
                              for row in factor_rows]}
            price_path, factor_path, schedule_path = root / "price.json", root / "factor.json", root / "schedule.json"
            write_json(price_path, price)
            write_json(factor_path, factors)
            write_json(schedule_path, {"version": 1, "calendar_id": "test-xnys-v1",
                                       "source": "test exchange schedule",
                                       "retrieved_at": "2024-01-01T00:00:00Z",
                                       "sessions": [session]})
            market = assemble_market_bundle([price_path], factor_path, "one test issuer", schedule_path)
            market_path, events_path = root / "market.json", root / "events.json"
            write_json(market_path, market)
            event = sec_snapshot_fixture(sec_cache)
            write_json(events_path, {"metadata": {"data_kind": "historical", "source": "sec_edgar"},
                                     "events": [event]})
            cfg = copy.deepcopy(self.cfg)
            cfg["research"].update(events_path=str(events_path), market_path=str(market_path))
            cfg["research"]["edgar"]["cache_dir"] = str(sec_cache)
            events, admitted_market, provenance = load_inputs(cfg)
            self.assertEqual(events[0]["provenance"]["source_integrity"], "sec_edgar_replay_verified")
            self.assertEqual(provenance["market"]["market_integrity"]["status"],
                             "price_factor_replayed_only")
            self.assertEqual(admitted_market["metadata"]["data_kind"], "historical")

    def test_sec_text_and_clock_are_reverified_against_raw_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.cfg["research"]["edgar"]["cache_dir"] = tmp
            event = sec_snapshot_fixture(Path(tmp))
            verify_sec_snapshots([event], self.cfg)
            forged = copy.deepcopy(event)
            forged["source"] = "unverified_adapter"
            with self.assertRaisesRegex(ValueError, "SEC event source"):
                verify_sec_snapshots([forged], self.cfg)
            event["document_text"] = "invented direction"
            with self.assertRaisesRegex(ValueError, "text differs"):
                verify_sec_snapshots([event], self.cfg)
            event["document_text"] = "Actual filing text."
            event["available_at"] = "2024-01-02T20:01:00Z"
            with self.assertRaisesRegex(ValueError, "clocks differ"):
                verify_sec_snapshots([event], self.cfg)
            event["available_at"] = "2024-01-02T21:01:00Z"
            event["ticker"] = "MISMATCH"
            with self.assertRaisesRegex(ValueError, "event_id"):
                verify_sec_snapshots([event], self.cfg)

    def test_sec_snapshot_allows_the_collector_maximum_day_latency(self):
        # The collector permits a 24-hour conservative availability delay, and
        # replay must preserve that legal configuration rather than rejecting it.
        with tempfile.TemporaryDirectory() as tmp:
            self.cfg["research"]["edgar"]["cache_dir"] = tmp
            event = sec_snapshot_fixture(Path(tmp), latency=86400, fetched="2024-01-02T21:00:00Z")
            verify_sec_snapshots([event], self.cfg)

    def test_sec_replay_rejects_forged_receipts_and_broken_identity_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            self.cfg["research"]["edgar"]["cache_dir"] = tmp
            event = sec_snapshot_fixture(cache)
            forged = copy.deepcopy(event)
            forged["provenance"]["document"]["retrieved_at"] = "2024-01-02T21:00:00Z"
            with self.assertRaisesRegex(ValueError, "receipt is absent"):
                verify_sec_snapshots([forged], self.cfg)

            forged = copy.deepcopy(event)
            forged["primary_document"] = "other.htm"
            with self.assertRaisesRegex(ValueError, "primary document"):
                verify_sec_snapshots([forged], self.cfg)

            forged = copy.deepcopy(event)
            forged["provenance"]["submissions_kind"] = "archive"
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_sec_snapshots([forged], self.cfg)

            forged = copy.deepcopy(event)
            forged["provenance"]["header_identity"]["form"] = "10-K"
            with self.assertRaisesRegex(ValueError, "saved header identity"):
                verify_sec_snapshots([forged], self.cfg)

            # Even a self-consistent cache receipt is rejected when the SEC's
            # current submissions payload identifies a different issuer CIK.
            forged = copy.deepcopy(event)
            bad_current = {"cik": "0000000002", "filings": {"recent": {
                "accessionNumber": [event["accession"]], "form": ["8-K"],
                "filingDate": ["2024-01-02"], "primaryDocument": ["report.htm"],
                "acceptanceDateTime": ["2024-01-02T21:00:00Z"],
            }, "files": []}}
            raw = json.dumps(bad_current).encode()
            digest = hashlib.sha256(raw).hexdigest()
            object_path = cache / "objects" / digest[:2] / digest
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(raw)
            receipt = {"sha256": digest, "retrieved_at": "2025-01-01T00:00:00Z",
                       "url": forged["provenance"]["current_submissions"]["url"],
                       "content_type": "application/json", "raw_path": str(object_path.resolve())}
            encoded = json.dumps(receipt, sort_keys=True).encode("utf-8")
            sidecar = cache / "receipts" / hashlib.sha256(receipt["url"].encode()).hexdigest() / f"{hashlib.sha256(encoded).hexdigest()}.json"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_bytes(encoded)
            forged["provenance"]["current_submissions"] = receipt
            with self.assertRaisesRegex(ValueError, "current submissions CIK"):
                verify_sec_snapshots([forged], self.cfg)


class ProspectivePaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "paper.json"
        self.cfg = load_config(DEFAULT_RESEARCH_CONFIG)
        envelope, self.market = create_fixture()
        self.market["metadata"]["data_kind"] = "historical"  # test-only provider seam
        raw = copy.deepcopy(envelope["events"][:1])
        raw[0]["availability_mode"] = "live"
        raw[0]["retrieved_at"] = raw[0]["available_at"]
        self.events = prepare_events(raw, self.market, self.cfg)
        self.now = parse_utc(raw[0]["available_at"]) + timedelta(seconds=30)
        self.attach_price_replay(self.market, parse_utc(self.market["metadata"]["retrieved_at"]))
        # A live record may know the announced future session schedule but not
        # tomorrow's bars/factors.  Build a point-in-time market snapshot for
        # recording, and retain the full later bundle solely for marking.
        self.record_market = self.market_at(self.now)

    def market_at(self, at):
        market = copy.deepcopy(self.market)
        market["metadata"]["retrieved_at"] = iso_utc(at)
        clocks = {row["session"]: parse_utc(row["close_at"])
                  for row in market["session_schedule"]["sessions"]}
        market["bars"] = [row for row in market["bars"] if parse_utc(row["close_at"]) <= at]
        for key in ("benchmark", "factors", "sector_returns"):
            market[key] = [row for row in market[key] if clocks[row["session"]] <= at]
        self.attach_price_replay(market, at)
        return market

    def attach_price_replay(self, market, retrieved_at):
        """Create deterministic offline provider captures for the paper audit seam."""
        stamp = iso_utc(retrieved_at)
        cache = Path(self.tmp.name) / "raw-market"
        components, lineage = [], []
        for ticker in sorted({row["ticker"] for row in market["bars"]}):
            selected = [row for row in market["bars"] if row["ticker"] == ticker]
            for row in selected:
                row["available_at"] = stamp
            sessions = [{key: row[key] for key in ("session", "open_at", "close_at")} for row in selected]
            payload = {"Meta Data": {"2. Symbol": ticker}, "Time Series (Daily)": {
                row["session"]: {"5. adjusted close": repr(row["adjusted_close"]),
                                 "6. volume": repr(row["volume"])} for row in selected}}
            receipt, receipt_path = _persist_raw_snapshot(
                cache, _canonical_json_bytes(payload), provider="Alpha Vantage",
                endpoint="https://www.alphavantage.co/query",
                identity={"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker},
                sessions=sessions, adapter=ALPHA_VANTAGE_ADJUSTED_ADAPTER,
                parser_version=ALPHA_VANTAGE_PARSER_VERSION, retrieved_at=retrieved_at,
            )
            lineage_id = f"alpha_vantage:TIME_SERIES_DAILY_ADJUSTED:{ticker}"
            component_metadata = {
                "data_kind": "historical", "source": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
                "price_lineage_id": lineage_id, "adjustment": "split_and_dividend",
                "retrieved_at": stamp, "input_sha256": receipt["sha256"],
                "raw_receipt": receipt, "raw_receipt_path": receipt_path,
                "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest(),
                "usage": "Adjusted-price ratios only; levels are restated as of retrieval",
            }
            components.append({"role": "prices", "metadata": component_metadata})
            lineage.append({"id": lineage_id, "source": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
                            "identity": receipt["identity"]})

        # The fixture stores decimal returns, whereas the Fama--French parser
        # accepts percent values.  Reconstruct the market fields from this raw
        # CSV rather than retaining fixture rows, so the historical test seam
        # exercises the identical receipt/replay contract as a provider input.
        factor_csv = ",Mkt-RF,SMB,HML,RF\n" + "".join(
            ",".join((row["session"].replace("-", ""),
                      *(format(float(row[field]) * 100.0, ".17g")
                        for field in ("mkt_rf", "smb", "hml", "rf")))) + "\n"
            for row in market["factors"]
        )
        factor_raw = factor_csv.encode("utf-8")
        factor_receipt, factor_receipt_path = _persist_factor_snapshot(
            cache, factor_raw, retrieved_at=retrieved_at,
        )
        replayed_factors = parse_fama_french_daily(factor_raw)
        market["factors"] = replayed_factors
        market["benchmark"] = [{"session": row["session"], "return": row["mkt_rf"] + row["rf"]}
                               for row in replayed_factors]
        factor_metadata = {
            "data_kind": "historical", "source": FAMA_FRENCH_DAILY_URL,
            "retrieved_at": stamp, "input_sha256": factor_receipt["sha256"],
            "raw_receipt": factor_receipt, "raw_receipt_path": factor_receipt_path,
            "raw_receipt_sha256": hashlib.sha256(_canonical_json_bytes(factor_receipt)).hexdigest(),
            "factor_lineage_id": FAMA_FRENCH_FACTOR_LINEAGE_ID,
            "return_unit": "decimal", "factor_vintage": "retrieved_snapshot",
            "benchmark_definition": FAMA_FRENCH_BENCHMARK_DEFINITION,
            "usage": FAMA_FRENCH_USAGE,
        }
        components.append({"role": "factors", "metadata": factor_metadata})
        market["metadata"].update({"data_kind": "historical", "source": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
                                   "price_lineage": sorted(lineage, key=lambda row: row["id"]),
                                   "factor_lineage": {"id": FAMA_FRENCH_FACTOR_LINEAGE_ID,
                                                      "source": FAMA_FRENCH_DAILY_URL,
                                                      "identity": factor_receipt["identity"]},
                                   "component_metadata": components})

    def record(self):
        return record_decisions(self.events, self.path, settings=self.cfg["research"]["evaluation"],
                                data_kind="historical", market=self.record_market, now=self.now)

    def test_record_is_idempotent_and_clock_is_prospective(self):
        first = self.record()
        again = self.record()
        self.assertEqual(first, again)
        self.assertEqual(len(first["entries"]), 1)
        self.assertEqual(parse_utc(first["entries"][0]["decision_at"]), self.now)

    def test_fixture_replay_and_future_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            record_decisions(self.events, self.path, settings={}, data_kind="fixture",
                             market=self.record_market, now=self.now)
        self.events[0]["availability_mode"] = "historical"
        with self.assertRaisesRegex(ValueError, "historical reconstruction"):
            self.record()
        self.events[0]["availability_mode"] = "live"
        self.events[0]["features_available_at"] = "2025-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "future"):
            self.record()

    def test_live_fetch_of_stale_filing_cannot_enter_prospective_ledger(self):
        # Live availability may be recent solely because the first fetch was
        # recent. The SEC acceptance, not that fetch, defines filing freshness.
        self.events[0]["event_time"] = iso_utc(self.now - timedelta(days=365))
        with self.assertRaisesRegex(ValueError, "filing acceptance is too old"):
            self.record()
        self.assertFalse(self.path.exists())

    def test_paper_freshness_policy_is_frozen_in_ledger(self):
        ledger = self.record()
        entry = ledger["entries"][0]
        self.assertEqual(entry["event_time"], self.events[0]["event_time"])
        self.assertEqual(entry["max_filing_age_seconds"], 259200)
        entry["event_time"] = iso_utc(self.now - timedelta(days=365))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            verify_ledger(ledger)

    def test_ledger_corruption_is_detected(self):
        ledger = self.record()
        ledger["entries"][0]["score"] += .1
        with self.assertRaisesRegex(ValueError, "hash chain"):
            verify_ledger(ledger)

    def test_marks_have_no_predecision_fill_and_deduct_costs(self):
        ledger = self.record()
        marked = mark_positions(ledger, self.market, "2025-01-01T00:00:00Z", horizon=5)
        row = marked["rows"][0]
        self.assertEqual(row["status"], "closed")
        self.assertGreater(parse_utc(row["entry_at"]), self.now)
        self.assertAlmostEqual(row["net_return"], row["gross_return"] - row["cost"])
        self.assertFalse(marked["capital_deployment_allowed"])

    def test_missing_entry_session_and_unplanned_horizon_rejected(self):
        ledger = self.record()
        first_future = next(b for b in self.market["bars"] if b["ticker"] == self.events[0]["ticker"]
                            and parse_utc(b["close_at"]) > self.now)
        self.market["bars"].remove(first_future)
        self.attach_price_replay(self.market, parse_utc("2025-01-01T00:00:00Z"))
        with self.assertRaisesRegex(ValueError, "missing planned market bar"):
            mark_positions(ledger, self.market, "2025-01-01T00:00:00Z", horizon=5)
        with self.assertRaisesRegex(ValueError, "not planned"):
            mark_positions(ledger, self.market, "2025-01-01T00:00:00Z", horizon=7)

    def test_changed_or_missing_frozen_execution_data_fails_closed(self):
        ledger = self.record()
        truncated = copy.deepcopy(self.market)
        planned_entry = ledger["entries"][0]["execution_plan"]["entry"]["session"]
        ticker = self.events[0]["ticker"]
        truncated["bars"] = [row for row in truncated["bars"]
                             if not (row["ticker"] == ticker and row["session"] == planned_entry)]
        self.attach_price_replay(truncated, parse_utc("2025-01-01T00:00:00Z"))
        with self.assertRaisesRegex(ValueError, "missing planned market bar"):
            mark_positions(ledger, truncated, "2025-01-01T00:00:00Z", horizon=5)

        foreign = copy.deepcopy(self.market)
        last = [bar for bar in foreign["bars"] if bar["ticker"] == ticker][-1]
        last["session"] = "2099-01-01"
        with self.assertRaisesRegex(ValueError, "absent from the timestamped"):
            mark_positions(ledger, foreign, "2025-01-01T00:00:00Z", horizon=5)

        changed = copy.deepcopy(self.market)
        for clock in changed["session_schedule"]["sessions"]:
            if clock["session"] == planned_entry:
                clock["close_at"] = clock["close_at"].replace("21:00:00", "21:01:00")
        for bar in changed["bars"]:
            if bar["session"] == planned_entry:
                bar["close_at"] = bar["close_at"].replace("21:00:00", "21:01:00")
                if "available_at" in bar:
                    bar["available_at"] = bar["available_at"].replace("21:00:00", "21:01:00")
        self.attach_price_replay(changed, parse_utc("2025-01-01T00:00:00Z"))
        with self.assertRaisesRegex(ValueError, "changed or omitted a frozen"):
            mark_positions(ledger, changed, "2025-01-01T00:00:00Z", horizon=5)

    def test_same_day_missing_scheduled_close_and_preclose_pending_are_distinct(self):
        # The original filing was known by the following afternoon.  Freeze a
        # same-day 21:00Z close, then prove a 22:00Z missing bar is not deferred.
        first_later = next(row for row in self.market["session_schedule"]["sessions"]
                           if parse_utc(row["close_at"]) > self.now)
        decision = parse_utc(first_later["close_at"]) - timedelta(hours=1)
        ledger = record_decisions(self.events, Path(self.tmp.name) / "same-day.json",
                                  settings=self.cfg["research"]["evaluation"], data_kind="historical",
                                  market=self.market_at(decision), now=decision)
        self.assertEqual(ledger["entries"][0]["execution_plan"]["entry"], first_later)
        pending = mark_positions(ledger, self.market_at(decision + timedelta(minutes=30)),
                                 iso_utc(decision + timedelta(minutes=30)), horizon=5)
        self.assertEqual(pending["rows"][0]["status"], "pending")
        missing = self.market_at(decision + timedelta(hours=2))
        missing["bars"] = [row for row in missing["bars"] if not (
            row["ticker"] == self.events[0]["ticker"] and row["session"] == first_later["session"])]
        self.attach_price_replay(missing, decision + timedelta(hours=2))
        with self.assertRaisesRegex(ValueError, "missing planned market bar"):
            mark_positions(ledger, missing, iso_utc(decision + timedelta(hours=2)), horizon=5)

    def test_paper_and_evaluation_share_the_frozen_execution_schedule(self):
        ledger = self.record()
        row = _event_row(self.events[0], 5, _market(self.market),
                         _settings(self.cfg["research"]["evaluation"]))
        plan = ledger["entries"][0]["execution_plan"]
        self.assertEqual(row["_execution_plan"]["entry"], plan["entry"])
        self.assertEqual(row["_execution_plan"]["exits"]["5"], plan["exits"]["5"])

    def test_later_schedule_extension_and_missing_future_factors_do_not_move_an_open_mark(self):
        ledger = self.record()
        plan = ledger["entries"][0]["execution_plan"]
        cutoff = parse_utc(plan["window"][1]["close_at"])
        later = self.market_at(cutoff)
        later["factors"] = [row for row in later["factors"] if row["session"] != plan["window"][1]["session"]]
        later["session_schedule"]["sessions"].append({"session": "2099-01-01",
            "open_at": "2099-01-01T14:30:00Z", "close_at": "2099-01-01T21:00:00Z"})
        # Model this as a genuine later snapshot rather than mutating rows
        # behind the persisted factor receipt.
        self.attach_price_replay(later, cutoff)
        marked = mark_positions(ledger, later, iso_utc(cutoff), horizon=5)
        self.assertEqual(marked["rows"][0]["status"], "open")

    def test_mark_requires_as_of_snapshot_and_replay_valid_price_lineage(self):
        ledger = self.record()
        # Current 2025 bundle cannot be passed off as a 2023 mark merely because
        # its code filters bars by the requested historical cutoff.
        with self.assertRaisesRegex(ValueError, "retrieved after the mark cutoff"):
            mark_positions(ledger, self.market, iso_utc(self.now), horizon=5)

        later = self.market_at(parse_utc("2025-01-01T00:00:00Z"))
        # A mutable assembled lineage cannot claim a different provider identity:
        # replay rejects it before marking can compare it to the frozen ledger.
        later["metadata"]["price_lineage"][0]["source"] = "different adjusted-price vendor"
        with self.assertRaisesRegex(ValueError, "price_lineage differs"):
            mark_positions(ledger, later, "2025-01-01T00:00:00Z", horizon=5)


if __name__ == "__main__":
    unittest.main()
