"""CLI for immutable historical research runs; no synthetic fallback on data errors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from common import PKG_ROOT, load_config, write_json
from data.market_data import _json_bytes, load_market_bundle, verify_historical_market_provenance
from data.pit import iso_utc, parse_utc
from graph.graph_io import save_graph
from research.features import FEATURE_VERSION, build_graph, prepare_events


DEFAULT_RESEARCH_CONFIG = PKG_ROOT / "config" / "research.yaml"


def resolve(cfg: dict, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["paths"]["root"]) / p


def _read_sec_receipt(cache: Path, receipt: dict, label: str, objects: dict[str, bytes]) -> bytes:
    """Read a declared SEC cache object and its immutable receipt sidecar.

    The normalized event envelope is untrusted input during replay.  Bind its
    receipt fields to the exact sidecar written by ``_Cache.save`` before using
    the declared hash; never dereference the envelope's ``raw_path``.
    """
    if not isinstance(receipt, dict):
        raise ValueError(f"SEC {label} receipt must be an object")
    expected_fields = {"url", "sha256", "raw_path", "retrieved_at", "content_type"}
    if set(receipt) != expected_fields:
        raise ValueError(f"SEC {label} receipt fields do not match the immutable cache schema")
    digest = receipt.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"SEC {label} receipt requires a SHA256")
    if not isinstance(receipt.get("url"), str) or not receipt["url"].startswith("https://"):
        raise ValueError(f"SEC {label} receipt requires an HTTPS URL")
    parse_utc(receipt.get("retrieved_at", ""))
    if not isinstance(receipt.get("content_type"), str):
        raise ValueError(f"SEC {label} receipt content_type must be a string")
    object_path = cache / "objects" / digest[:2] / digest
    if not isinstance(receipt.get("raw_path"), str) or Path(receipt["raw_path"]).resolve() != object_path.resolve():
        raise ValueError(f"SEC {label} receipt raw_path does not match the content-addressed object")
    encoded = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_hash = hashlib.sha256(encoded).hexdigest()
    url_hash = hashlib.sha256(receipt["url"].encode("utf-8")).hexdigest()
    sidecar = cache / "receipts" / url_hash / f"{receipt_hash}.json"
    if not sidecar.is_file() or sidecar.read_bytes() != encoded:
        raise ValueError(f"SEC {label} receipt is absent from the immutable cache")
    if digest not in objects:
        # Resolve by digest under the configured cache, never receipt.raw_path.
        raw = object_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("SEC snapshot cache hash mismatch")
        objects[digest] = raw
    return objects[digest]


def _sec_submission_row(raw: bytes, accession: str) -> dict:
    """Find exactly one accession in either current or archived SEC submissions JSON."""
    from data.edgar_documents import _rows

    try:
        payload = _json_bytes(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("SEC submissions cache object is not valid strict JSON") from exc
    table = payload.get("filings", {}).get("recent") if isinstance(payload, dict) and "filings" in payload else payload
    rows = _rows(table)
    matches = [row for row in rows if row["accessionNumber"] == accession]
    if len(matches) != 1:
        raise ValueError("SEC submissions metadata must contain exactly one selected accession")
    return matches[0]


def _current_submissions(raw: bytes, cik: str) -> dict:
    """Parse the issuer's current SEC submission index and bind its CIK."""
    try:
        payload = _json_bytes(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("SEC current submissions cache object is not valid strict JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("filings"), dict):
        raise ValueError("SEC current submissions cache object is malformed")
    try:
        actual = f"{int(str(payload.get('cik'))):010d}"
    except (TypeError, ValueError) as exc:
        raise ValueError("SEC current submissions cache object has an invalid CIK") from exc
    if actual != cik:
        raise ValueError("SEC current submissions CIK differs from the event CIK")
    if not isinstance(payload["filings"].get("files", []), list):
        raise ValueError("SEC current submissions archive index is malformed")
    return payload


def _archive_names(current: dict) -> set[str]:
    names = set()
    for record in current["filings"].get("files", []):
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("SEC current submissions archive index has an invalid filename")
        names.add(record["name"])
    return names


def _verify_ticker_mapping(event: dict, mapping: dict, cache: Path, objects: dict[str, bytes]) -> None:
    ticker, cik = event["ticker"], event["cik"]
    if not isinstance(mapping, dict):
        raise ValueError("SEC event is missing ticker-mapping provenance")
    if mapping.get("kind") == "explicit_config":
        if mapping.get("ticker") != ticker or mapping.get("cik") != cik:
            raise ValueError("SEC event ticker/CIK differs from explicit source mapping")
        # This is a user-supplied security-master/configuration label, not SEC
        # proof of a historical ticker-to-CIK mapping.
        return
    raw = _read_sec_receipt(cache, mapping, "ticker mapping", objects)
    if mapping["url"] != "https://www.sec.gov/files/company_tickers.json":
        raise ValueError("SEC ticker mapping must use the official ticker index URL")
    try:
        records = _json_bytes(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("SEC ticker mapping cache object is not valid strict JSON") from exc
    matches = False
    if isinstance(records, dict):
        for row in records.values():
            if not isinstance(row, dict):
                continue
            raw_cik = str(row.get("cik_str", ""))
            if raw_cik.isdigit() and str(row.get("ticker", "")).upper() == ticker and int(raw_cik) == int(cik):
                matches = True
                break
    if not matches:
        raise ValueError("SEC event ticker/CIK is absent from the saved ticker mapping")


def verify_sec_snapshots(events: list[dict], cfg: dict) -> None:
    """Reconnect imported SEC text and clocks to hashed source bytes before scoring."""
    from datetime import timedelta
    from data.edgar_documents import (ARCHIVES, PARSER_VERSION, SUBMISSIONS,
                                      acceptance_from_header, extract_document_text,
                                      parse_header_identity)
    cache = resolve(cfg, cfg["research"].get("edgar", {}).get("cache_dir", "results/raw/sec"))
    objects = {}
    for event in events:
        if event.get("source") != "sec_edgar":
            raise ValueError("historical SEC event source must be sec_edgar")
        required = ("event_id", "ticker", "cik", "accession", "form", "filing_date",
                    "primary_document", "document_url", "document_sha256")
        if any(not isinstance(event.get(key), str) or not event[key] for key in required):
            raise ValueError("SEC event is missing issuer/accession/document identity")
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", event["ticker"]):
            raise ValueError("SEC event ticker is invalid")
        if not re.fullmatch(r"\d{10}", event["cik"]) or int(event["cik"]) < 1:
            raise ValueError("SEC event CIK must be a zero-padded positive 10-digit value")
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", event["accession"]):
            raise ValueError("SEC event accession is invalid")
        if event["event_id"] != f"sec_{event['ticker']}_{event['accession']}":
            raise ValueError("SEC event_id must bind the ticker and accession")
        provenance = event.get("provenance", {})
        if provenance.get("parser_version") != PARSER_VERSION:
            raise ValueError("SEC snapshot parser version is missing or incompatible")
        receipts = {}
        for key in ("header", "document", "submissions", "current_submissions"):
            receipt = provenance.get(key, {})
            _read_sec_receipt(cache, receipt, key, objects)
            receipts[key] = receipt
        accession, cik = event["accession"], event["cik"]
        directory = f"{ARCHIVES}{int(cik)}/{accession.replace('-', '')}/"
        expected_header_url = f"{directory}{accession}.hdr.sgml"
        if receipts["header"]["url"] != expected_header_url:
            raise ValueError("SEC header URL does not match the event CIK/accession")
        current_url = f"{SUBMISSIONS}CIK{cik}.json"
        if receipts["current_submissions"]["url"] != current_url:
            raise ValueError("SEC current submissions URL does not match the event CIK")
        metadata_url = receipts["submissions"]["url"]
        historical_pattern = rf"{re.escape(SUBMISSIONS)}CIK{cik}-submissions-\d+\.json"
        if metadata_url != current_url and not re.fullmatch(historical_pattern, metadata_url):
            raise ValueError("SEC selected submissions URL does not match the event CIK")
        current = _current_submissions(objects[receipts["current_submissions"]["sha256"]], cik)
        kind = provenance.get("submissions_kind")
        archive_name = provenance.get("submissions_archive_name")
        if metadata_url == current_url:
            if kind != "current" or archive_name is not None:
                raise ValueError("SEC selected current submissions provenance is inconsistent")
            selected_raw = objects[receipts["current_submissions"]["sha256"]]
        else:
            expected_name = metadata_url.rsplit("/", 1)[-1]
            if kind != "archive" or archive_name != expected_name:
                raise ValueError("SEC selected archived submissions provenance is inconsistent")
            if expected_name not in _archive_names(current):
                raise ValueError("SEC selected archived submissions file is absent from the current index")
            selected_raw = objects[receipts["submissions"]["sha256"]]
        selected = _sec_submission_row(selected_raw, accession)
        if (selected["form"] != event["form"] or selected["filingDate"] != event["filing_date"]
                or selected["primaryDocument"] != event["primary_document"]):
            raise ValueError("SEC event form/date/primary document differs from saved submissions metadata")
        document_name = event["primary_document"]
        if not isinstance(document_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", document_name):
            raise ValueError("SEC selected primary-document name is invalid")
        document = receipts["document"]
        if (event["document_sha256"] != document["sha256"] or event["document_url"] != document["url"]
                or document["url"] != f"{directory}{document_name}"):
            raise ValueError("SEC event document identity differs from raw provenance")
        expected = extract_document_text(objects[document["sha256"]], document_name,
                                         document.get("content_type", ""))
        if expected != event["document_text"]:
            raise ValueError("SEC event text differs from the hashed primary document")
        accepted = acceptance_from_header(objects[receipts["header"]["sha256"]])
        header_identity = parse_header_identity(objects[receipts["header"]["sha256"]])
        if (header_identity["accession"] != accession or header_identity["form"] != event["form"]
                or cik not in header_identity["ciks"]):
            raise ValueError("SEC SGML header identity differs from the event")
        saved_identity = provenance.get("header_identity")
        if saved_identity is not None and saved_identity != header_identity:
            raise ValueError("SEC saved header identity differs from the raw SGML header")
        fetched = max(parse_utc(r["retrieved_at"]) for r in receipts.values())
        latency = float(provenance["latency_seconds"])
        if not 0 <= latency <= 86400:
            raise ValueError("invalid SEC snapshot latency")
        available = accepted + timedelta(seconds=latency)
        mode = event.get("availability_mode")
        if mode not in ("historical", "live") or mode != provenance.get("availability_mode"):
            raise ValueError("inconsistent SEC snapshot availability mode")
        if mode == "live":
            available = max(available, fetched)
        if (parse_utc(event["event_time"]) != accepted or parse_utc(event["retrieved_at"]) != fetched
                or parse_utc(event["available_at"]) != available):
            raise ValueError("SEC event clocks differ from raw provenance")
        _verify_ticker_mapping(event, provenance.get("ticker_mapping"), cache, objects)


def load_inputs(cfg: dict):
    r = cfg.get("research")
    if not r:
        raise ValueError("historical initial graph build requires config/research.yaml and validated research inputs")
    event_path = resolve(cfg, r["events_path"])
    if not event_path.exists():
        raise ValueError(f"missing EDGAR events: {event_path}; run make research-fetch RCFG=... first")
    raw = event_path.read_bytes()
    # Use the same duplicate-key/non-finite JSON policy as market inputs.  An
    # envelope hash is not meaningful if parsers can disagree about its fields.
    envelope = _json_bytes(raw)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("events"), list):
        raise ValueError("events JSON requires {metadata: {data_kind: ...}, events: [...]}")
    kind = envelope.get("metadata", {}).get("data_kind")
    if kind not in ("historical", "fixture"):
        raise ValueError("events metadata.data_kind must be historical or fixture")
    market = load_market_bundle(resolve(cfg, r["market_path"]))
    if market["metadata"]["data_kind"] != kind:
        raise ValueError("cannot mix fixture events with historical market data (or vice versa)")
    if not envelope["events"]:
        raise ValueError("no filing events; empty studies cannot be evaluated")
    source = envelope.get("metadata", {}).get("source")
    if kind == "historical":
        # Historical replay currently has exactly one supported evidence adapter.
        # A normalized JSON envelope is writable input, so neither its own label
        # nor an individual event label may opt out of SEC receipt replay.
        if source != "sec_edgar":
            raise ValueError("historical research events must declare the sec_edgar adapter")
        if any(event.get("source") != "sec_edgar" for event in envelope["events"]):
            raise ValueError("historical research events must all use the sec_edgar adapter")
        verify_sec_snapshots(envelope["events"], cfg)
        market_integrity = verify_historical_market_provenance(market)
        # This is run-local admission evidence, not a replacement for the raw
        # receipt chain above.  It prevents the evaluator's promotion report
        # from treating a merely self-labelled event as verified.
        for event in envelope["events"]:
            provenance = event.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError("historical SEC event lacks provenance")
            event["provenance"] = {**provenance, "source_integrity": "sec_edgar_replay_verified"}
        market["provenance"] = {**market.get("provenance", {}), "market_integrity": market_integrity}
    else:
        # Fixtures are intentionally unable to masquerade as a historical SEC
        # study.  New real-source adapters must be added explicitly above with
        # their own immutable-replay verifier.
        if not isinstance(source, str) or not source.lower().startswith(("fixture", "fabricated")):
            raise ValueError("fixture research events must declare a fixture source")
        if any(not isinstance(event.get("source"), str)
               or not event["source"].lower().startswith(("fixture", "fabricated"))
               for event in envelope["events"]):
            raise ValueError("fixture research events must all use fixture sources")
    events = prepare_events(envelope["events"], market, cfg)
    provenance = {"events_path": str(event_path), "events_sha256": hashlib.sha256(raw).hexdigest(),
                  "events_metadata": envelope["metadata"], "market": market.get("provenance", {})}
    return events, market, provenance


def historical_graph(cfg: dict):
    events, market, _ = load_inputs(cfg)
    return build_graph(events, market, cfg)


def fetch(cfg: dict) -> Path:
    from data.edgar_documents import collect_filings
    path = resolve(cfg, cfg["research"]["events_path"])
    if path.exists():
        raise FileExistsError(f"events snapshot already exists: {path}; configure a new events_path")
    events = collect_filings(cfg)
    payload = {"metadata": {"data_kind": "historical", "source": "sec_edgar",
                            "retrieved_at": iso_utc(datetime.now(timezone.utc)),
                            "availability_mode": cfg["research"]["edgar"]["availability_mode"]},
               "events": events}
    # Keep normalized snapshots as immutable as the underlying raw bytes. Reuse an
    # identical payload, otherwise require a distinct path for a refreshed fetch.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return path


def run(cfg: dict) -> Path:
    from evaluation.walk_forward import evaluate_research
    events, market, provenance = load_inputs(cfg)
    settings = dict(cfg["research"]["evaluation"])
    settings.setdefault("seed", cfg.get("seed", 42))
    result = evaluate_research(events, market, settings, historical_source_config=cfg)
    run_id = cfg["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
        raise ValueError("run_id must be a simple directory name")
    output = resolve(cfg, cfg["research"]["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    G = build_graph(events, market, cfg)
    save_graph(G, output / "graph.json")
    write_json(output / "events.json", {"metadata": market["metadata"], "events": events})
    write_json(output / "evaluation.json", result)
    sources = ["research/features.py", "research/pipeline.py", "evaluation/walk_forward.py",
               "data/edgar_documents.py", "data/pit.py", "data/market_data.py",
               "signaling/signal_game.py", "signaling/types.py", "graph/ontology.py"]
    code_hashes = {s: hashlib.sha256((PKG_ROOT / s).read_bytes()).hexdigest() for s in sources}
    manifest = {
        "run_id": run_id, "created_at": iso_utc(datetime.now(timezone.utc)),
        "data_kind": market["metadata"]["data_kind"], "feature_version": FEATURE_VERSION,
        "events": len(events), "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
        "provenance": provenance, "code_sha256": code_hashes,
        "evaluation_settings": settings, "feature_assumptions": cfg["research"].get("features", {}),
        "signaling_assumptions": cfg.get("signaling", {}),
        "promotion": {"capital_deployment_allowed": False, "reasons": [
            "An independently reviewed prospective paper record is required.",
            "Direction and credibility rules have not been empirically calibrated.",
            "Single-source EDGAR cannot identify propagation or source-cost effects.",
        ]},
        "interpretation": "Historical reconstruction if data_kind=historical; fixture validates code only.",
    }
    write_json(output / "manifest.json", manifest)
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_RESEARCH_CONFIG))
    ap.add_argument("--fetch", action="store_true", help="collect real EDGAR documents to a new input snapshot")
    ap.add_argument("--demo", action="store_true", help="generate a labeled offline test fixture and evaluate it")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.demo and args.fetch:
        ap.error("--demo and --fetch are mutually exclusive")
    if args.demo:
        from research.fixture import create_fixture
        folder = resolve(cfg, cfg["research"]["output_dir"]) / (cfg["run_id"] + "-inputs")
        folder.mkdir(parents=True, exist_ok=False)
        events, market = create_fixture(cfg.get("seed", 42))
        write_json(folder / "events.json", events)
        write_json(folder / "market.json", market)
        cfg["research"]["events_path"] = str(folder / "events.json")
        cfg["research"]["market_path"] = str(folder / "market.json")
    output = fetch(cfg) if args.fetch else run(cfg)
    print(f"{'EDGAR snapshot' if args.fetch else 'Research artifacts'}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
