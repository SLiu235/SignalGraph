from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

from data.market_data import (_canonical_json_bytes, assemble_market_bundle,
                              fetch_alpha_vantage_adjusted, fetch_fama_french_daily,
                              load_market_bundle, parse_alpha_vantage_adjusted,
                              parse_fama_french_daily, validate_market_bundle,
                              verify_historical_factor_fragment,
                              verify_historical_market_provenance,
                              verify_historical_price_fragment, main)


def bundle():
    return {"metadata": {"data_kind": "fixture", "source": "offline test",
                         "adjustment": "split_and_dividend", "universe_policy": "fixed test issuer",
                         "retrieved_at": "2025-01-04T00:00:00Z"},
            "session_schedule": {"version": 1, "calendar_id": "fixture-xnys-v1",
                                 "source": "offline test calendar", "retrieved_at": "2025-01-01T00:00:00Z",
                                 "sessions": [
                                     {"session": "2025-01-02", "open_at": "2025-01-02T14:30:00Z", "close_at": "2025-01-02T21:00:00Z"},
                                     {"session": "2025-01-03", "open_at": "2025-01-03T14:30:00Z", "close_at": "2025-01-03T21:00:00Z"},
                                 ]},
            "bars": [{"ticker": "TEST", "session": "2025-01-02", "open_at": "2025-01-02T14:30:00Z",
                      "close_at": "2025-01-02T21:00:00Z", "adjusted_close": 100.0},
                     {"ticker": "TEST", "session": "2025-01-03", "open_at": "2025-01-03T14:30:00Z",
                      "close_at": "2025-01-03T21:00:00Z", "adjusted_close": 102.0}],
            "benchmark": [{"session": "2025-01-03", "return": 0.01}],
            "factors": [{"session": "2025-01-03", "mkt_rf": 0.009, "smb": 0.002,
                         "hml": -0.001, "rf": 0.001}]}


FACTOR_CSV = b"Example research data\n,Mkt-RF,SMB,HML,RF\n20250102,1.00,-0.20,0.30,0.01\n20250103,-0.50,0.10,0.20,0.01\n\nCopyright data provider\n"


class MarketBundleTests(unittest.TestCase):
    def test_valid_bundle_is_copied_and_file_bytes_are_hashed(self):
        source = bundle()
        validated = validate_market_bundle(source)
        validated["bars"][0]["adjusted_close"] = 999
        self.assertEqual(source["bars"][0]["adjusted_close"], 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.json"
            raw = json.dumps(source).encode()
            path.write_bytes(raw)
            result = load_market_bundle(path)
            self.assertEqual(result["provenance"]["input_sha256"], hashlib.sha256(raw).hexdigest())

    def test_adjustment_and_data_kind_are_explicit(self):
        for field, value in (("adjustment", "raw"), ("data_kind", "live"), ("universe_policy", "")):
            with self.subTest(field=field):
                data = bundle()
                data["metadata"][field] = value
                with self.assertRaises(ValueError):
                    validate_market_bundle(data)

    def test_bars_reject_bad_time_price_duplicates_and_order(self):
        for field, value in (("open_at", "2025-01-02T14:30:00"),
                             ("open_at", "2025-01-02T09:30:00-05:00"),
                             ("open_at", "2025-01-02T22:00:00Z"),
                             ("adjusted_close", float("nan")), ("adjusted_close", 0)):
            with self.subTest(field=field, value=value):
                data = bundle()
                data["bars"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_market_bundle(data)
        for rows in ([bundle()["bars"][0]] * 2, list(reversed(bundle()["bars"]))):
            data = bundle()
            data["bars"] = rows
            with self.assertRaises(ValueError):
                validate_market_bundle(data)

    def test_duplicate_factors_and_overlapping_memberships_fail(self):
        data = bundle()
        data["factors"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate factors"):
            validate_market_bundle(data)
        data = bundle()
        membership = {"ticker": "TEST", "sector": "Tech", "valid_from": "2025-01-01T00:00:00Z",
                      "valid_to": None, "available_at": "2024-12-31T00:00:00Z"}
        data["memberships"] = [membership, copy.deepcopy(membership)]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_market_bundle(data)

    def test_json_duplicate_keys_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"metadata":{},"metadata":{}}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                load_market_bundle(path)


class FactorAdapterTests(unittest.TestCase):
    def test_zip_and_percent_to_decimal_conversion(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("F-F_Research_Data_Factors_daily.CSV", FACTOR_CSV)
        rows = parse_fama_french_daily(stream.getvalue())
        self.assertEqual(rows[0]["mkt_rf"], 0.01)
        self.assertEqual(rows[0]["rf"], 0.0001)
        self.assertEqual(rows[1]["session"], "2025-01-03")

    def test_missing_sentinel_and_duplicate_dates_are_rejected(self):
        for data in (FACTOR_CSV.replace(b"1.00", b"-99.99"),
                     FACTOR_CSV.replace(b"20250103", b"20250102")):
            with self.assertRaises(ValueError):
                parse_fama_french_daily(data)

    def test_fetch_caches_replayable_raw_provenance_and_total_market_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = FACTOR_CSV
            result = fetch_fama_french_daily(tmp)
            factors, benchmark, receipt = verify_historical_factor_fragment(result)
            self.assertAlmostEqual(result["benchmark"][0]["return"], 0.0101)
            self.assertEqual(factors, result["factors"])
            self.assertEqual(benchmark, result["benchmark"])
            self.assertEqual(Path(receipt["raw_path"]).read_bytes(), FACTOR_CSV)
            self.assertEqual(result["metadata"]["input_sha256"], hashlib.sha256(FACTOR_CSV).hexdigest())
            self.assertNotIn("available_at", result["factors"][0])
            self.assertNotIn("available_at", result["benchmark"][0])

    def test_existing_cache_is_verified_without_replacing_corrupt_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = FACTOR_CSV
            result = fetch_fama_french_daily(tmp)
            _, _, receipt = verify_historical_factor_fragment(result)
            cache = Path(receipt["raw_path"])
            cache.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(result)
            with self.assertRaisesRegex(ValueError, "integrity"):
                fetch_fama_french_daily(tmp)
            self.assertEqual(cache.read_bytes(), b"corrupt")

    def test_factor_redirect_is_rejected_before_caching(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.geturl.return_value = "http://untrusted.example/factors.zip"
            with self.assertRaisesRegex(ValueError, "redirected away"):
                fetch_fama_french_daily(tmp)

    def test_historical_factor_replay_rejects_edited_rows_benchmark_and_availability_claims(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = FACTOR_CSV
            original = fetch_fama_french_daily(tmp)
            verify_historical_factor_fragment(original)

            changed_factor = copy.deepcopy(original)
            changed_factor["factors"][0]["mkt_rf"] += 0.01
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(changed_factor)

            changed_benchmark = copy.deepcopy(original)
            changed_benchmark["benchmark"][0]["return"] += 0.01
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(changed_benchmark)

            forged_availability = copy.deepcopy(original)
            forged_availability["benchmark"][0]["available_at"] = "2025-01-02T21:01:00Z"
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(forged_availability)

            sidecar = Path(original["metadata"]["raw_receipt_path"])
            sidecar_bytes = sidecar.read_bytes()
            sidecar.write_bytes(b"altered sidecar")
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(original)
            sidecar.write_bytes(sidecar_bytes)

            missing_raw_receipt = copy.deepcopy(original)
            missing_raw_receipt["metadata"].pop("raw_receipt")
            with self.assertRaises(ValueError):
                verify_historical_factor_fragment(missing_raw_receipt)


class BundleCompositionTests(unittest.TestCase):
    def test_composition_preserves_source_hashes_and_fixture_label(self):
        source = bundle()
        prices = {"metadata": source["metadata"], "bars": source["bars"]}
        factors = {"metadata": {**source["metadata"], "return_unit": "decimal"},
                   "benchmark": source["benchmark"], "factors": source["factors"]}
        with tempfile.TemporaryDirectory() as tmp:
            price_path, factor_path, schedule_path = (Path(tmp) / "prices.json", Path(tmp) / "factors.json",
                                                       Path(tmp) / "schedule.json")
            price_path.write_text(json.dumps(prices))
            factor_path.write_text(json.dumps(factors))
            schedule_path.write_text(json.dumps(source["session_schedule"]))
            result = assemble_market_bundle([price_path], factor_path, "fixed demonstration universe", schedule_path)
            self.assertEqual(result["metadata"]["data_kind"], "fixture")
            self.assertEqual(len(result["provenance"]["components"]), 3)
            price_component = next(item for item in result["provenance"]["components"]
                                   if item["role"] == "prices")
            self.assertEqual(price_component["input_sha256"],
                             hashlib.sha256(price_path.read_bytes()).hexdigest())
            composed = Path(tmp) / "composed.json"
            composed.write_text(json.dumps(result))
            loaded = load_market_bundle(composed)
            self.assertEqual(len(loaded["provenance"]["components"]), 3)
            self.assertEqual(loaded["provenance"]["input_sha256"], hashlib.sha256(composed.read_bytes()).hexdigest())
            prices["metadata"]["data_kind"] = "historical"
            price_path.write_text(json.dumps(prices))
            with self.assertRaisesRegex(ValueError, "raw_receipt"):
                assemble_market_bundle([price_path], factor_path, "selected issuers", schedule_path)
            prices["metadata"]["data_kind"] = "fixture"
            price_path.write_text(json.dumps(prices))
            with self.assertRaisesRegex(ValueError, "(price_lineage IDs|duplicate bar)"):
                assemble_market_bundle([price_path, price_path], factor_path, "selected issuers", schedule_path)
            prices["metadata"]["retrieved_at"] = "2025-01-02T12:00:00Z"
            price_path.write_text(json.dumps(prices))
            with self.assertRaisesRegex(ValueError, "fragment bars"):
                assemble_market_bundle([price_path], factor_path, "selected issuers", schedule_path)

    def test_cli_existing_output_prevents_download_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.fetch_fama_french_daily") as fetch:
            output = Path(tmp) / "factors.json"
            output.write_text("original")
            with self.assertRaises(FileExistsError):
                main(["fetch-factors", "--cache-dir", tmp, "--output", str(output)])
            fetch.assert_not_called()
            self.assertEqual(output.read_text(), "original")


class AdjustedPriceAdapterTests(unittest.TestCase):
    def test_requires_explicit_sessions_and_adjusted_close(self):
        payload = {"Meta Data": {"2. Symbol": "TEST"}, "Time Series (Daily)": {
            "2025-01-02": {"5. adjusted close": "50.0", "6. volume": "1000"},
            "2025-01-03": {"5. adjusted close": "51.0", "6. volume": "1200"}}}
        sessions = [{key: value for key, value in row.items() if key in ("session", "open_at", "close_at")}
                    for row in bundle()["bars"]]
        result = parse_alpha_vantage_adjusted(payload, "TEST", sessions)
        self.assertEqual(result[1]["adjusted_close"] / result[0]["adjusted_close"] - 1, 0.020000000000000018)
        with self.assertRaisesRegex(ValueError, "session table"):
            parse_alpha_vantage_adjusted(payload, "TEST", [])
        del payload["Time Series (Daily)"]["2025-01-02"]["5. adjusted close"]
        with self.assertRaisesRegex(ValueError, "missing adjusted"):
            parse_alpha_vantage_adjusted(payload, "TEST", sessions)

    def test_entitlement_response_does_not_become_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "entitlement"):
            parse_alpha_vantage_adjusted({"Information": "premium required"}, "TEST", [])

    def test_raw_provider_capture_replays_every_price_bar_and_never_stores_the_key(self):
        payload = {"Meta Data": {"2. Symbol": "TEST"}, "Time Series (Daily)": {
            "2025-01-02": {"5. adjusted close": "50.0", "6. volume": "1000"},
            "2025-01-03": {"5. adjusted close": "51.0", "6. volume": "1200"}}}
        sessions = [{key: value for key, value in row.items() if key in ("session", "open_at", "close_at")}
                    for row in bundle()["bars"]]
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.side_effect = [json.dumps(payload).encode(), FACTOR_CSV]
            prices = fetch_alpha_vantage_adjusted("TEST", "secret-key", sessions, cache_dir=Path(tmp) / "raw")
            expected, receipt = verify_historical_price_fragment(prices)
            self.assertEqual(expected, prices["bars"])
            self.assertNotIn("secret-key", json.dumps(prices))

            altered = copy.deepcopy(prices)
            altered["bars"][0]["adjusted_close"] = 999.0
            with self.assertRaisesRegex(ValueError, "differ from the immutable"):
                verify_historical_price_fragment(altered)

            # A self-consistent receipt for a different provider must not be
            # interpreted with the Alpha Vantage parser merely because it has
            # the same field shape and raw payload bytes.
            forged = copy.deepcopy(prices)
            forged_receipt = forged["metadata"]["raw_receipt"]
            forged_receipt["provider"] = "Unrelated provider"
            forged_sidecar = Path(tmp) / "forged-receipt.json"
            forged_sidecar.write_bytes(_canonical_json_bytes(forged_receipt))
            forged["metadata"]["raw_receipt_path"] = str(forged_sidecar)
            forged["metadata"]["raw_receipt_sha256"] = hashlib.sha256(
                _canonical_json_bytes(forged_receipt)).hexdigest()
            with self.assertRaisesRegex(ValueError, "sidecar|provider endpoint"):
                verify_historical_price_fragment(forged)

            for field, value in (("source", "Unrelated provider"),
                                 ("input_sha256", "0" * 64),
                                 ("usage", "unreviewed price data")):
                forged = copy.deepcopy(prices)
                forged["metadata"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata differs"):
                    verify_historical_price_fragment(forged)

            factors = fetch_fama_french_daily(Path(tmp) / "raw-factors")
            verify_historical_factor_fragment(factors)
            price_path, factor_path, schedule_path = (Path(tmp) / "prices.json", Path(tmp) / "factors.json",
                                                       Path(tmp) / "schedule.json")
            price_path.write_text(json.dumps(prices))
            factor_path.write_text(json.dumps(factors))
            schedule_path.write_text(json.dumps(bundle()["session_schedule"]))
            assembled = assemble_market_bundle([price_path], factor_path, "one reviewed test issuer", schedule_path)
            integrity = verify_historical_market_provenance(assembled)
            self.assertEqual(integrity["status"], "price_factor_replayed_only")
            self.assertEqual(integrity["replayed_components"], ["prices", "factors"])
            self.assertEqual(integrity["unreplayed_components"], ["session_schedule"])
            self.assertTrue(integrity["factor_receipt_hashes"])

            declared_controls = copy.deepcopy(assembled)
            declared_controls["memberships"] = [{"ticker": "TEST", "sector": "Technology",
                                                   "valid_from": "2025-01-01T00:00:00Z",
                                                   "valid_to": None,
                                                   "available_at": "2025-01-01T00:00:00Z"}]
            declared_controls["sector_returns"] = [
                {"session": "2025-01-03", "sector": "Technology", "return": 0.01}]
            control_scope = verify_historical_market_provenance(declared_controls)
            self.assertEqual(control_scope["unreplayed_components"],
                             ["session_schedule", "memberships", "sector_returns"])

            altered_factor_bundle = copy.deepcopy(assembled)
            altered_factor_bundle["factors"][0]["mkt_rf"] += 0.01
            with self.assertRaises(ValueError):
                verify_historical_market_provenance(altered_factor_bundle)

            altered_benchmark_bundle = copy.deepcopy(assembled)
            altered_benchmark_bundle["benchmark"][0]["return"] += 0.01
            with self.assertRaises(ValueError):
                verify_historical_market_provenance(altered_benchmark_bundle)

            # Receipt hashes are not merely labels: modifying the cached bytes
            # makes replay reject the retained historical component.
            Path(receipt["raw_path"]).write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "object hash mismatch"):
                verify_historical_market_provenance(assembled)

    def test_network_error_cannot_print_key_bearing_url(self):
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen", side_effect=URLError("url?apikey=secret")):
            with self.assertRaises(RuntimeError) as caught:
                fetch_alpha_vantage_adjusted("TEST", "secret", [], cache_dir=tmp)
            self.assertNotIn("secret", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_lowercase_symbol_is_canonicalized_to_a_replayable_provider_fragment(self):
        payload = {"Meta Data": {"2. Symbol": "TEST"}, "Time Series (Daily)": {
            "2025-01-02": {"5. adjusted close": "50.0", "6. volume": "1000"},
            "2025-01-03": {"5. adjusted close": "51.0", "6. volume": "1200"}}}
        sessions = [{key: value for key, value in row.items() if key in ("session", "open_at", "close_at")}
                    for row in bundle()["bars"]]
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            prices = fetch_alpha_vantage_adjusted("test", "secret-key", sessions, cache_dir=tmp)
            self.assertEqual([row["ticker"] for row in prices["bars"]], ["TEST", "TEST"])
            verify_historical_price_fragment(prices)

    def test_redirected_provider_response_is_rejected_before_caching(self):
        sessions = [{key: value for key, value in row.items() if key in ("session", "open_at", "close_at")}
                    for row in bundle()["bars"]]
        with tempfile.TemporaryDirectory() as tmp, patch("data.market_data.urlopen") as opener:
            opener.return_value.__enter__.return_value.geturl.return_value = "http://untrusted.example/query"
            with self.assertRaisesRegex(ValueError, "redirected away") as caught:
                fetch_alpha_vantage_adjusted("TEST", "secret-key", sessions, cache_dir=tmp)
            self.assertNotIn("secret-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
