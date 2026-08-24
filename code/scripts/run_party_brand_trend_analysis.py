#!/usr/bin/env python3
"""Run the complete Party Brand Index temporal-trend reporting specification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = REPO_ROOT / "LLM_Pipeline"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from src.trend_analysis import hamed_rao_sen_result  # noqa: E402


DEFAULT_PBI_FILE = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "stable_id_migration"
    / "corrected_run"
    / "method_updates"
    / "party_brand_index_v1_accuracy_5000"
    / "party_brand_index_timeseries.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "UKRAID-Politicization"
    / "Factiva_Data"
    / "stable_id_migration"
    / "corrected_run"
    / "method_updates"
    / "party_brand_index_breakpoints_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbi-file", type=Path, default=DEFAULT_PBI_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag", type=int, default=3)
    args = parser.parse_args()
    pbi_file = args.pbi_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(pbi_file).sort_values("year_month", kind="stable")
    expected = pd.date_range("2022-03-01", "2024-12-01", freq="MS")
    if len(frame) != 34 or not pd.DatetimeIndex(
        pd.to_datetime(frame["year_month"])
    ).equals(expected):
        raise ValueError("Party Brand Index must cover all 34 expected months")
    result = hamed_rao_sen_result(
        frame["party_brand_index"].to_numpy(dtype=float),
        series_name="Party Brand Index",
        lag=args.lag,
        scale="Party Brand Index points per month",
    ).values
    result["period_start"] = "2022-03"
    result["period_end"] = "2024-12"
    result["validation_status"] = "pending independent numerical validation"
    pd.DataFrame([result]).to_csv(
        output_dir / "party_brand_index_trend.csv", index=False
    )
    metadata = {
        "schema": "paper3-party-brand-trend-analysis-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending independent numerical validation",
        "input_file": str(pbi_file),
        "input_sha256": sha256_file(pbi_file),
        "lag_selection": (
            "lag 3 retained because the reconstructed raw series has its only "
            "approximately significant raw-series autocorrelation within the "
            "first three lags at lag 3; the Hamed-Rao calculation itself uses "
            "significant detrended-rank autocorrelations"
        ),
        "sen_reference": {
            "authors": "Pranab Kumar Sen",
            "year": 1968,
            "title": "Estimates of the Regression Coefficient Based on Kendall's Tau",
            "journal": "Journal of the American Statistical Association",
            "volume_issue_pages": "63(324), 1379-1389",
            "doi": "10.1080/01621459.1968.10480934",
        },
        "hamed_rao_reference": {
            "authors": "K. H. Hamed and A. Ramachandra Rao",
            "year": 1998,
            "title": "A modified Mann-Kendall trend test for autocorrelated data",
            "journal": "Journal of Hydrology",
            "volume_issue_pages": "204(1-4), 182-196",
            "doi": "10.1016/S0022-1694(97)00125-X",
        },
    }
    (output_dir / "party_brand_index_trend_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame([result]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

