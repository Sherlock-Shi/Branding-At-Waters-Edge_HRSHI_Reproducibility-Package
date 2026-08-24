"""Run the isolated joint-measurement and Party Brand Index pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = REPO_ROOT / "code"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from src.measurement_bootstrap import (  # noqa: E402
    JointBootstrapConfig,
    run_joint_measurement_bootstrap,
)
from src.party_brand_index import reconstruct_party_brand_index  # noqa: E402


DEFAULT_ARTICLE_FILE = (
    REPO_ROOT
    / "data"
    / "aggregated"
    / "article_level_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "statistics"
    / "party_brand_index_pilot"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _prefix_values(n_bootstrap: int) -> Iterable[int]:
    candidates = [100, 200, 300, 400, 500, 1000, 2000, 5000]
    values = [value for value in candidates if value < n_bootstrap]
    values.append(n_bootstrap)
    return sorted(set(values))


def bootstrap_prefix_stability(
    measurement_replicates: pd.DataFrame,
    pbi_replicates: pd.DataFrame,
    ci_lower: float,
    ci_upper: float,
) -> pd.DataFrame:
    """Calculate nested-prefix diagnostics without treating them as proof of B."""

    measurement_parts = []
    for value_column in (
        "rep_mean",
        "dem_mean",
        "polarization",
        "rep_raw_iqr",
        "dem_raw_iqr",
    ):
        part = measurement_replicates[
            ["replicate", "year_month", "dimension", value_column]
        ].rename(columns={value_column: "value"})
        part["metric"] = part["dimension"] + "_" + value_column
        measurement_parts.append(
            part[["replicate", "year_month", "metric", "value"]]
        )
    pbi = pbi_replicates[
        ["replicate", "year_month", "party_brand_index"]
    ]
    long = pd.concat(
        [
            *measurement_parts,
            pbi.rename(columns={"party_brand_index": "value"}).assign(
                metric="party_brand_index"
            ),
        ],
        ignore_index=True,
    )
    n_bootstrap = int(long["replicate"].max())
    rows = []
    for (year_month, metric), group in long.groupby(
        ["year_month", "metric"], sort=True
    ):
        group = group.sort_values("replicate", kind="stable")
        full = group.loc[group["replicate"].le(n_bootstrap), "value"].dropna()
        full_lower, full_upper = np.percentile(full, [ci_lower, ci_upper])
        full_length = full_upper - full_lower
        for prefix in _prefix_values(n_bootstrap):
            values = group.loc[group["replicate"].le(prefix), "value"].dropna()
            lower, upper = np.percentile(values, [ci_lower, ci_upper])
            length = upper - lower
            relative_length_deviation = (
                abs(length - full_length) / abs(full_length)
                if full_length != 0
                else np.nan
            )
            endpoint_scale = max(abs(full_lower), abs(full_upper), 1e-12)
            rows.append(
                {
                    "year_month": str(year_month),
                    "metric": metric,
                    "prefix_replicates": int(prefix),
                    "valid_replicates": int(len(values)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_length": float(length),
                    "full_pilot_ci_lower": float(full_lower),
                    "full_pilot_ci_upper": float(full_upper),
                    "full_pilot_ci_length": float(full_length),
                    "relative_length_deviation_from_full_pilot": float(
                        relative_length_deviation
                    ),
                    "maximum_relative_endpoint_deviation_from_full_pilot": float(
                        max(abs(lower - full_lower), abs(upper - full_upper))
                        / endpoint_scale
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "year_month", "prefix_replicates"], kind="stable"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-file", type=Path, default=DEFAULT_ARTICLE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument(
        "--stream-namespace",
        default="joint_article_cluster_measurement",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    article_file = args.article_file.resolve()
    output_dir = args.output_dir.resolve()
    if not article_file.exists():
        raise FileNotFoundError(article_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    article_sha256 = sha256_file(article_file)
    article_df = pd.read_csv(article_file, dtype={"stable_article_id": "string"})
    config = JointBootstrapConfig(
        n_bootstrap=args.n_bootstrap,
        master_seed=args.master_seed,
        namespace=args.stream_namespace,
    )
    measurement = run_joint_measurement_bootstrap(article_df, config)
    pbi = reconstruct_party_brand_index(
        measurement.monthly_stance,
        measurement.monthly_polarization,
        measurement.replicates,
        ci_lower=config.ci_lower,
        ci_upper=config.ci_upper,
        min_valid_fraction=config.min_valid_fraction,
    )
    prefix_stability = bootstrap_prefix_stability(
        measurement.replicates,
        pbi.replicates,
        config.ci_lower,
        config.ci_upper,
    )

    outputs: Dict[str, pd.DataFrame] = {
        "monthly_stance_joint_bootstrap.csv": measurement.monthly_stance,
        "monthly_polarization_joint_bootstrap.csv": measurement.monthly_polarization,
        "monthly_measurement_bootstrap_replicates.csv": measurement.replicates,
        "party_brand_index_timeseries.csv": pbi.timeseries,
        "monthly_message_coherence.csv": pbi.monthly_coherence,
        "party_brand_index_bootstrap_replicates.csv": pbi.replicates,
        "bootstrap_prefix_stability.csv": prefix_stability,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    measurement_metadata = dict(measurement.metadata)
    measurement_metadata.update(
        {
            "input_file": str(article_file),
            "input_sha256": article_sha256,
            "software": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    (output_dir / "measurement_bootstrap_metadata.json").write_text(
        json.dumps(measurement_metadata, indent=2, sort_keys=True, default=_json_ready),
        encoding="utf-8",
    )
    (output_dir / "party_brand_index_metadata.json").write_text(
        json.dumps(pbi.metadata, indent=2, sort_keys=True, default=_json_ready),
        encoding="utf-8",
    )

    runtime_seconds = time.perf_counter() - started
    print(f"Joint measurement pilot written to: {output_dir}")
    print(f"Rows: stance={len(measurement.monthly_stance)}, polarization={len(measurement.monthly_polarization)}")
    print(f"Rows: measurement replicates={len(measurement.replicates)}, PBI={len(pbi.timeseries)}")
    print(f"Point k={pbi.metadata['point_k']:.6f}")
    print(f"Point max |polarization|={pbi.metadata['point_max_polarization']:.6f}")
    print(f"Runtime seconds={runtime_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
