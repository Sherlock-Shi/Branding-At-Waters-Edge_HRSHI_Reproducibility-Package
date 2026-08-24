"""Production Party Brand Index measurement specification.

The specification begins with the promoted article-level aggregation and writes only
downstream statistical derivatives. It uses one joint within-month
article-cluster bootstrap so stance, polarization, raw IQR, message coherence,
and the Party Brand Index share the same resampling stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

from src.measurement_bootstrap import (
    BOOTSTRAP_SCHEMA,
    JointBootstrapConfig,
    run_joint_measurement_bootstrap,
)
from src.party_brand_index import PBI_SCHEMA, reconstruct_party_brand_index


PRODUCTION_SCHEMA = "paper3-party-brand-measurement-production-v1"
@dataclass
class PartyBrandMeasurementRun:
    """In-memory products from one measurement run."""

    monthly_stance: pd.DataFrame
    monthly_polarization: pd.DataFrame
    monthly_message_coherence: pd.DataFrame
    party_brand_index: pd.DataFrame
    measurement_replicates: pd.DataFrame
    party_brand_replicates: pd.DataFrame
    manifest: Dict[str, object]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = sha256()
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


def run_party_brand_measurement(
    article_file: Path | str,
    output_dir: Path | str,
    n_bootstrap: int = 5000,
    master_seed: int = 42,
    namespace: str = "joint_article_cluster_measurement",
) -> PartyBrandMeasurementRun:
    """Generate the validated production measurement specification.

    The observed stance and polarization formulas remain unchanged. Article
    count enters only through the joint resampling distribution. Every Party
    Brand Index replicate re-estimates raw IQR, the pooled-median calibration
    constant, and the maximum-polarization normalization.
    """

    article_path = Path(article_file).resolve()
    destination = Path(output_dir).resolve()
    if not article_path.exists():
        raise FileNotFoundError(article_path)
    destination.mkdir(parents=True, exist_ok=True)

    article_hash = sha256_file(article_path)
    article = pd.read_csv(article_path, dtype={"stable_article_id": "string"})
    config = JointBootstrapConfig(
        n_bootstrap=n_bootstrap,
        master_seed=master_seed,
        namespace=namespace,
    )
    measurement = run_joint_measurement_bootstrap(article, config)
    party_brand = reconstruct_party_brand_index(
        measurement.monthly_stance,
        measurement.monthly_polarization,
        measurement.replicates,
        ci_lower=config.ci_lower,
        ci_upper=config.ci_upper,
        min_valid_fraction=config.min_valid_fraction,
    )

    frames = {
        "monthly_stance_joint_bootstrap.csv": measurement.monthly_stance,
        "monthly_polarization_joint_bootstrap.csv": measurement.monthly_polarization,
        "monthly_measurement_bootstrap_replicates.csv": measurement.replicates,
        "monthly_message_coherence.csv": party_brand.monthly_coherence,
        "party_brand_index_timeseries.csv": party_brand.timeseries,
        "party_brand_index_bootstrap_replicates.csv": party_brand.replicates,
    }
    for filename, frame in frames.items():
        frame.to_csv(destination / filename, index=False)

    measurement_metadata = dict(measurement.metadata)
    measurement_metadata.update(
        {
            "contract_role": "active production measurement uncertainty",
            "input_file": str(article_path),
            "input_sha256": article_hash,
            "software": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    (destination / "measurement_bootstrap_metadata.json").write_text(
        json.dumps(
            measurement_metadata,
            indent=2,
            sort_keys=True,
            default=_json_ready,
        ),
        encoding="utf-8",
    )
    pbi_metadata = dict(party_brand.metadata)
    pbi_metadata["contract_role"] = "active production Party Brand Index"
    (destination / "party_brand_index_metadata.json").write_text(
        json.dumps(pbi_metadata, indent=2, sort_keys=True, default=_json_ready),
        encoding="utf-8",
    )

    run_payload = json.dumps(
        {
            "input_sha256": article_hash,
            "n_bootstrap": n_bootstrap,
            "master_seed": master_seed,
            "namespace": namespace,
            "schemas": [BOOTSTRAP_SCHEMA, PBI_SCHEMA, PRODUCTION_SCHEMA],
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest: Dict[str, object] = {
        "run_id": "party-brand-measurement-" + sha256(run_payload).hexdigest()[:16]
    }

    return PartyBrandMeasurementRun(
        monthly_stance=measurement.monthly_stance,
        monthly_polarization=measurement.monthly_polarization,
        monthly_message_coherence=party_brand.monthly_coherence,
        party_brand_index=party_brand.timeseries,
        measurement_replicates=measurement.replicates,
        party_brand_replicates=party_brand.replicates,
        manifest=manifest,
    )
