"""Generate the direct paired RQ2 Sen-slope contrast."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = REPOSITORY_ROOT / "LLM_Pipeline"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from src.party_slope_contrast import write_party_slope_contrast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "UKRAID-Politicization"
        / "Factiva_Data"
        / "pipeline_c"
        / "llm_statistics",
    )
    parser.add_argument(
        "--aggregation-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "UKRAID-Politicization"
        / "Factiva_Data"
        / "pipeline_c"
        / "llm_aggregated",
    )
    args = parser.parse_args()
    output, metadata = write_party_slope_contrast(
        args.aggregation_dir / "monthly_stance.csv",
        args.aggregation_dir / "monthly_polarization.csv",
        args.statistics_dir / "trend_tests.csv",
        args.statistics_dir,
    )
    print(f"Wrote {output}")
    print(f"Wrote {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
