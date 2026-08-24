"""Post-aggregation statistical analysis for Paper 3."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import DATA_PATH
from src.trend_analysis import hamed_rao_sen_result


EXPECTED_MONTHS = pd.period_range("2022-03", "2024-12", freq="M").astype(str)


class TeeLogger:
    """Write a concise run report to the terminal and a UTF-8 file."""

    def __init__(self, filepath: str) -> None:
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


class LLMStatisticalTests:
    """Generate the active statistical reporting inputs after aggregation."""

    INPUT_DIR = os.path.join(DATA_PATH, "aggregated")
    OUTPUT_DIR = os.path.join(DATA_PATH, "statistics")
    DIMENSIONS = ("q1", "q2", "q3", "q4")

    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        run_dominance: bool = False,
    ) -> None:
        self.INPUT_DIR = input_dir or self.INPUT_DIR
        self.OUTPUT_DIR = output_dir or self.OUTPUT_DIR
        self.run_dominance = run_dominance
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.LOG_FILE = os.path.join(
            self.OUTPUT_DIR,
            "statistical_tests_report.txt",
        )

    def run(self) -> Dict[str, object]:
        """Run the active reporting-input contract."""
        logger = TeeLogger(self.LOG_FILE)
        original_stdout = sys.stdout
        sys.stdout = logger
        try:
            results = self._run_active_contract()
        finally:
            sys.stdout = original_stdout
            logger.close()
        print(f"[REPORT] Active statistics report: {self.LOG_FILE}")
        return results

    def _run_active_contract(self) -> Dict[str, object]:
        print("=" * 72)
        print("PAPER 3 ACTIVE POST-AGGREGATION STATISTICAL CONTRACT")
        print("=" * 72)
        print(f"Generated UTC: {datetime.now(timezone.utc).isoformat()}")

        stance = self._load_complete_csv(
            Path(self.INPUT_DIR) / "monthly_stance.csv",
            {"year_month", "party", "dimension", "mean", "raw_iqr", "n_articles"},
            272,
        )
        polarization = self._load_complete_csv(
            Path(self.INPUT_DIR) / "monthly_polarization.csv",
            {
                "year_month",
                "dimension",
                "rep_mean",
                "dem_mean",
                "polarization",
                "rep_n",
                "dem_n",
            },
            136,
        )
        pbi = self._load_complete_csv(
            Path(self.OUTPUT_DIR) / "party_brand_index_timeseries.csv",
            {"year_month", "party_brand_index"},
            34,
        )
        trend = self._write_trend_contract(stance, polarization, pbi)
        dominance_ready = self._write_dominance_ready(polarization)

        print("Observed aggregation: raw-IQR production contract")
        print("Measurement uncertainty: joint 5,000 article-cluster replicates")
        print("Trend rows written: 4")
        return {
            "trend_tests": trend,
            "dominance_ready": dominance_ready,
        }

    @staticmethod
    def _load_complete_csv(
        path: Path,
        required_columns: set[str],
        expected_rows: int,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        if len(frame) != expected_rows:
            raise ValueError(
                f"{path.name} has {len(frame)} rows; expected {expected_rows}"
            )
        months = np.sort(frame["year_month"].astype(str).unique())
        if not np.array_equal(months, EXPECTED_MONTHS):
            raise ValueError(f"{path.name} does not contain the complete calendar")
        return frame

    def _write_trend_contract(
        self,
        stance: pd.DataFrame,
        polarization: pd.DataFrame,
        pbi: pd.DataFrame,
    ) -> pd.DataFrame:
        q1_pol = (
            polarization.loc[polarization["dimension"].eq("q1")]
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        q1_stance = stance.loc[stance["dimension"].eq("q1")].copy()
        republican = (
            q1_stance.loc[q1_stance["party"].eq("republican")]
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        democrat = (
            q1_stance.loc[q1_stance["party"].eq("democrat")]
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        pbi = pbi.sort_values("year_month").reset_index(drop=True)
        specifications = (
            (
                q1_pol["polarization"].to_numpy(float),
                "Q1 polarization",
                2,
                "Q1 polarization points per month",
            ),
            (
                republican["mean"].to_numpy(float),
                "Q1 Republican stance",
                2,
                "Q1 stance points per month",
            ),
            (
                democrat["mean"].to_numpy(float),
                "Q1 Democratic stance",
                2,
                "Q1 stance points per month",
            ),
            (
                pbi["party_brand_index"].to_numpy(float),
                "Party Brand Index",
                3,
                "Party Brand Index points per month",
            ),
        )
        rows = []
        for values, name, lag, scale in specifications:
            row = dict(
                hamed_rao_sen_result(
                    values,
                    series_name=name,
                    lag=lag,
                    scale=scale,
                ).values
            )
            row.update({"period_start": "2022-03", "period_end": "2024-12"})
            rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(Path(self.OUTPUT_DIR) / "trend_tests.csv", index=False)
        return result

    def _write_dominance_ready(self, polarization: pd.DataFrame) -> pd.DataFrame:
        pivot = polarization.pivot(
            index="year_month",
            columns="dimension",
            values="polarization",
        ).reset_index()
        pivot = pivot[["year_month", *self.DIMENSIONS]].sort_values("year_month")
        pivot = pivot.rename(
            columns={
                "q1": "q1_polarization",
                "q2": "q2_polarization",
                "q3": "q3_polarization",
                "q4": "q4_polarization",
            }
        )
        if len(pivot) != 34 or pivot.isna().any(axis=None):
            raise ValueError("dominance-ready matrix is incomplete")
        pivot.to_csv(Path(self.OUTPUT_DIR) / "dominance_ready.csv", index=False)
        return pivot

def main() -> None:
    LLMStatisticalTests(run_dominance=False).run()


if __name__ == "__main__":
    main()
