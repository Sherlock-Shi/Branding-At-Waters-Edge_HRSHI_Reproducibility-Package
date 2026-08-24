"""Build Paper 3 tables and publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = REPOSITORY_ROOT / "code"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from src.plot_style import validate_style_contract
from src.publication_contract import load_publication_inputs
from src.publication_figures import build_publication_figures
from src.publication_tables import write_table_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    args = parser.parse_args()
    output_directory = args.output_dir.resolve()
    if output_directory != REPOSITORY_ROOT.resolve():
        raise RuntimeError("publication outputs must remain inside the deposit repository")

    style_failures = validate_style_contract()
    if style_failures:
        raise RuntimeError("publication style contract failed: " + "; ".join(style_failures))
    inputs = load_publication_inputs(output_directory / "data")
    tables_directory = output_directory / "data" / "statistics"
    figures_directory = output_directory / "manuscript" / "figures"

    table_outputs, blueprint_path = write_table_contract(inputs, tables_directory)
    figure_outputs, figure_specs_path = build_publication_figures(inputs, figures_directory)
    print(f"Built {len(table_outputs)} machine-readable table files")
    print(f"Built {len(figure_outputs) // 2} figures in PNG and PDF")
    print(f"Wrote {blueprint_path}")
    print(f"Wrote {figure_specs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
