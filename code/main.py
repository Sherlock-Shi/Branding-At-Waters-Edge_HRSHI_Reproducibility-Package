#!/usr/bin/env python3
"""Run the Paper 3 downstream analysis pipeline."""

import os
import re
import sys
from pathlib import Path

# On Windows, make bundled R and its libraries discoverable when running under
# a Conda environment. The guarded registration is a no-op on macOS and Linux.
_RUNTIME_DLL_HANDLES = []
if os.name == "nt":
    runtime_directories = [
        sys.prefix,
        os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
        os.path.join(sys.prefix, "Library", "usr", "bin"),
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Scripts"),
        os.path.join(sys.prefix, "bin"),
        os.path.join(sys.prefix, "Lib", "R", "bin", "x64"),
    ]
    for runtime_directory in runtime_directories:
        if not os.path.isdir(runtime_directory):
            continue
        os.environ["PATH"] = (
            runtime_directory + os.pathsep + os.environ.get("PATH", "")
        )
        if hasattr(os, "add_dll_directory"):
            _RUNTIME_DLL_HANDLES.append(os.add_dll_directory(runtime_directory))

    bundled_r_home = os.path.join(sys.prefix, "Lib", "R")
    if "R_HOME" not in os.environ and os.path.isdir(bundled_r_home):
        os.environ["R_HOME"] = bundled_r_home

import argparse
import pandas as pd
from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Pipeline paths
DATA_PATH = str(config.DATA_ROOT)
PIPELINE_C_PATH = DATA_PATH

UNSUPPORTED_FLAGS = (
    "--llm_submit",
    "--llm_status",
    "--llm_download",
    "--llm_wait",
    "--full",
    "--force",
)


# =============================================================================
# Utility Functions
# =============================================================================

def prompt_user(message: str) -> bool:
    """Prompt user for yes/no confirmation."""
    while True:
        response = input(f"{message} [y/n]: ").strip().lower()
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


# =============================================================================
# Step 3: Aggregation
# =============================================================================

def run_aggregation(auto_yes: bool = False, output_dir: str = None) -> bool:
    """
    Run LLM score aggregation.
    
    Returns True if aggregation is complete.
    """
    print("\n" + "=" * 70)
    print("STEP 3: AGGREGATION")
    print("=" * 70)
    
    from src.llm_aggregator import LLMAggregator
    aggregator = LLMAggregator(
        input_csv=os.path.join(
            DATA_PATH, "scored", "factiva_llmrated_scores.csv"
        ),
        output_dir=output_dir,
    )
    
    # Check for existing output
    monthly_stance_file = os.path.join(aggregator.OUTPUT_DIR, 'monthly_stance.csv')
    article_scores_file = os.path.join(aggregator.OUTPUT_DIR, 'article_level_scores.csv')
    
    if os.path.exists(monthly_stance_file) and os.path.exists(article_scores_file):
        print(f"\n✓ Aggregated data found:")
        print(f"  - {monthly_stance_file}")
        print(f"  - {article_scores_file}")
        
        if not auto_yes and not prompt_user("Re-run aggregation?"):
            print("   Skipping aggregation")
            return True
    
    print("\n   Running aggregation...")
    aggregator.run()
    print("\n✓ Aggregation complete")
    return True


# =============================================================================
# Step 4: Statistical Tests
# =============================================================================

def run_statistical_tests(
    input_dir: str = None,
    output_dir: str = None,
    run_dominance: bool = True,
):
    """Run statistical tests (always executes)."""
    print("\n" + "=" * 70)
    print("STEP 4: STATISTICAL TESTS")
    print("=" * 70)
    
    resolved_input = input_dir or os.path.join(PIPELINE_C_PATH, "aggregated")
    resolved_output = output_dir or os.path.join(PIPELINE_C_PATH, "statistics")
    article_file = os.path.join(resolved_input, "article_level_scores.csv")

    print("\nGenerating the joint article-cluster bootstrap...")
    from src.party_brand_measurement import run_party_brand_measurement

    measurement_run = run_party_brand_measurement(
        article_file=article_file,
        output_dir=resolved_output,
        n_bootstrap=5000,
        master_seed=42,
    )
    print(f"Measurement run ID: {measurement_run.manifest['run_id']}")

    from src.llm_statistical_tests import LLMStatisticalTests
    stats = LLMStatisticalTests(
        input_dir=resolved_input,
        output_dir=resolved_output,
        run_dominance=run_dominance,
    )
    stats.run()

    print("\nGenerating the direct paired RQ2 Sen-slope contrast...")
    from src.party_slope_contrast import write_party_slope_contrast

    write_party_slope_contrast(
        Path(resolved_input) / "monthly_stance.csv",
        Path(resolved_input) / "monthly_polarization.csv",
        Path(resolved_output) / "trend_tests.csv",
        Path(resolved_output),
    )
    print(
        "\nRunning the joint common-AR regime-likelihood search "
        "with location-adjusted BIC..."
    )
    from scripts.joint_ar_breakpoint_analysis import (
        run_joint_ar_breakpoint_analysis,
    )

    monthly_polarization = pd.read_csv(
        os.path.join(resolved_input, "monthly_polarization.csv")
    )
    party_brand_index = pd.read_csv(
        os.path.join(resolved_output, "party_brand_index_timeseries.csv")
    )
    run_joint_ar_breakpoint_analysis(
        monthly_polarization=monthly_polarization,
        party_brand_index=party_brand_index,
        statistics_output_dir=resolved_output,
        runs_per_mode=5,
        generate_plots=True,
    )

    measurement_replicates = os.path.join(
        resolved_output, "monthly_measurement_bootstrap_replicates.csv"
    )
    polarization_file = os.path.join(resolved_input, "monthly_polarization.csv")

    print("\nRunning the party-aware polarization and message-coherence model...")
    from src.h3_party_coherence import run_h3_analysis

    run_h3_analysis(
        polarization_file=polarization_file,
        coherence_file=os.path.join(
            resolved_output, "monthly_message_coherence.csv"
        ),
        measurement_replicate_file=measurement_replicates,
        output_dir=resolved_output,
    )
    if run_dominance:
        print("\nRunning dependence-preserving multivariate dominance analysis...")
        from src.dominance_time_series import run_dominance_analysis

        run_dominance_analysis(
            dominance_file=os.path.join(resolved_output, "dominance_ready.csv"),
            measurement_replicate_file=measurement_replicates,
            output_dir=resolved_output,
            master_seed=42,
        )
    print("\nComparing trend-only, break-only, and break-plus-trend models...")
    from scripts.trend_break_analysis import run_trend_break_analysis

    run_trend_break_analysis(
        polarization_file=polarization_file,
        statistics_dir=resolved_output,
    )
    print("\n✓ Statistical tests complete")


# =============================================================================
# Step 5: Visualization
# =============================================================================

def run_visualization(
    aggregated_dir: str = None,
    statistics_dir: str = None,
    output_dir: str = None,
):
    """Run visualization generation (always executes)."""
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZATION")
    print("=" * 70)
    
    from src.llm_visualizations import LLMVisualizations
    viz = LLMVisualizations(
        aggregated_dir=aggregated_dir,
        statistics_dir=statistics_dir,
        output_dir=output_dir,
    )
    if not viz.run():
        raise RuntimeError("Visualization pipeline did not complete")
    print("\n✓ Visualization complete")


def downstream_directories(output_root: str = None):
    """Resolve downstream directories."""
    root = (
        validate_downstream_output_root(output_root)
        if output_root is not None
        else PIPELINE_C_PATH
    )
    return (
        os.path.join(root, "aggregated"),
        os.path.join(root, "statistics"),
        os.path.join(root, "statistics"),
    )


def validate_downstream_output_root(output_root: str) -> str:
    """Resolve a custom root inside the approved stable-ID derivative namespace."""
    resolved = Path(output_root).expanduser().resolve(strict=False)
    approved_root = Path(DATA_PATH, "stable_id_migration").resolve(strict=False)
    try:
        relative = resolved.relative_to(approved_root)
    except ValueError as exc:
        raise ValueError(
            f"Custom output root {resolved} is outside the approved versioned "
            f"derivative namespace {approved_root}. Omit --output_root for the "
            "canonical production route."
        ) from exc
    if not relative.parts:
        raise ValueError(
            "Custom output root must be a versioned child directory inside "
            f"{approved_root}, not the namespace root itself."
        )
    if relative.parts[0].casefold() == "corrected_run":
        raise ValueError(
            "The controlled corrected_run tree and all of its descendants are "
            "not custom rerun targets. Choose a separate versioned child of "
            f"{approved_root}."
        )
    if re.fullmatch(r".+(?:_v|-v)\d+", resolved.name, re.IGNORECASE) is None:
        raise ValueError(
            "Custom output root must end with an explicit version suffix such as "
            f"_v1 or -v1: {resolved}"
        )
    return str(resolved)


def validate_fresh_downstream_output_root(output_root: str) -> str:
    """Require a custom full-run root that does not already exist."""
    resolved = validate_downstream_output_root(output_root)
    if Path(resolved).exists():
        raise ValueError(
            "Custom full-run output root already exists and cannot be reused: "
            f"{resolved}. Choose a new versioned root so stale or partial outputs "
            "cannot enter the run."
        )
    return resolved


def reserve_fresh_downstream_output_root(output_root: str) -> str:
    """Atomically reserve a validated custom full-run root."""
    resolved = validate_fresh_downstream_output_root(output_root)
    try:
        Path(resolved).mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            "Custom full-run output root was created by another process before "
            f"reservation completed: {resolved}"
        ) from exc
    return resolved


def validate_skip_dominance_output_root(output_root: str | None) -> str:
    """Require smoke runs to use an explicit approved derivative root."""
    if output_root is None:
        raise ValueError(
            '--skip_dominance is smoke-only and requires an explicit '
            'non-production --output_root.'
        )
    return validate_downstream_output_root(output_root)


def _run_downstream_stages(output_root: str | None, run_dominance: bool) -> None:
    """Execute downstream stages after the output boundary has been prepared."""
    aggregated_dir, statistics_dir, visualization_dir = downstream_directories(
        output_root
    )
    run_aggregation(auto_yes=True, output_dir=aggregated_dir)
    run_statistical_tests(
        input_dir=aggregated_dir,
        output_dir=statistics_dir,
        run_dominance=run_dominance,
    )
    run_visualization(
        aggregated_dir=aggregated_dir,
        statistics_dir=statistics_dir,
        output_dir=visualization_dir,
    )


def run_downstream(output_root: str = None, run_dominance: bool = True):
    """Run downstream stages from the scored sentence file."""
    if output_root is not None:
        if not run_dominance:
            validate_skip_dominance_output_root(output_root)
        output_root = reserve_fresh_downstream_output_root(output_root)
    elif not run_dominance:
        validate_skip_dominance_output_root(None)
    _run_downstream_stages(output_root, run_dominance)


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: list[str] | None = None):
    """Parse command-line arguments and run the selected stage."""
    parser = argparse.ArgumentParser(
        description='UKRAID Politicization downstream analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/main.py --downstream --data-root <path>
  python code/main.py --llm_aggregate --data-root <path>
  python code/main.py --llm_stats --data-root <path>
  python code/main.py --llm_viz --data-root <path>
        """
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing the data/ tree.",
    )
    
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--llm_aggregate",
        action="store_true",
        help="Run aggregation",
    )
    actions.add_argument(
        "--llm_stats",
        action="store_true",
        help="Run statistical tests",
    )
    actions.add_argument(
        "--llm_viz",
        action="store_true",
        help="Generate visualizations",
    )
    actions.add_argument(
        "--downstream",
        action="store_true",
        help="Run aggregation, statistics, and visualization from the scored sentence file",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Optional versioned derivative root ending in _vN or -vN below "
            "data/stable_id_migration. Omit for data/aggregated and data/statistics. "
            "Full and aggregation-only custom "
            "runs require a new root that does not already exist."
        ),
    )
    parser.add_argument(
        "--check_output_root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check_skip_dominance_contract",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip_dominance",
        action="store_true",
        help="Skip the dependence-preserving dominance analysis during a smoke run",
    )

    raw_args = list(sys.argv[1:] if argv is None else argv)
    requested_upstream = [
        token for token in raw_args if token in UNSUPPORTED_FLAGS
    ]
    if requested_upstream:
        parser.error(
            "Flags not available in this distribution: " + ", ".join(requested_upstream)
        )
    args = parser.parse_args(raw_args)
    config.set_data_root(args.data_root)
    global DATA_PATH, PIPELINE_C_PATH
    DATA_PATH = str(config.DATA_ROOT)
    PIPELINE_C_PATH = DATA_PATH

    if args.check_output_root is not None:
        try:
            resolved_probe_root = validate_downstream_output_root(args.check_output_root)
        except ValueError as exc:
            parser.error(str(exc))
        print(f'Approved derivative output root: {resolved_probe_root}')
        return 0

    if args.check_skip_dominance_contract:
        try:
            validate_skip_dominance_output_root(None)
        except ValueError as exc:
            parser.error(str(exc))
        print('Unexpectedly accepted a canonical skip-dominance smoke run.')
        return 1

    safe_action_requested = any(
        (
            args.downstream,
            args.llm_aggregate,
            args.llm_stats,
            args.llm_viz,
        )
    )
    if not safe_action_requested:
        print('No action selected. The active CLI performs no upstream operation by default.')
        parser.print_help()
        return 0

    if args.output_root is not None:
        try:
            args.output_root = validate_downstream_output_root(args.output_root)
            if args.downstream or args.llm_aggregate:
                args.output_root = validate_fresh_downstream_output_root(
                    args.output_root
                )
        except ValueError as exc:
            parser.error(str(exc))

    if args.skip_dominance:
        if not args.downstream:
            parser.error(
                "--skip_dominance applies only to a fresh --downstream smoke run."
            )
        try:
            args.output_root = validate_skip_dominance_output_root(args.output_root)
        except ValueError as exc:
            parser.error(str(exc))
    
    print("=" * 70)
    print("UKRAID POLITICIZATION ANALYSIS PIPELINE")
    print("=" * 70)
    
    if args.downstream:
        run_downstream(
            output_root=args.output_root,
            run_dominance=not args.skip_dominance,
        )
        return
    
    if args.llm_aggregate:
        aggregation_root = (
            reserve_fresh_downstream_output_root(args.output_root)
            if args.output_root else None
        )
        run_aggregation(
            auto_yes=True,
            output_dir=(
                os.path.join(aggregation_root, "aggregated")
                if aggregation_root else None
            ),
        )
        return
    
    if args.llm_stats:
        run_statistical_tests(
            input_dir=(
                os.path.join(args.output_root, "aggregated")
                if args.output_root else None
            ),
            output_dir=(
                os.path.join(args.output_root, "statistics")
                if args.output_root else None
            ),
            run_dominance=not args.skip_dominance,
        )
        return
    
    if args.llm_viz:
        run_visualization(
            aggregated_dir=(
                os.path.join(args.output_root, "aggregated")
                if args.output_root else None
            ),
            statistics_dir=(
                os.path.join(args.output_root, "statistics")
                if args.output_root else None
            ),
            output_dir=(
                os.path.join(args.output_root, "statistics")
                if args.output_root else None
            ),
        )
        return
    

if __name__ == "__main__":
    main()
