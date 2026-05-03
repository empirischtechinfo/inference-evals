"""
Compare results from multiple model evaluations.

This script reads the JSON result files and creates comparison tables/charts.

Usage:
    python compare_models.py --results_dir ./results
"""

import argparse
import json
import os
import glob
from typing import Dict, List
import csv


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result JSON files from directory."""
    results = {}
    pattern = os.path.join(results_dir, "medical_eval_*.json")

    for file_path in glob.glob(pattern):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                model_name = data.get("model", os.path.basename(file_path))
                results[model_name] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return results


def create_comparison_table(results: Dict[str, Dict]) -> str:
    """Create a markdown comparison table."""
    if not results:
        return "No results found."

    # Get all datasets
    all_datasets = set()
    for data in results.values():
        all_datasets.update(data.get("datasets", {}).keys())
    all_datasets = sorted(all_datasets)

    # Build table
    lines = [
        "\n" + "=" * 80,
        "📊 MODEL COMPARISON",
        "=" * 80,
        "",
    ]

    # Header
    header = "| Dataset |"
    separator = "|---------|"
    model_names = sorted(results.keys())

    for model in model_names:
        header += f" {model[:20]:<20} |"
        separator += "---------------------|"

    lines.extend([header, separator])

    # Data rows
    for dataset in all_datasets:
        row = f"| {dataset:<20} |"
        for model in model_names:
            data = results[model]
            if dataset in data.get("datasets", {}):
                acc = data["datasets"][dataset]["accuracy"]
                row += f" {acc:>19.2f}% |"
            else:
                row += f" {'N/A':>20} |"
        lines.append(row)

    # Overall row
    overall_row = "| **OVERALL** |"
    for model in model_names:
        acc = results[model].get("overall_accuracy", 0)
        overall_row += f" **{acc:>17.2f}%** |"
    lines.append(overall_row)

    return "\n".join(lines)


def export_leaderboard_csv(results: Dict[str, Dict], output_path: str):
    """Export a leaderboard-style CSV."""
    # Get all datasets
    all_datasets = set()
    for data in results.values():
        all_datasets.update(data.get("datasets", {}).keys())
    all_datasets = sorted(all_datasets)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Model"] + all_datasets + ["Overall Average"]
        writer.writerow(header)

        # Rows
        for model_name in sorted(results.keys()):
            data = results[model_name]
            row = [model_name]

            for dataset in all_datasets:
                if dataset in data.get("datasets", {}):
                    acc = data["datasets"][dataset]["accuracy"]
                    row.append(f"{acc:.2f}")
                else:
                    row.append("N/A")

            row.append(f"{data.get('overall_accuracy', 0):.2f}")
            writer.writerow(row)

    print(f"📊 Leaderboard exported to: {output_path}")


def print_ranking(results: Dict[str, Dict]):
    """Print model ranking by overall accuracy."""
    rankings = []
    for model_name, data in results.items():
        rankings.append((model_name, data.get("overall_accuracy", 0)))

    rankings.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 50)
    print("🏆 MODEL RANKINGS (by Overall Accuracy)")
    print("=" * 50)

    for i, (model, score) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {model:<30} {score:>6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare medical evaluation results across multiple models"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/leaderboard.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    # Load all results
    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        print("Expected files: medical_eval_*.json")
        return

    print(f"\nLoaded results for {len(results)} model(s)")

    # Print comparison table
    print(create_comparison_table(results))

    # Print rankings
    print_ranking(results)

    # Export leaderboard CSV
    export_leaderboard_csv(results, args.output)


if __name__ == "__main__":
    main()
