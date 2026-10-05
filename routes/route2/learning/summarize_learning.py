"""Aggregate training seeds as independent runs and optionally plot validation curves."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    runs = []
    for path in paths:
        report = json.loads(path.read_text())
        tests = [
            snapshot
            for snapshot in report["evaluations"]
            if snapshot["split"] == "test"
        ]
        if report["status"] != "completed" or len(tests) != 1:
            raise ValueError(f"Incomplete run or unexpected test count: {path}")
        test = tests[0]
        pairs = list(zip(test["initial_policy"], test["trained_policy"]))
        if any(initial["seed"] != trained["seed"] for initial, trained in pairs):
            raise ValueError(f"Unpaired test seeds: {path}")
        initial_returns = [initial["return"] for initial, _ in pairs]
        final_returns = [trained["return"] for _, trained in pairs]
        gain = statistics.mean(final_returns) - statistics.mean(initial_returns)
        updates = report["iterations"]
        runs.append(
            {
                "file": path.name,
                "seed": report["config"]["seed"],
                "env_id": report["config"]["env_id"],
                "device": report["config"]["device"],
                "transitions": updates[-1]["transitions"],
                "optimizer_steps": sum(
                    row["update"]["optimizer_steps"] for row in updates
                ),
                "iteration_seconds_total": sum(row["seconds"] for row in updates),
                "test_episodes": len(pairs),
                "initial_mean_return": statistics.mean(initial_returns),
                "final_mean_return": statistics.mean(final_returns),
                "paired_mean_gain": gain,
                "paired_positive_episode_fraction": sum(
                    final > initial
                    for initial, final in zip(initial_returns, final_returns)
                )
                / len(pairs),
                "resume_regression": report.get("resume_regression"),
                "validation_curve": [
                    {
                        key: snapshot[key]
                        for key in (
                            "transitions",
                            "initial_mean_return",
                            "trained_mean_return",
                        )
                    }
                    for snapshot in report["evaluations"]
                    if snapshot["split"] == "validation"
                ],
            }
        )
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Duplicate training seeds")
    envs = {run["env_id"] for run in runs}
    if len(envs) != 1:
        raise ValueError("Do not aggregate different environments")
    gains = [run["paired_mean_gain"] for run in runs]
    final_means = [run["final_mean_return"] for run in runs]
    expected_seeds = {7, 17, 27}
    complete = {run["seed"] for run in runs} == expected_seeds
    goal = (
        (sum(value > 0 for value in gains) >= 2 and statistics.median(gains) >= 300)
        if envs == {"Pendulum-v1"}
        else None
    )
    return {
        "date": "2026-10-05",
        "environment": next(iter(envs)),
        "runs": runs,
        "training_seeds_complete": complete,
        "positive_gain_runs": sum(value > 0 for value in gains),
        "gain_mean_across_training_seeds": statistics.mean(gains),
        "gain_median_across_training_seeds": statistics.median(gains),
        "final_return_mean_across_training_seeds": statistics.mean(final_means),
        "final_return_sample_std_across_training_seeds": statistics.stdev(final_means)
        if len(final_means) > 1
        else None,
        "pendulum_engineering_target_met": (complete and goal)
        if goal is not None
        else None,
        "statistical_scope": "Training seed is the outer unit. Three runs do not establish statistical significance.",
        "integration_scope": "Ordinary-module Worker, shared rollout/update, CPU Ray transport; official FSDP/VLA unverified.",
    }


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    colors = ["#2563eb", "#e58022", "#059669"]
    for run, color in zip(summary["runs"], colors):
        curve = run["validation_curve"]
        axes[0].plot(
            [row["transitions"] for row in curve],
            [row["trained_mean_return"] for row in curve],
            marker="o",
            color=color,
            label=f"seed {run['seed']}",
        )
        axes[1].plot(
            [0, 1],
            [run["initial_mean_return"], run["final_mean_return"]],
            marker="o",
            color=color,
        )
    axes[0].set(
        xlabel="Training transitions",
        ylabel="Raw episode return (higher is better)",
        title="Validation: fixed held-out starts",
    )
    axes[0].legend(frameon=False)
    axes[0].ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Initial", "Trained"],
        ylabel="Mean raw return",
        title="Separate test: paired final comparison",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{summary['environment']} on S4000 / RLinf Worker / MUSA PPO", fontsize=13
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(summary, args.plot)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}))


if __name__ == "__main__":
    main()
