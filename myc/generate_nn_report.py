#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "checkpoints-and-logs" / "local"
OUT_DIR = ROOT / "myc"
ASSETS_DIR = OUT_DIR / "assets"
REPORT_PATH = OUT_DIR / "nn_experiment_report.html"
SUMMARY_CSV_PATH = OUT_DIR / "nn_experiment_summary.csv"
SUMMARY_JSON_PATH = OUT_DIR / "nn_experiment_summary.json"
PLOT_ITER_LIMIT = 3000
EXCLUDED_FAMILIES = {"transformer_1d_old", "transformer_2d_old"}
COMMON_RUN_DEFAULTS = {
    "n_envs": 64,
    "n_steps": 32,
    "device": "cuda",
    "enable_monitors": True,
}

RUN_RE = re.compile(
    r"^(?P<algo>A2C|PPO) (?P<ts>\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})(?:_(?P<label>.+))?$"
)

FAMILY_ORDER = [
    "fc",
    "cnn1",
    "cnn2",
    "cnn3",
    "cnn4",
    "cnn5",
    "transformer_1d",
    "transformer_2d",
]
ALGO_ORDER = {"A2C": 0, "PPO": 1}

ALGO_COLORS = {"A2C": "#2b6cb0", "PPO": "#d1495b"}
FAMILY_COLORS = {
    "fc": "#355070",
    "cnn1": "#287271",
    "cnn2": "#3d7ea6",
    "cnn3": "#6a994e",
    "cnn4": "#c98b2f",
    "cnn5": "#bc4749",
    "transformer_1d": "#457b9d",
    "transformer_2d": "#2a9d8f",
}

CNN_INFERENCE = {
    "cnn1": "CNN-A: 5x5 + BatchNorm",
    "cnn2": "CNN-B: 3x3 + BatchNorm",
    "cnn3": "CNN-C: 5x5 + GroupNorm",
    "cnn4": "CNN-D: 3x3 + GroupNorm",
    "cnn5": "CNN-E: 3x3 + GroupNorm + PreAct residual + SE",
}

TRANSFORMER_FOCUS = {
    "transformer_1d": "1D sinusoidal positional encoding",
    "transformer_2d": "2D learnable row/col positional encoding",
}


@dataclass
class RunRecord:
    name: str
    algorithm: str
    timestamp: str
    label: str
    family: str
    category: str
    short_name: str
    focus: str
    eval_epochs: list[int]
    eval_rewards: list[float]
    eval_pegs: list[float]
    plot_eval_epochs: list[int]
    plot_eval_rewards: list[float]
    plot_eval_pegs: list[float]
    train_epochs: list[int]
    train_rewards: list[float]
    plot_train_epochs: list[int]
    plot_train_rewards: list[float]
    entropy_epochs: list[int]
    entropy_values: list[float]
    plot_entropy_epochs: list[int]
    plot_entropy_values: list[float]
    grad_epochs: list[int]
    grad_values: list[float]
    plot_grad_epochs: list[int]
    plot_grad_values: list[float]
    grad_source: str
    monitor_available: bool
    metrics: dict
    plot_metrics: dict
    note: str
    plot_note: str


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    if not records:
        raise SystemExit("No architecture-tagged experiment runs were found.")

    charts = write_charts(records)
    summary_rows = build_summary_rows(records)
    write_summary_csv(summary_rows)
    write_summary_json(records, charts)
    write_html_report(records, charts)


def load_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        raw_label = match.group("label") or ""
        if not any(token in raw_label for token in ("fc", "cnn", "transformer", "transfomer")):
            continue
        normalized = normalize_label(raw_label)
        if normalized in EXCLUDED_FAMILIES:
            continue
        record = build_record(run_dir, match.group("algo"), match.group("ts"), raw_label)
        if record is not None:
            records.append(record)
    records.sort(key=record_sort_key)
    return records


def build_record(run_dir: Path, algorithm: str, timestamp: str, raw_label: str) -> RunRecord | None:
    label = normalize_label(raw_label)
    family = label
    category = infer_category(family)
    short_name = build_short_name(algorithm, family)
    focus = focus_for_family(family)

    eval_rows = read_numeric_csv(run_dir / "logs" / "evaluation_metrics.csv")
    history_rows = read_numeric_csv(run_dir / "logs" / "training_history_full.csv")
    if not eval_rows or not history_rows:
        return None

    monitor_path = run_dir / "results" / "monitor_summary.json"
    monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
    monitor_available = bool(monitor)

    eval_epochs = [int(row["epoch"]) for row in eval_rows]
    eval_rewards = [float(row["reward"]) for row in eval_rows]
    eval_pegs = [float(row["pegs_left"]) for row in eval_rows]
    plot_eval_epochs, plot_eval_rewards, plot_eval_pegs = truncate_three_series(
        eval_epochs,
        eval_rewards,
        eval_pegs,
        PLOT_ITER_LIMIT,
    )

    train_epochs = [int(row["epoch"]) for row in history_rows]
    train_rewards = [float(row["mean_reward"]) for row in history_rows]
    plot_train_epochs, plot_train_rewards = truncate_pair_series(train_epochs, train_rewards, PLOT_ITER_LIMIT)
    entropy_values = [float(row["entropy"]) for row in history_rows]
    entropy_epochs = train_epochs.copy()
    plot_entropy_epochs, plot_entropy_values = truncate_pair_series(entropy_epochs, entropy_values, PLOT_ITER_LIMIT)

    grad_epochs, grad_values, grad_source = extract_grad_trace(history_rows, monitor, algorithm)
    plot_grad_epochs, plot_grad_values = truncate_pair_series(grad_epochs, grad_values, PLOT_ITER_LIMIT)

    wall_seconds_per_epoch = [float(row["collect_time"]) + float(row["train_time"]) for row in history_rows]
    cumulative_wall = np.cumsum(wall_seconds_per_epoch)
    plot_history_rows = [row for row in history_rows if int(row["epoch"]) <= PLOT_ITER_LIMIT]
    plot_grad_source = grad_source
    if not plot_grad_values:
        history_plot_grad_epochs = [int(row["epoch"]) for row in plot_history_rows]
        history_plot_grad_values = [float(row.get("grad_norm_mean", 0.0)) for row in plot_history_rows]
        if history_plot_grad_values and not (algorithm == "PPO" and all(abs(value) < 1e-12 for value in history_plot_grad_values)):
            plot_grad_epochs = history_plot_grad_epochs
            plot_grad_values = history_plot_grad_values
            plot_grad_source = "training_history_window"
        elif algorithm == "PPO":
            plot_grad_source = "missing"

    eval_metrics = summarize_eval_metrics(eval_epochs, eval_rewards, eval_pegs, cumulative_wall)
    plot_eval_metrics = summarize_eval_metrics(plot_eval_epochs, plot_eval_rewards, plot_eval_pegs, cumulative_wall)

    gradient_monitor = monitor.get("gradient_monitor", {})
    entropy_monitor = monitor.get("entropy_monitor", {})
    loss_monitor = monitor.get("loss_monitor", {})

    if grad_values:
        grad_final_fallback = grad_values[-1]
        grad_peak_fallback = max(grad_values)
    else:
        grad_final_fallback = None
        grad_peak_fallback = None

    grad_final = gradient_monitor.get("latest", grad_final_fallback)
    if gradient_monitor.get("history"):
        grad_peak = max(float(item.get("grad_norm_max", 0.0)) for item in gradient_monitor["history"])
    else:
        grad_peak = grad_peak_fallback

    if grad_final == 0.0 and not monitor_available and algorithm == "PPO":
        grad_final = None
    if grad_peak == 0.0 and not monitor_available and algorithm == "PPO":
        grad_peak = None

    plot_grad_final = plot_grad_values[-1] if plot_grad_values else None
    plot_grad_peak = max(plot_grad_values) if plot_grad_values else None
    if plot_grad_final == 0.0 and not monitor_available and algorithm == "PPO":
        plot_grad_final = None
    if plot_grad_peak == 0.0 and not monitor_available and algorithm == "PPO":
        plot_grad_peak = None

    note = build_note(
        best_eval_reward=eval_metrics["best_eval_reward"],
        final_eval_reward=eval_metrics["final_eval_reward"],
        min_pegs_left=eval_metrics["min_pegs_left"],
        first_one_peg_epoch=eval_metrics["first_one_peg_epoch"],
        final_entropy=entropy_values[-1],
        final_grad=grad_final,
    )
    plot_note = build_note(
        best_eval_reward=plot_eval_metrics["best_eval_reward"],
        final_eval_reward=plot_eval_metrics["final_eval_reward"],
        min_pegs_left=plot_eval_metrics["min_pegs_left"],
        first_one_peg_epoch=plot_eval_metrics["first_one_peg_epoch"],
        final_entropy=plot_entropy_values[-1] if plot_entropy_values else entropy_values[-1],
        final_grad=plot_grad_final,
    )

    metrics = {
        **eval_metrics,
        "epochs": len(history_rows),
        "wall_clock_minutes": float(sum(wall_seconds_per_epoch) / 60.0),
        "seconds_per_epoch": float(sum(wall_seconds_per_epoch) / len(wall_seconds_per_epoch)),
        "collect_seconds_per_epoch": float(sum(float(row["collect_time"]) for row in history_rows) / len(history_rows)),
        "train_seconds_per_epoch": float(sum(float(row["train_time"]) for row in history_rows) / len(history_rows)),
        "best_train_reward": max(train_rewards),
        "final_train_reward": train_rewards[-1],
        "final_total_loss": float(history_rows[-1]["total_loss"]),
        "final_actor_loss": float(history_rows[-1]["actor_loss"]),
        "final_critic_loss": float(history_rows[-1]["critic_loss"]),
        "mean_critic_loss": float(statistics.fmean(float(row["critic_loss"]) for row in history_rows)),
        "final_entropy": float(entropy_values[-1]),
        "mean_entropy": float(entropy_monitor.get("mean_entropy", statistics.fmean(entropy_values))),
        "final_grad": grad_final,
        "peak_grad": grad_peak,
        "gradient_source": grad_source,
        "loss_trend": loss_monitor.get("trend"),
    }
    plot_metrics = {
        **plot_eval_metrics,
        "epochs": len(plot_history_rows),
        "wall_clock_minutes": float(sum(float(row["collect_time"]) + float(row["train_time"]) for row in plot_history_rows) / 60.0)
        if plot_history_rows
        else 0.0,
        "seconds_per_epoch": float(
            sum(float(row["collect_time"]) + float(row["train_time"]) for row in plot_history_rows) / len(plot_history_rows)
        )
        if plot_history_rows
        else 0.0,
        "collect_seconds_per_epoch": float(sum(float(row["collect_time"]) for row in plot_history_rows) / len(plot_history_rows))
        if plot_history_rows
        else 0.0,
        "train_seconds_per_epoch": float(sum(float(row["train_time"]) for row in plot_history_rows) / len(plot_history_rows))
        if plot_history_rows
        else 0.0,
        "best_train_reward": max(plot_train_rewards) if plot_train_rewards else None,
        "final_train_reward": plot_train_rewards[-1] if plot_train_rewards else None,
        "final_total_loss": float(plot_history_rows[-1]["total_loss"]) if plot_history_rows else None,
        "final_actor_loss": float(plot_history_rows[-1]["actor_loss"]) if plot_history_rows else None,
        "final_critic_loss": float(plot_history_rows[-1]["critic_loss"]) if plot_history_rows else None,
        "mean_critic_loss": float(statistics.fmean(float(row["critic_loss"]) for row in plot_history_rows)) if plot_history_rows else None,
        "final_entropy": float(plot_entropy_values[-1]) if plot_entropy_values else None,
        "mean_entropy": float(statistics.fmean(plot_entropy_values)) if plot_entropy_values else None,
        "final_grad": plot_grad_final,
        "peak_grad": plot_grad_peak,
        "gradient_source": plot_grad_source,
        "loss_trend": loss_monitor.get("trend"),
    }

    return RunRecord(
        name=run_dir.name,
        algorithm=algorithm,
        timestamp=timestamp,
        label=label,
        family=family,
        category=category,
        short_name=short_name,
        focus=focus,
        eval_epochs=eval_epochs,
        eval_rewards=eval_rewards,
        eval_pegs=eval_pegs,
        plot_eval_epochs=plot_eval_epochs,
        plot_eval_rewards=plot_eval_rewards,
        plot_eval_pegs=plot_eval_pegs,
        train_epochs=train_epochs,
        train_rewards=train_rewards,
        plot_train_epochs=plot_train_epochs,
        plot_train_rewards=plot_train_rewards,
        entropy_epochs=entropy_epochs,
        entropy_values=entropy_values,
        plot_entropy_epochs=plot_entropy_epochs,
        plot_entropy_values=plot_entropy_values,
        grad_epochs=grad_epochs,
        grad_values=grad_values,
        plot_grad_epochs=plot_grad_epochs,
        plot_grad_values=plot_grad_values,
        grad_source=grad_source,
        monitor_available=monitor_available,
        metrics=metrics,
        plot_metrics=plot_metrics,
        note=note,
        plot_note=plot_note,
    )


def normalize_label(label: str) -> str:
    return label.replace("transfomer", "transformer")


def infer_category(family: str) -> str:
    if family == "fc":
        return "FC"
    if family.startswith("cnn"):
        return "CNN"
    return "Transformer"


def build_short_name(algorithm: str, family: str) -> str:
    if family == "fc":
        family_name = "FC"
    elif family.startswith("cnn"):
        family_name = family.upper()
    else:
        family_name = family.replace("transformer_", "Transformer ").replace("_old", " old").replace("_", " ").title()
        family_name = family_name.replace("1D", "1D").replace("2D", "2D")
    return f"{algorithm} · {family_name}"


def focus_for_family(family: str) -> str:
    if family == "fc":
        return "147-d flatten -> GELU MLP -> policy/value heads"
    if family.startswith("cnn"):
        return CNN_INFERENCE.get(family, family)
    return TRANSFORMER_FOCUS.get(family, family)


def extract_grad_trace(history_rows: list[dict], monitor: dict, algorithm: str) -> tuple[list[int], list[float], str]:
    gradient_monitor = monitor.get("gradient_monitor", {})
    history = gradient_monitor.get("history") or []
    if history:
        epochs = [int(item["epoch"]) for item in history]
        values = [float(item["grad_norm_mean"]) for item in history]
        return epochs, values, "monitor_summary"

    epochs = [int(row["epoch"]) for row in history_rows]
    values = [float(row.get("grad_norm_mean", 0.0)) for row in history_rows]
    if algorithm == "PPO" and all(abs(value) < 1e-12 for value in values):
        return [], [], "missing"
    return epochs, values, "training_history"


def truncate_pair_series(xs: list[int], ys: list[float], limit: int) -> tuple[list[int], list[float]]:
    kept = [(x, y) for x, y in zip(xs, ys) if x <= limit]
    if not kept:
        return [], []
    return [item[0] for item in kept], [item[1] for item in kept]


def truncate_three_series(
    xs: list[int],
    ys1: list[float],
    ys2: list[float],
    limit: int,
) -> tuple[list[int], list[float], list[float]]:
    kept = [(x, y1, y2) for x, y1, y2 in zip(xs, ys1, ys2) if x <= limit]
    if not kept:
        return [], [], []
    return (
        [item[0] for item in kept],
        [item[1] for item in kept],
        [item[2] for item in kept],
    )


def summarize_eval_metrics(
    eval_epochs: list[int],
    eval_rewards: list[float],
    eval_pegs: list[float],
    cumulative_wall: np.ndarray,
) -> dict:
    if not eval_epochs:
        return {
            "best_eval_reward": None,
            "final_eval_reward": None,
            "mean_eval_reward": None,
            "best_eval_epoch": None,
            "first_one_peg_epoch": None,
            "time_to_one_peg_minutes": None,
            "final_pegs_left": None,
            "min_pegs_left": None,
            "eval_regression_gap": None,
            "eval_last10_mean": None,
            "eval_last10_std": None,
        }

    best_eval_reward = max(eval_rewards)
    best_eval_index = eval_rewards.index(best_eval_reward)
    best_eval_epoch = eval_epochs[best_eval_index]
    final_eval_reward = eval_rewards[-1]
    final_pegs_left = eval_pegs[-1]
    min_pegs_left = min(eval_pegs)
    eval_gap = best_eval_reward - final_eval_reward
    first_one_peg_epoch = next((epoch for epoch, pegs in zip(eval_epochs, eval_pegs) if pegs <= 1.0), None)
    time_to_one_peg_minutes = (
        float(cumulative_wall[first_one_peg_epoch]) / 60.0
        if first_one_peg_epoch is not None and first_one_peg_epoch < len(cumulative_wall)
        else None
    )
    last_rewards = eval_rewards[-10:] if len(eval_rewards) >= 10 else eval_rewards

    return {
        "best_eval_reward": best_eval_reward,
        "final_eval_reward": final_eval_reward,
        "mean_eval_reward": float(sum(eval_rewards) / len(eval_rewards)),
        "best_eval_epoch": best_eval_epoch,
        "first_one_peg_epoch": first_one_peg_epoch,
        "time_to_one_peg_minutes": time_to_one_peg_minutes,
        "final_pegs_left": final_pegs_left,
        "min_pegs_left": min_pegs_left,
        "eval_regression_gap": eval_gap,
        "eval_last10_mean": float(sum(last_rewards) / len(last_rewards)),
        "eval_last10_std": float(statistics.pstdev(last_rewards)) if len(last_rewards) > 1 else 0.0,
    }


def parse_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        value = value.strip()
        if value == "":
            nested: dict = {}
            container[key] = nested
            stack.append((indent, nested))
        else:
            container[key] = parse_scalar(value)
    return root


def parse_scalar(value: str):
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_algorithm_configs() -> dict[str, dict]:
    return {
        "A2C": parse_simple_yaml(ROOT / "config" / "agent-trainer" / "actor-critic-trainer-config.yaml"),
        "PPO": parse_simple_yaml(ROOT / "config" / "agent-trainer" / "ppo-trainer-config.yaml"),
    }


def build_note(
    *,
    best_eval_reward: float,
    final_eval_reward: float,
    min_pegs_left: float,
    first_one_peg_epoch: int | None,
    final_entropy: float,
    final_grad: float | None,
) -> str:
    notes: list[str] = []
    gap = best_eval_reward - final_eval_reward
    if first_one_peg_epoch is not None:
        notes.append("1-peg solved")
        if gap <= 0.02:
            notes.append("held solution")
        else:
            notes.append("late regression")
    elif min_pegs_left <= 2.0:
        if gap <= 0.05:
            notes.append("stable at 2 pegs")
        else:
            notes.append("peaked at 2 pegs")
    elif best_eval_reward >= 0.9:
        notes.append("briefly improved then collapsed")
    else:
        notes.append("no clear breakthrough")

    if final_entropy < 0.05:
        notes.append("entropy collapse")
    if final_grad is not None and final_grad >= 50:
        notes.append("gradient spike")
    return "; ".join(dict.fromkeys(notes))


def build_summary_rows(records: list[RunRecord]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        rows.append(
            {
                "run": record.name,
                "algorithm": record.algorithm,
                "family": record.family,
                "category": record.category,
                "short_name": record.short_name,
                "focus": record.focus,
                "window_iter_limit": PLOT_ITER_LIMIT,
                "window_best_eval_reward": round_or_none(record.plot_metrics["best_eval_reward"], 6),
                "window_final_eval_reward": round_or_none(record.plot_metrics["final_eval_reward"], 6),
                "window_mean_eval_reward": round_or_none(record.plot_metrics["mean_eval_reward"], 6),
                "window_best_eval_epoch": record.plot_metrics["best_eval_epoch"],
                "window_first_one_peg_epoch": record.plot_metrics["first_one_peg_epoch"],
                "window_time_to_one_peg_minutes": round_or_none(record.plot_metrics["time_to_one_peg_minutes"], 3),
                "window_final_pegs_left": round_or_none(record.plot_metrics["final_pegs_left"], 3),
                "window_min_pegs_left": round_or_none(record.plot_metrics["min_pegs_left"], 3),
                "window_eval_regression_gap": round_or_none(record.plot_metrics["eval_regression_gap"], 6),
                "window_epochs": record.plot_metrics["epochs"],
                "window_wall_clock_minutes": round_or_none(record.plot_metrics["wall_clock_minutes"], 3),
                "window_seconds_per_epoch": round_or_none(record.plot_metrics["seconds_per_epoch"], 3),
                "window_final_critic_loss": round_or_none(record.plot_metrics["final_critic_loss"], 6),
                "window_mean_critic_loss": round_or_none(record.plot_metrics["mean_critic_loss"], 6),
                "window_final_entropy": round_or_none(record.plot_metrics["final_entropy"], 6),
                "window_mean_entropy": round_or_none(record.plot_metrics["mean_entropy"], 6),
                "window_final_grad": round_or_none(record.plot_metrics["final_grad"], 6),
                "window_note": record.plot_note,
                "best_eval_reward": round(record.metrics["best_eval_reward"], 6),
                "final_eval_reward": round(record.metrics["final_eval_reward"], 6),
                "mean_eval_reward": round(record.metrics["mean_eval_reward"], 6),
                "best_eval_epoch": record.metrics["best_eval_epoch"],
                "first_one_peg_epoch": record.metrics["first_one_peg_epoch"],
                "time_to_one_peg_minutes": round_or_none(record.metrics["time_to_one_peg_minutes"], 3),
                "final_pegs_left": round(record.metrics["final_pegs_left"], 3),
                "min_pegs_left": round(record.metrics["min_pegs_left"], 3),
                "eval_regression_gap": round(record.metrics["eval_regression_gap"], 6),
                "epochs": record.metrics["epochs"],
                "wall_clock_minutes": round(record.metrics["wall_clock_minutes"], 3),
                "seconds_per_epoch": round(record.metrics["seconds_per_epoch"], 3),
                "final_train_reward": round(record.metrics["final_train_reward"], 6),
                "best_train_reward": round(record.metrics["best_train_reward"], 6),
                "final_total_loss": round(record.metrics["final_total_loss"], 6),
                "final_actor_loss": round(record.metrics["final_actor_loss"], 6),
                "final_critic_loss": round(record.metrics["final_critic_loss"], 6),
                "mean_critic_loss": round(record.metrics["mean_critic_loss"], 6),
                "final_entropy": round(record.metrics["final_entropy"], 6),
                "mean_entropy": round(record.metrics["mean_entropy"], 6),
                "final_grad": round_or_none(record.metrics["final_grad"], 6),
                "peak_grad": round_or_none(record.metrics["peak_grad"], 6),
                "grad_source": record.metrics["gradient_source"],
                "note": record.note,
            }
        )
    return rows


def write_summary_csv(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(records: list[RunRecord], charts: dict[str, str]) -> None:
    payload = {
        "record_count": len(records),
        "plot_iter_limit": PLOT_ITER_LIMIT,
        "runs": [record_to_json(record) for record in records],
        "charts": charts,
        "excluded_runs": [
            "A2C 2026-04-18 21-56-01",
            "A2C 2026-04-20 05-56-15_transformer_1d_old",
            "A2C 2026-04-20 07-30-41_transformer_2d_old",
            "PPO 2026-04-19 14-14-29_transformer_1d_old",
            "PPO 2026-04-19 18-22-35_transfomer_2d_old",
        ],
        "excluded_families": sorted(EXCLUDED_FAMILIES),
        "notes": {
            "cnn_mapping_is_inferred": True,
            "cnn_mapping_detail": CNN_INFERENCE,
            "cross_family_hyperparameters_differ": True,
            "plot_window": f"All report charts use epochs <= {PLOT_ITER_LIMIT}",
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def record_to_json(record: RunRecord) -> dict:
    payload = asdict(record)
    return payload


def write_charts(records: list[RunRecord]) -> dict[str, str]:
    charts: dict[str, str] = {}
    family_axis_labels = ["FC", "C1", "C2", "C3", "C4", "C5", "T1", "T2"]

    charts["overall_final_eval"] = save_svg(
        "overall_final_eval.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["final_eval_reward"],
                    "color": ALGO_COLORS[record.algorithm],
                    "secondary": record.plot_metrics["best_eval_reward"],
                }
                for record in sorted(records, key=lambda item: item.plot_metrics["final_eval_reward"], reverse=True)
            ],
            title=f"Evaluation Reward by Run within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Solid bar = reward at the last evaluation <= {PLOT_ITER_LIMIT} iter; dot = best reward reached within the same window",
            value_label="Reward",
            domain=(0.0, 2.05),
        ),
    )

    unstable = sorted(records, key=lambda item: item.plot_metrics["eval_regression_gap"], reverse=True)[:10]
    charts["overall_regression_gap"] = save_svg(
        "overall_regression_gap.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["eval_regression_gap"],
                    "color": FAMILY_COLORS[record.family],
                }
                for record in unstable
            ],
            title=f"Regression Gap within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Difference between best and last evaluation reward, both restricted to the 0-{PLOT_ITER_LIMIT} iter window",
            value_label="Best - Final",
            domain=(0.0, max(record.plot_metrics["eval_regression_gap"] for record in unstable) * 1.08),
        ),
    )

    solved = [record for record in records if record.plot_metrics["first_one_peg_epoch"] is not None]
    charts["solved_efficiency"] = save_svg(
        "solved_efficiency.svg",
        horizontal_bar_chart(
            [
                {
                    "label": f"{record.short_name} ({fmt(record.plot_metrics['time_to_one_peg_minutes'], 1)} min)",
                    "value": record.plot_metrics["first_one_peg_epoch"],
                    "color": ALGO_COLORS[record.algorithm],
                }
                for record in sorted(solved, key=lambda item: item.plot_metrics["first_one_peg_epoch"])
            ],
            title=f"Epoch to First 1-Peg Evaluation within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Only runs that solved within the 0-{PLOT_ITER_LIMIT} iter window are shown; wall-clock minutes are appended in labels",
            value_label="Evaluation epoch",
        ),
    )

    charts["family_final_eval"] = save_svg(
        "family_final_eval.svg",
        grouped_bar_chart(
            categories=FAMILY_ORDER,
            labels=family_axis_labels,
            groups=[
                {
                    "name": "A2C",
                    "color": ALGO_COLORS["A2C"],
                    "values": [lookup_plot_metric(records, "A2C", family, "final_eval_reward") for family in FAMILY_ORDER],
                },
                {
                    "name": "PPO",
                    "color": ALGO_COLORS["PPO"],
                    "values": [lookup_plot_metric(records, "PPO", family, "final_eval_reward") for family in FAMILY_ORDER],
                },
            ],
            title=f"Family-Level Evaluation Reward within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Family labels use the directory names; values are the last evaluation <= {PLOT_ITER_LIMIT} iter",
            y_label="Final reward",
            y_domain=(0.0, 2.05),
        ),
    )

    charts["a2c_fc_eval"] = save_svg(
        "fc_eval_curves.svg",
        line_chart(
            [
                series_from_record(record, ALGO_COLORS[record.algorithm], dashed=(record.algorithm == "PPO"))
                for record in records
                if record.family == "fc"
            ],
            title=f"FC Evaluation Curves (0-{PLOT_ITER_LIMIT} Iter)",
            subtitle=f"Only the first {PLOT_ITER_LIMIT} iter are plotted, showing early-stage sample efficiency instead of the full training outcome",
            x_label="Evaluation epoch",
            y_label="Evaluation reward",
            y_domain=(0.0, 2.05),
        ),
    )

    charts["a2c_cnn_eval"] = save_svg(
        "a2c_cnn_eval_curves.svg",
        line_chart(
            [
                series_from_record(record, FAMILY_COLORS[record.family])
                for record in records
                if record.algorithm == "A2C" and record.category == "CNN"
            ],
            title=f"A2C CNN Evaluation Curves (0-{PLOT_ITER_LIMIT} Iter)",
            subtitle=f"In the first {PLOT_ITER_LIMIT} iter, the 3x3 variants remain more stable than the 5x5 variants",
            x_label="Evaluation epoch",
            y_label="Evaluation reward",
            y_domain=(0.0, 2.05),
        ),
    )

    charts["ppo_cnn_eval"] = save_svg(
        "ppo_cnn_eval_curves.svg",
        line_chart(
            [
                series_from_record(record, FAMILY_COLORS[record.family])
                for record in records
                if record.algorithm == "PPO" and record.category == "CNN"
            ],
            title=f"PPO CNN Evaluation Curves (0-{PLOT_ITER_LIMIT} Iter)",
            subtitle=f"Within the first {PLOT_ITER_LIMIT} iter, cnn1 and cnn3 are already the leading variants",
            x_label="Evaluation epoch",
            y_label="Evaluation reward",
            y_domain=(0.0, 2.05),
        ),
    )

    charts["a2c_transformer_eval"] = save_svg(
        "a2c_transformer_eval_curves.svg",
        line_chart(
            [
                series_from_record(record, FAMILY_COLORS[record.family])
                for record in records
                if record.algorithm == "A2C" and record.category == "Transformer"
            ],
            title=f"A2C Transformer Evaluation Curves (0-{PLOT_ITER_LIMIT} Iter)",
            subtitle=f"Only the retained 1D and 2D runs are shown; 2D remains the more stable choice within the first {PLOT_ITER_LIMIT} iter",
            x_label="Evaluation epoch",
            y_label="Evaluation reward",
            y_domain=(0.0, 1.05),
        ),
    )

    charts["ppo_transformer_eval"] = save_svg(
        "ppo_transformer_eval_curves.svg",
        line_chart(
            [
                series_from_record(record, FAMILY_COLORS[record.family])
                for record in records
                if record.algorithm == "PPO" and record.category == "Transformer"
            ],
            title=f"PPO Transformer Evaluation Curves (0-{PLOT_ITER_LIMIT} Iter)",
            subtitle=f"Only the retained 1D and 2D runs are shown; neither reaches the 1-peg regime by iter {PLOT_ITER_LIMIT}",
            x_label="Evaluation epoch",
            y_label="Evaluation reward",
            y_domain=(0.0, 2.05),
        ),
    )

    transformer_records = [record for record in records if record.category == "Transformer"]
    charts["transformer_gradients"] = save_svg(
        "transformer_gradient_final.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["final_grad"] or 0.0,
                    "color": ALGO_COLORS[record.algorithm],
                }
                for record in sorted(
                    transformer_records,
                    key=lambda item: (item.plot_metrics["final_grad"] or -1.0),
                    reverse=True,
                )
            ],
            title=f"Transformer Gradient Norm at Iter {PLOT_ITER_LIMIT}",
            subtitle=f"Gradient traces are truncated to the last available point within the 0-{PLOT_ITER_LIMIT} iter window",
            value_label="Gradient norm",
        ),
    )

    charts["runtime_per_epoch"] = save_svg(
        "runtime_per_epoch.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["seconds_per_epoch"],
                    "color": ALGO_COLORS[record.algorithm],
                }
                for record in sorted(records, key=lambda item: item.plot_metrics["seconds_per_epoch"], reverse=True)
            ],
            title=f"Average Wall-Clock Seconds per Epoch within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"The computation cost is averaged only over epochs 0-{PLOT_ITER_LIMIT}",
            value_label="Seconds / epoch",
        ),
    )

    charts["entropy_final"] = save_svg(
        "entropy_final.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["final_entropy"],
                    "color": ALGO_COLORS[record.algorithm],
                    "secondary": record.plot_metrics["mean_entropy"],
                }
                for record in sorted(records, key=lambda item: item.plot_metrics["final_entropy"])
            ],
            title=f"Final Policy Entropy by Run within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Solid bar = entropy at the last training point <= {PLOT_ITER_LIMIT}; dot = mean entropy within the same window",
            value_label="Entropy",
            domain=(0.0, max(record.plot_metrics["final_entropy"] for record in records) * 1.08),
        ),
    )

    charts["critic_loss_final"] = save_svg(
        "critic_loss_final.svg",
        horizontal_bar_chart(
            [
                {
                    "label": record.short_name,
                    "value": record.plot_metrics["final_critic_loss"],
                    "color": ALGO_COLORS[record.algorithm],
                }
                for record in sorted(records, key=lambda item: item.plot_metrics["final_critic_loss"], reverse=True)
            ],
            title=f"Final Critic Loss by Run within First {PLOT_ITER_LIMIT} Iter",
            subtitle=f"Critic loss is taken from the last training point <= {PLOT_ITER_LIMIT}; lower is not automatically better if evaluation reward collapses",
            value_label="Critic loss",
            domain=(0.0, max(record.plot_metrics["final_critic_loss"] for record in records) * 1.08),
        ),
    )

    charts["architecture_fc"] = save_svg("architecture_fc.svg", architecture_card_svg("FC", "#355070", [
        "Input board: 7x7x3",
        "Flatten -> 147",
        "Embedding MLP: 3 x 256 GELU",
        "Policy head: 3 layers -> 132 logits",
        "Value head: 2 layers -> scalar",
    ]))
    charts["architecture_cnn"] = save_svg("architecture_cnn.svg", architecture_card_svg("CNN", "#287271", [
        "Input board: 7x7x3",
        "Conv stem -> 32 channels",
        "Residual conv block(s)",
        "A-E ablations vary kernel / norm / residual style",
        "Flattened policy head -> 132, value head -> 1",
    ]))
    charts["architecture_transformer"] = save_svg("architecture_transformer.svg", architecture_card_svg("Transformer", "#2a9d8f", [
        "Input board -> 49 tokens x 3",
        "Linear embedding -> 128",
        "1D sinusoidal or 2D learnable positional encoding",
        "2-layer encoder backbone + separate heads",
        "Policy output 132, value output 1",
    ]))

    return charts


def save_svg(filename: str, content: str) -> str:
    path = ASSETS_DIR / filename
    path.write_text(content)
    return f"assets/{filename}"


def write_html_report(records: list[RunRecord], charts: dict[str, str]) -> None:
    algo_configs = load_algorithm_configs()
    counts = Counter(record.algorithm for record in records)
    solved = [record for record in records if record.plot_metrics["first_one_peg_epoch"] is not None]
    best_final = max(record.plot_metrics["final_eval_reward"] for record in records)
    best_final_runs = [record for record in records if math.isclose(record.plot_metrics["final_eval_reward"], best_final)]
    fastest_epoch = min(solved, key=lambda item: item.plot_metrics["first_one_peg_epoch"]) if solved else None
    fastest_time = min(solved, key=lambda item: item.plot_metrics["time_to_one_peg_minutes"]) if solved else None
    biggest_gap = max(records, key=lambda item: item.plot_metrics["eval_regression_gap"])
    lowest_entropy = sorted(records, key=lambda item: item.plot_metrics["final_entropy"])[:3]
    highest_entropy = sorted(records, key=lambda item: item.plot_metrics["final_entropy"], reverse=True)[:3]
    highest_critic = sorted(records, key=lambda item: item.plot_metrics["final_critic_loss"], reverse=True)[:3]
    lowest_critic = sorted(records, key=lambda item: item.plot_metrics["final_critic_loss"])[:3]

    a2c_fc = find_record(records, "A2C", "fc")
    ppo_fc = find_record(records, "PPO", "fc")
    a2c_transformer_1d = find_record(records, "A2C", "transformer_1d")
    a2c_transformer_2d = find_record(records, "A2C", "transformer_2d")
    ppo_transformer_1d = find_record(records, "PPO", "transformer_1d")
    ppo_transformer_2d = find_record(records, "PPO", "transformer_2d")
    fc_window_text = (
        f"{escape(a2c_fc.short_name)} 在前 {PLOT_ITER_LIMIT} iter 内已经于 "
        f"<strong>{a2c_fc.plot_metrics['first_one_peg_epoch']}</strong> epoch 首次进入 1-peg 区间，"
        f"窗口末评估奖励保持在 <strong>{fmt(a2c_fc.plot_metrics['final_eval_reward'], 3)}</strong>；"
        if a2c_fc.plot_metrics["first_one_peg_epoch"] is not None
        else f"在前 {PLOT_ITER_LIMIT} iter 窗口里，{escape(a2c_fc.short_name)} 的最终评估奖励为 "
        f"<strong>{fmt(a2c_fc.plot_metrics['final_eval_reward'], 3)}</strong>，尚未进入 1-peg 区间；"
    )

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NN Experiment Analysis Report</title>
  <style>
    :root {{
      --bg: #f6f3ed;
      --paper: #fffdf9;
      --ink: #1d2433;
      --muted: #5d6472;
      --line: #d8d1c5;
      --accent: #355070;
      --accent-2: #287271;
      --good: #2f855a;
      --warn: #c05621;
      --bad: #c53030;
      --shadow: 0 14px 36px rgba(29, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(40, 114, 113, 0.10), transparent 28%),
        radial-gradient(circle at top left, rgba(53, 80, 112, 0.10), transparent 32%),
        var(--bg);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      line-height: 1.6;
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 36px 28px 72px;
    }}
    header {{
      background: linear-gradient(135deg, rgba(53, 80, 112, 0.96), rgba(40, 114, 113, 0.90));
      color: #fff;
      padding: 28px 30px;
      border-radius: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.15;
      letter-spacing: 0.02em;
    }}
    header p {{
      margin: 0;
      max-width: 980px;
      color: rgba(255, 255, 255, 0.9);
    }}
    section {{
      background: var(--paper);
      border: 1px solid rgba(216, 209, 197, 0.72);
      border-radius: 22px;
      padding: 24px 24px 28px;
      margin-top: 22px;
      box-shadow: var(--shadow);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 26px;
      line-height: 1.2;
    }}
    h3 {{
      margin: 16px 0 10px;
      font-size: 20px;
    }}
    p {{
      margin: 8px 0 0;
    }}
    .lede {{
      color: var(--muted);
      max-width: 1040px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      border-radius: 18px;
      padding: 18px 18px 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,243,237,0.82));
      border: 1px solid rgba(216, 209, 197, 0.86);
    }}
    .card .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 6px;
    }}
    .card .metric {{
      font-size: 30px;
      line-height: 1.1;
      margin: 0 0 6px;
    }}
    .callout {{
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(53, 80, 112, 0.06);
      border-left: 5px solid var(--accent);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 18px;
      align-items: start;
      margin-top: 12px;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-top: 14px;
    }}
    .chart {{
      margin-top: 16px;
      border: 1px solid rgba(216, 209, 197, 0.86);
      border-radius: 18px;
      overflow: hidden;
      background: #fff;
    }}
    .chart img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .bullet-list {{
      margin: 10px 0 0;
      padding-left: 20px;
    }}
    .bullet-list li {{
      margin: 8px 0;
    }}
    .family-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .family-box {{
      padding: 16px;
      border-radius: 16px;
      background: rgba(246, 243, 237, 0.78);
      border: 1px solid rgba(216, 209, 197, 0.86);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid rgba(216, 209, 197, 0.86);
      padding: 10px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      position: sticky;
      top: 0;
      background: var(--paper);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(40, 114, 113, 0.12);
      color: #1f5a59;
      border: 1px solid rgba(40, 114, 113, 0.18);
    }}
    .mono {{
      font-family: "SFMono-Regular", ui-monospace, "Cascadia Code", "Source Code Pro", Menlo, monospace;
      font-size: 13px;
    }}
    .matrix td, .matrix th {{
      text-align: center;
    }}
    footer {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }}
    @media (max-width: 980px) {{
      .two-col, .grid-3 {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>FC / CNN / Transformer 训练日志数据分析报告</h1>
      <p>
        本报告分析了 <strong>{len(records)}</strong> 个带明确网络标签的实验目录，来源于
        <span class="mono">checkpoints-and-logs/local</span>。核心指标来自
        <span class="mono">training_history_full.csv</span>、
        <span class="mono">evaluation_metrics.csv</span> 和可用的
        <span class="mono">monitor_summary.json</span>。按你的新要求，所有图表与正文主结论统一只看
        <strong>前 {PLOT_ITER_LIMIT} iter</strong>，同时完全排除四个
        <span class="mono">transformer_1/2d_old</span> 旧批次实验。
      </p>
    </header>

    <section>
      <h2>一页结论</h2>
      <p class="lede">
        报告重点是前 {PLOT_ITER_LIMIT} iter 内的训练回报、评估回报、收敛稳定性和计算效率。跨家族对比时需要保留一个重要限定：
        FC、CNN、Transformer 的 YAML 并不只改了网络结构，同时也带有不同学习率、正则化和损失系数，
        所以下面的“跨家族结论”本质上是“架构 + 该家族配置组合”的结论；只有 CNN1-5 和 Transformer 1D/2D
        这两组消融内部对比更接近单变量。
      </p>
      <div class="cards">
        <div class="card">
          <div class="eyebrow">覆盖实验</div>
          <div class="metric">{len(records)}</div>
          <div>{counts['A2C']} 个 A2C，{counts['PPO']} 个 PPO</div>
        </div>
        <div class="card">
          <div class="eyebrow">1 Peg 成功数</div>
          <div class="metric">{len(solved)}</div>
          <div>{sum(1 for item in solved if item.algorithm == 'A2C')} 个 A2C，{sum(1 for item in solved if item.algorithm == 'PPO')} 个 PPO</div>
        </div>
        <div class="card">
          <div class="eyebrow">最佳最终评估</div>
          <div class="metric">{fmt(best_final, 3)}</div>
          <div>窗口口径：{", ".join(item.short_name for item in best_final_runs)}</div>
        </div>
        <div class="card">
          <div class="eyebrow">最快到 1 Peg</div>
          <div class="metric">{escape(fastest_epoch.short_name if fastest_epoch else '—')}</div>
          <div>按评估 epoch: {fastest_epoch.plot_metrics['first_one_peg_epoch'] if fastest_epoch else '—'}；按墙钟时间最快的是 {escape(fastest_time.short_name) if fastest_time else '—'}，{fmt(fastest_time.plot_metrics['time_to_one_peg_minutes'], 1) if fastest_time else '—'} 分钟</div>
        </div>
      </div>
      <div class="callout">
        <strong>CNN 变体说明：</strong>
        目录中的 <span class="mono">cnn1</span> 到 <span class="mono">cnn5</span> 没有保存显式 variant 键。
        本报告按照配置文件 <span class="mono">config/nn/conv-policy-value-ablations.yaml</span> 的 A→E 顺序进行推断映射，
        即 <span class="mono">cnn1≈A</span>、<span class="mono">cnn2≈B</span>、<span class="mono">cnn3≈C</span>、
        <span class="mono">cnn4≈D</span>、<span class="mono">cnn5≈E</span>。这条映射应视为高概率推断，而不是日志中直接记录的事实。
      </div>
      <ul class="bullet-list">
        <li>
          <strong>FC 家族：</strong>
          {fc_window_text}{escape(ppo_fc.short_name)} 也停在
          <strong>{fmt(ppo_fc.plot_metrics['final_eval_reward'], 3)}</strong>。
        </li>
        <li>
          <strong>PPO 的最强组集中在 CNN：</strong>
          {", ".join(item.short_name for item in solved if item.algorithm == 'PPO')} 都进入了 1-peg 区间，
          其中 {escape(fastest_epoch.short_name if fastest_epoch else '—')} 的评估样本效率最高。
        </li>
        <li>
          <strong>A2C 的 Transformer 明显更脆弱：</strong>
          {escape(a2c_transformer_1d.short_name)} 在 iter {PLOT_ITER_LIMIT} 时的梯度范数达到
          {fmt(a2c_transformer_1d.plot_metrics['final_grad'], 3)}，
          明显高于 {escape(a2c_transformer_2d.short_name)} 的
          {fmt(a2c_transformer_2d.plot_metrics['final_grad'], 3)}，
          同时熵更低，表现出更强的塌缩倾向。
        </li>
        <li>
          <strong>前 {PLOT_ITER_LIMIT} iter 内也能观察到明显回退：</strong>
          回退幅度最大的实验是 {escape(biggest_gap.short_name)}，
          窗口内最佳评估奖励与窗口末评估奖励相差 {fmt(biggest_gap.plot_metrics['eval_regression_gap'], 3)}。
        </li>
      </ul>
    </section>

    <section>
      <h2>实验范围与方法</h2>
      <div class="two-col">
        <div>
          <h3>纳入范围</h3>
          <ul class="bullet-list">
            <li>纳入所有目录名带 <span class="mono">_fc</span>、<span class="mono">_cnn*</span>、<span class="mono">_transformer*</span> 的实验。</li>
            <li>排除 1 个无架构后缀目录：<span class="mono">A2C 2026-04-18 21-56-01</span>。</li>
            <li>按你的要求额外排除 4 个旧版 Transformer 实验：<span class="mono">A2C/PPO + transformer_1d_old/transformer_2d_old</span>。</li>
            <li>训练过程使用 <span class="mono">training_history_full.csv</span>；评估曲线使用 <span class="mono">evaluation_metrics.csv</span>。</li>
            <li>梯度稳定性优先用 <span class="mono">monitor_summary.json</span>，缺失时退回到训练历史。</li>
            <li>所有图表、矩阵表和正文主结论统一只使用 <span class="mono">epoch &lt;= {PLOT_ITER_LIMIT}</span> 的数据窗口。</li>
          </ul>
        </div>
        <div>
          <h3>指标定义</h3>
          <ul class="bullet-list">
            <li><strong>Best / Final evaluation reward：</strong> 都限定在前 {PLOT_ITER_LIMIT} iter 内。</li>
            <li><strong>1-peg epoch：</strong> 第一次达到 <span class="mono">pegs_left &lt;= 1</span> 且 epoch 不超过 {PLOT_ITER_LIMIT}。</li>
            <li><strong>Regression gap：</strong> <span class="mono">best_eval - final_eval</span>，两者都按窗口口径计算。</li>
            <li><strong>Final critic loss：</strong> iter {PLOT_ITER_LIMIT} 前最后一个训练点的 <span class="mono">critic_loss</span>。</li>
            <li><strong>Final entropy：</strong> iter {PLOT_ITER_LIMIT} 前最后一个训练点的策略熵。</li>
            <li><strong>Final gradient：</strong> iter {PLOT_ITER_LIMIT} 前最后一个可用梯度监控值。</li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>A2C / PPO 实验配置参数</h2>
      <p class="lede">
        这里列的是当前仓库里训练入口 <span class="mono">runTorch.py</span> 的公共默认参数，以及
        <span class="mono">config/agent-trainer</span> 下的算法训练器配置。它们定义了本批实验的基础训练预算与 PPO 特有超参数。
      </p>
      <div class="two-col">
        <div>
          <h3>公共运行参数</h3>
          <ul class="bullet-list">
            <li><span class="mono">n_envs = {COMMON_RUN_DEFAULTS['n_envs']}</span></li>
            <li><span class="mono">n_steps = {COMMON_RUN_DEFAULTS['n_steps']}</span></li>
            <li><span class="mono">device = {COMMON_RUN_DEFAULTS['device']}</span>（训练入口默认值）</li>
            <li><span class="mono">enable_monitors = {str(COMMON_RUN_DEFAULTS['enable_monitors']).lower()}</span></li>
            <li>学习率默认来自对应网络 YAML 的 <span class="mono">optimizer.lr</span>，除非命令行显式覆盖。</li>
          </ul>
        </div>
        <div>
          <h3>A2C 训练器参数</h3>
          <ul class="bullet-list">
            <li><span class="mono">batch_size = {algo_configs['A2C'].get('batch_size')}</span></li>
            <li><span class="mono">n_iter = {algo_configs['A2C'].get('n_iter')}</span></li>
            <li><span class="mono">n_optim_steps = {algo_configs['A2C'].get('n_optim_steps')}</span></li>
            <li><span class="mono">n_games_train = {algo_configs['A2C'].get('n_games_train')}</span></li>
            <li><span class="mono">n_games_eval = {algo_configs['A2C'].get('n_games_eval')}</span></li>
            <li><span class="mono">n_steps_update = {algo_configs['A2C'].get('n_steps_update')}</span></li>
            <li><span class="mono">log_every = {algo_configs['A2C'].get('log_every')}</span></li>
            <li><span class="mono">seed = {algo_configs['A2C'].get('seed')}</span></li>
          </ul>
        </div>
      </div>
      <div class="two-col">
        <div>
          <h3>PPO 训练器参数</h3>
          <ul class="bullet-list">
            <li><span class="mono">batch_size = {algo_configs['PPO'].get('batch_size')}</span></li>
            <li><span class="mono">n_iter = {algo_configs['PPO'].get('n_iter')}</span></li>
            <li><span class="mono">n_optim_steps = {algo_configs['PPO'].get('n_optim_steps')}</span></li>
            <li><span class="mono">n_games_train = {algo_configs['PPO'].get('n_games_train')}</span></li>
            <li><span class="mono">n_games_eval = {algo_configs['PPO'].get('n_games_eval')}</span></li>
            <li><span class="mono">n_steps_update = {algo_configs['PPO'].get('n_steps_update')}</span></li>
            <li><span class="mono">log_every = {algo_configs['PPO'].get('log_every')}</span></li>
            <li><span class="mono">seed = {algo_configs['PPO'].get('seed')}</span></li>
          </ul>
        </div>
        <div>
          <h3>PPO 专有超参数</h3>
          <ul class="bullet-list">
            <li><span class="mono">clip_epsilon = {algo_configs['PPO'].get('ppo', {}).get('clip_epsilon')}</span></li>
            <li><span class="mono">value_loss_coef = {algo_configs['PPO'].get('ppo', {}).get('value_loss_coef')}</span></li>
            <li><span class="mono">entropy_coef = {algo_configs['PPO'].get('ppo', {}).get('entropy_coef')}</span></li>
            <li><span class="mono">max_grad_norm = {algo_configs['PPO'].get('ppo', {}).get('max_grad_norm')}</span></li>
            <li><span class="mono">use_gae = {str(algo_configs['PPO'].get('ppo', {}).get('use_gae')).lower()}</span></li>
            <li><span class="mono">gae_lambda = {algo_configs['PPO'].get('ppo', {}).get('gae_lambda')}</span></li>
            <li><span class="mono">discount = {algo_configs['PPO'].get('ppo', {}).get('discount')}</span></li>
            <li><span class="mono">normalize_advantages = {str(algo_configs['PPO'].get('ppo', {}).get('normalize_advantages')).lower()}</span></li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>网络结构侧写</h2>
      <p class="lede">
        这里按当前代码与配置文件总结三类网络的结构变化轴。FC 与 Transformer 主要是家族级差异，CNN 和 Transformer 位置编码内部则属于更标准的消融。
      </p>
      <div class="grid-3">
        <div class="chart"><img src="{charts['architecture_fc']}" alt="FC architecture" /></div>
        <div class="chart"><img src="{charts['architecture_cnn']}" alt="CNN architecture" /></div>
        <div class="chart"><img src="{charts['architecture_transformer']}" alt="Transformer architecture" /></div>
      </div>
      <div class="family-grid">
        <div class="family-box">
          <h3>FC</h3>
          <p>全局展平后直接送入多层 MLP。优点是实现简单、单步训练开销低；缺点是完全丢掉 2D 局部结构先验。</p>
        </div>
        <div class="family-box">
          <h3>CNN</h3>
          <p>共享卷积骨干加残差块，再接 policy / value head。cnn1-cnn5 这组实验本质上在测试核大小、归一化方式和残差增强模块。</p>
        </div>
        <div class="family-box">
          <h3>Transformer</h3>
          <p>将棋盘展平成 49 个 token，先做线性嵌入，再注入 1D 或 2D 位置编码。本报告仅保留新批次的 1D/2D 对比，不再展示旧批次结果。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>全局对比</h2>
      <div class="chart"><img src="{charts['overall_final_eval']}" alt="Final evaluation reward by run" /></div>
      <div class="chart"><img src="{charts['family_final_eval']}" alt="Final evaluation reward by family and algorithm" /></div>
      <div class="two-col">
        <div class="chart"><img src="{charts['solved_efficiency']}" alt="Epoch to first one-peg evaluation" /></div>
        <div class="chart"><img src="{charts['overall_regression_gap']}" alt="Regression gap" /></div>
      </div>
      <h3>读图要点</h3>
      <ul class="bullet-list">
        <li>所有全局图都按前 {PLOT_ITER_LIMIT} iter 重算，所以这里比较的是“早期效率”，不是完整训练终局。</li>
        <li>最终评估奖励并不是唯一标准；一些实验在前 {PLOT_ITER_LIMIT} iter 内已经出现回退，所以要同时看 regression gap。</li>
        <li>按窗口内评估 epoch 看，最快的 1-peg 方案来自 PPO-CNN；在保留实验里，Transformer 还没有在 iter {PLOT_ITER_LIMIT} 前进入 1-peg。</li>
      </ul>
      <div class="table-wrap">
        {render_matrix_table(records)}
      </div>
    </section>

    <section>
      <h2>FC 家族</h2>
      <div class="chart"><img src="{charts['a2c_fc_eval']}" alt="FC evaluation curves" /></div>
      <p>
        如果只看前 {PLOT_ITER_LIMIT} iter，FC 还没有把 A2C 和 PPO 拉开到“是否 1-peg solved”这么大的差距。
        {escape(a2c_fc.short_name)} 的窗口末评估奖励为
        <strong>{fmt(a2c_fc.plot_metrics['final_eval_reward'], 3)}</strong>，
        {escape(ppo_fc.short_name)} 为
        <strong>{fmt(ppo_fc.plot_metrics['final_eval_reward'], 3)}</strong>。
        也就是说，FC 的优势在 {PLOT_ITER_LIMIT} iter 窗口内已经能看见，但它仍然更像一条稳步爬升曲线，而不是最激进的早期解法。
      </p>
      <p>
        从早期曲线看，FC 更像是一个稳步爬升但启动不算快的基线。
      </p>
    </section>

    <section>
      <h2>CNN 消融</h2>
      <div class="two-col">
        <div class="chart"><img src="{charts['a2c_cnn_eval']}" alt="A2C CNN curves" /></div>
        <div class="chart"><img src="{charts['ppo_cnn_eval']}" alt="PPO CNN curves" /></div>
      </div>
      <h3>主要发现</h3>
      <ul class="bullet-list">
        <li>
          <strong>A2C-CNN：</strong> 如果采用上面的 A→E 推断映射，那么 3x3 系列
          (<span class="mono">cnn2/cnn4/cnn5</span>) 在前 {PLOT_ITER_LIMIT} iter 的稳定性明显好于 5x5 系列
          (<span class="mono">cnn1/cnn3</span>)。前者大多维持在 2 pegs 左右，后者更容易提前回退。
        </li>
        <li>
          <strong>PPO-CNN：</strong> <span class="mono">cnn1</span> 和 <span class="mono">cnn3</span> 都能在前 {PLOT_ITER_LIMIT} iter 内进入 1-peg 区间，
          说明 5x5 核在 PPO 下并不吃亏；但 <span class="mono">cnn5</span> 在加入 PreAct + SE 后反而从
          <span class="mono">{fmt(find_record(records, 'PPO', 'cnn5').plot_metrics['best_eval_reward'], 3)}</span>
          回退到 <span class="mono">{fmt(find_record(records, 'PPO', 'cnn5').plot_metrics['final_eval_reward'], 3)}</span>，
          这意味着“更复杂的残差增强”至少没有改善前 {PLOT_ITER_LIMIT} iter 的早期效率。
        </li>
        <li>
          <strong>更稳的结论：</strong> 这组 CNN 内部消融比跨家族对比更可信，因为它们共享同一个卷积家族框架，只改核大小、归一化和残差样式。
        </li>
      </ul>
    </section>

    <section>
      <h2>Transformer 位置编码与稳定性</h2>
      <div class="two-col">
        <div class="chart"><img src="{charts['a2c_transformer_eval']}" alt="A2C transformer curves" /></div>
        <div class="chart"><img src="{charts['ppo_transformer_eval']}" alt="PPO transformer curves" /></div>
      </div>
      <div class="two-col">
        <div>
          <p>
            在 A2C 下，Transformer 仍然是最容易出现训练不稳的一类。{escape(a2c_transformer_1d.short_name)} 在 iter {PLOT_ITER_LIMIT}
            时的梯度范数为 <strong>{fmt(a2c_transformer_1d.plot_metrics['final_grad'], 3)}</strong>，
            熵为 <strong>{fmt(a2c_transformer_1d.plot_metrics['final_entropy'], 3)}</strong>；
            而 {escape(a2c_transformer_2d.short_name)} 分别是
            <strong>{fmt(a2c_transformer_2d.plot_metrics['final_grad'], 3)}</strong> 和
            <strong>{fmt(a2c_transformer_2d.plot_metrics['final_entropy'], 3)}</strong>。
            这说明在保留实验里，2D 位置编码至少提升了前 {PLOT_ITER_LIMIT} iter 的稳定性。
          </p>
          <p>
            但这个优势并没有在 PPO 下转化成绝对性能突破。{escape(ppo_transformer_1d.short_name)} 和
            {escape(ppo_transformer_2d.short_name)} 在前 {PLOT_ITER_LIMIT} iter 内都没有达到 1 peg，
            且窗口末评估奖励分别只有 <strong>{fmt(ppo_transformer_1d.plot_metrics['final_eval_reward'], 3)}</strong> 和
            <strong>{fmt(ppo_transformer_2d.plot_metrics['final_eval_reward'], 3)}</strong>。
          </p>
        </div>
        <div class="chart"><img src="{charts['transformer_gradients']}" alt="Transformer gradients" /></div>
      </div>
      <p>
        由于旧版四个实验已被移除，这里只保留新批次 1D/2D 的同窗比较。
      </p>
    </section>

    <section>
      <h2>Critic Loss 与策略熵</h2>
      <p class="lede">
        这两个指标都取自前 {PLOT_ITER_LIMIT} iter 窗口内最后一个训练点。熵更接近“策略是否过早变得确定”，critic loss 更接近“价值函数当前拟合误差”；
        它们都能辅助解释稳定性，但都不能单独替代评估回报。
      </p>
      <div class="chart"><img src="{charts['entropy_final']}" alt="Final policy entropy by run" /></div>
      <div class="chart"><img src="{charts['critic_loss_final']}" alt="Final critic loss by run" /></div>
      <h3>读图要点</h3>
      <ul class="bullet-list">
        <li>
          <strong>最低熵的一组基本都对应塌缩：</strong>
          {escape(lowest_entropy[0].short_name)}、{escape(lowest_entropy[1].short_name)}、{escape(lowest_entropy[2].short_name)}
          的窗口末熵分别只有
          <strong>{fmt(lowest_entropy[0].plot_metrics['final_entropy'], 3)}</strong>、
          <strong>{fmt(lowest_entropy[1].plot_metrics['final_entropy'], 3)}</strong> 和
          <strong>{fmt(lowest_entropy[2].plot_metrics['final_entropy'], 3)}</strong>，
          其中前两个实验的窗口末评估奖励已经退到
          <strong>{fmt(lowest_entropy[0].plot_metrics['final_eval_reward'], 3)}</strong> 和
          <strong>{fmt(lowest_entropy[1].plot_metrics['final_eval_reward'], 3)}</strong>。
        </li>
        <li>
          <strong>熵也不是越高越好：</strong>
          {escape(highest_entropy[0].short_name)} 的窗口末熵最高，为
          <strong>{fmt(highest_entropy[0].plot_metrics['final_entropy'], 3)}</strong>，
          但窗口末评估奖励只有 <strong>{fmt(highest_entropy[0].plot_metrics['final_eval_reward'], 3)}</strong>。
          相比之下，真正跑到 1-peg 的 {escape(find_record(records, 'PPO', 'cnn1').short_name)} 和
          {escape(find_record(records, 'PPO', 'cnn3').short_name)} 分别停在
          <strong>{fmt(find_record(records, 'PPO', 'cnn1').plot_metrics['final_entropy'], 3)}</strong> 和
          <strong>{fmt(find_record(records, 'PPO', 'cnn3').plot_metrics['final_entropy'], 3)}</strong>，
          说明中等熵水平更像“既保留探索、又没有完全塌缩”的平衡点。
        </li>
        <li>
          <strong>critic loss 与最终表现不是单调关系：</strong>
          最高的窗口末 critic loss 出现在
          {escape(highest_critic[0].short_name)}，达到
          <strong>{fmt(highest_critic[0].plot_metrics['final_critic_loss'], 3)}</strong>，
          这和它的梯度尖峰、评估回退是一致的；但最低的三项里也包含
          {escape(lowest_critic[0].short_name)} 和 {escape(lowest_critic[1].short_name)}，
          它们的 critic loss 虽低，窗口末评估奖励却只有
          <strong>{fmt(lowest_critic[0].plot_metrics['final_eval_reward'], 3)}</strong> 和
          <strong>{fmt(lowest_critic[1].plot_metrics['final_eval_reward'], 3)}</strong>。
        </li>
        <li>
          <strong>更健康的组合通常出现在“中等 critic loss + 非零熵”：</strong>
          例如 {escape(a2c_fc.short_name)} 的窗口末 critic loss /
          entropy 为 <strong>{fmt(a2c_fc.plot_metrics['final_critic_loss'], 3)}</strong> /
          <strong>{fmt(a2c_fc.plot_metrics['final_entropy'], 3)}</strong>，
          而 {escape(find_record(records, 'PPO', 'cnn1').short_name)} 和
          {escape(find_record(records, 'PPO', 'cnn3').short_name)} 也都在非零熵下保持了零回退。
        </li>
      </ul>
    </section>

    <section>
      <h2>效率与成本</h2>
      <div class="chart"><img src="{charts['runtime_per_epoch']}" alt="Runtime per epoch" /></div>
      <p>
        A2C-FC 在前 {PLOT_ITER_LIMIT} iter 内平均每个 epoch 只需
        <strong>{fmt(a2c_fc.plot_metrics['seconds_per_epoch'], 3)}</strong> 秒；
        PPO-Transformer 1D 约为 <strong>{fmt(ppo_transformer_1d.plot_metrics['seconds_per_epoch'], 3)}</strong> 秒。
        这解释了为什么 PPO-Transformer 即使曲线不占优，仍会消耗明显更多的优化时间。
      </p>
    </section>

    <section>
      <h2>实验汇总（前 {PLOT_ITER_LIMIT} iter 口径）</h2>
      <div class="table-wrap">
        {render_summary_table(records)}
      </div>
    </section>

    <section>
      <h2>建议</h2>
      <ul class="bullet-list">
        <li>如果下一轮目标是“前 {PLOT_ITER_LIMIT} iter 内尽快拿到高评估表现”，优先保留 <span class="mono">PPO + CNN1/CNN3</span>。</li>
        <li>如果继续做 CNN 消融，建议先围绕核大小和归一化做更细粒度验证，再决定是否保留 PreAct + SE；当前证据并不支持它改善早期效率。</li>
        <li>如果继续做 Transformer，先把训练稳定性作为第一优先级，尤其是 A2C 下的梯度异常和熵塌缩。</li>
        <li>如果要做严格的“只改网络结构”结论，建议把 FC / CNN / Transformer 的学习率、熵系数和损失系数统一后再复现实验。</li>
      </ul>
    </section>

    <footer>
      产物已同时导出为 HTML、CSV 和 JSON。HTML 可直接本地打开；CSV/JSON 可继续用于二次分析。
    </footer>
  </div>
</body>
</html>
"""
    REPORT_PATH.write_text(report)


def render_matrix_table(records: list[RunRecord]) -> str:
    rows = []
    for family in FAMILY_ORDER:
        cells = []
        for algorithm in ("A2C", "PPO"):
            record = find_record(records, algorithm, family)
            if record is None:
                cells.append("<td>—</td>")
                continue
            value = record.plot_metrics["final_eval_reward"]
            pegs = record.plot_metrics["final_pegs_left"]
            color = reward_cell_color(value)
            text = f"{fmt(value, 3)}<br/><span class='mono'>pegs={fmt(pegs, 1)}</span>"
            cells.append(f"<td style='background:{color};'>{text}</td>")
        rows.append(f"<tr><th>{escape(family)}</th>{''.join(cells)}</tr>")
    return (
        "<table class='matrix'>"
        f"<thead><tr><th>Family</th><th>A2C @ iter≤{PLOT_ITER_LIMIT}</th><th>PPO @ iter≤{PLOT_ITER_LIMIT}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_summary_table(records: list[RunRecord]) -> str:
    rows = []
    critic_scale = max((record.plot_metrics["final_critic_loss"] or 0.0) for record in records) or 1.0
    for record in records:
        reward_color = reward_cell_color(record.plot_metrics["final_eval_reward"])
        gap_color = gap_cell_color(record.plot_metrics["eval_regression_gap"])
        entropy_color = entropy_cell_color(record.plot_metrics["final_entropy"] or 0.0)
        critic_color = critic_loss_cell_color(record.plot_metrics["final_critic_loss"] or 0.0, critic_scale)
        grad_display = "—" if record.plot_metrics["final_grad"] is None else fmt(record.plot_metrics["final_grad"], 3)
        rows.append(
            "<tr>"
            f"<td><strong>{escape(record.short_name)}</strong><br/><span class='mono'>{escape(record.name)}</span></td>"
            f"<td>{escape(record.focus)}</td>"
            f"<td style='background:{reward_color};'><strong>{fmt(record.plot_metrics['final_eval_reward'], 3)}</strong><br/>best {fmt(record.plot_metrics['best_eval_reward'], 3)}</td>"
            f"<td>{fmt(record.plot_metrics['final_pegs_left'], 1)}<br/>best {fmt(record.plot_metrics['min_pegs_left'], 1)}</td>"
            f"<td>{record.plot_metrics['best_eval_epoch']:,}</td>"
            f"<td>{record.plot_metrics['first_one_peg_epoch'] if record.plot_metrics['first_one_peg_epoch'] is not None else '—'}</td>"
            f"<td>{fmt(record.plot_metrics['wall_clock_minutes'], 1)}<br/>{fmt(record.plot_metrics['seconds_per_epoch'], 3)} s/epoch</td>"
            f"<td style='background:{gap_color};'>{fmt(record.plot_metrics['eval_regression_gap'], 3)}</td>"
            f"<td style='background:{entropy_color};'>{fmt(record.plot_metrics['final_entropy'], 3)}</td>"
            f"<td style='background:{critic_color};'>{fmt(record.plot_metrics['final_critic_loss'], 3)}</td>"
            f"<td>{grad_display}</td>"
            f"<td>{escape(record.plot_note)}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        f"<th>Run</th><th>Focus</th><th>Eval reward @ iter≤{PLOT_ITER_LIMIT}</th><th>Pegs left</th>"
        "<th>Best epoch</th><th>1-peg epoch</th><th>Cost</th><th>Regression</th>"
        "<th>Entropy</th><th>Critic</th><th>Grad</th><th>Reading</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def read_numeric_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def record_sort_key(record: RunRecord) -> tuple[int, int, str]:
    family_idx = FAMILY_ORDER.index(record.family) if record.family in FAMILY_ORDER else len(FAMILY_ORDER)
    return ALGO_ORDER.get(record.algorithm, 99), family_idx, record.timestamp


def find_record(records: list[RunRecord], algorithm: str, family: str) -> RunRecord | None:
    for record in records:
        if record.algorithm == algorithm and record.family == family:
            return record
    return None


def lookup_metric(records: list[RunRecord], algorithm: str, family: str, key: str) -> float | None:
    record = find_record(records, algorithm, family)
    return None if record is None else record.metrics[key]


def lookup_plot_metric(records: list[RunRecord], algorithm: str, family: str, key: str) -> float | None:
    record = find_record(records, algorithm, family)
    return None if record is None else record.plot_metrics[key]


def round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def ellipsize(text: str, max_len: int = 36) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def svg_style() -> str:
    return (
        "<style>"
        "text{font-family:Georgia,'Times New Roman',serif;fill:#1d2433}"
        ".muted{fill:#6b7280;font-size:12px}"
        ".title{font-size:24px;font-weight:700}"
        ".subtitle{font-size:13px;fill:#5d6472}"
        ".axis{stroke:#6b7280;stroke-width:1}"
        ".grid{stroke:#e7e0d4;stroke-width:1}"
        ".legend{font-size:12px}"
        "</style>"
    )


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if math.isclose(domain_min, domain_max):
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def nice_domain(values: Iterable[float], pad: float = 0.06) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    span = hi - lo
    return lo - span * pad, hi + span * pad


def ticks(domain: tuple[float, float], count: int = 5) -> list[float]:
    lo, hi = domain
    if math.isclose(lo, hi):
        return [lo]
    return [lo + (hi - lo) * idx / (count - 1) for idx in range(count)]


def horizontal_bar_chart(
    items: list[dict],
    *,
    title: str,
    subtitle: str,
    value_label: str,
    domain: tuple[float, float] | None = None,
) -> str:
    count = max(1, len(items))
    width = 1120
    height = 96 + count * 34 + 40
    left = 330
    right = width - 70
    top = 70
    bottom = height - 38

    values = [float(item["value"]) for item in items]
    lo, hi = domain if domain is not None else nice_domain(values, pad=0.08)
    if math.isclose(lo, hi):
        hi = lo + 1.0

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        svg_style(),
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fff' />",
        f"<text class='title' x='26' y='34'>{escape(title)}</text>",
        f"<text class='subtitle' x='26' y='56'>{escape(subtitle)}</text>",
    ]

    for tick in ticks((lo, hi), 6):
        x = scale(tick, lo, hi, left, right)
        parts.append(f"<line class='grid' x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{bottom}' />")
        parts.append(f"<text class='muted' x='{x:.2f}' y='{bottom + 18}' text-anchor='middle'>{fmt(tick, 2)}</text>")

    parts.append(f"<text class='muted' x='{(left + right) / 2:.2f}' y='{height - 10}' text-anchor='middle'>{escape(value_label)}</text>")

    for idx, item in enumerate(items):
        y = top + idx * 34
        label = ellipsize(item["label"], 48)
        value = float(item["value"])
        bar_width = max(0.0, scale(value, lo, hi, left, right) - left)
        parts.append(f"<text x='{left - 12}' y='{y + 20}' text-anchor='end'>{escape(label)}</text>")
        parts.append(f"<rect x='{left}' y='{y + 6}' width='{bar_width:.2f}' height='18' rx='8' fill='{item['color']}' opacity='0.88' />")
        parts.append(f"<text x='{left + bar_width + 8:.2f}' y='{y + 20}'>{fmt(value, 3)}</text>")
        secondary = item.get("secondary")
        if secondary is not None:
            dot_x = scale(float(secondary), lo, hi, left, right)
            parts.append(f"<circle cx='{dot_x:.2f}' cy='{y + 15}' r='4.5' fill='#1d2433' />")

    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(
    *,
    categories: list[str],
    labels: list[str] | None,
    groups: list[dict],
    title: str,
    subtitle: str,
    y_label: str,
    y_domain: tuple[float, float],
) -> str:
    width = 1120
    height = 470
    left = 78
    right = width - 30
    top = 82
    bottom = height - 76
    plot_width = right - left
    plot_height = bottom - top
    group_count = len(groups)
    step = plot_width / max(1, len(categories))
    inner = step * 0.7
    bar_width = inner / max(1, group_count)

    y_lo, y_hi = y_domain
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        svg_style(),
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fff' />",
        f"<text class='title' x='26' y='34'>{escape(title)}</text>",
        f"<text class='subtitle' x='26' y='56'>{escape(subtitle)}</text>",
    ]
    for tick in ticks(y_domain, 5):
        y = scale(tick, y_lo, y_hi, bottom, top)
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.2f}' x2='{right}' y2='{y:.2f}' />")
        parts.append(f"<text class='muted' x='{left - 10}' y='{y + 4:.2f}' text-anchor='end'>{fmt(tick, 2)}</text>")
    parts.append(f"<text class='muted' x='18' y='{(top + bottom) / 2:.2f}' transform='rotate(-90, 18, {(top + bottom) / 2:.2f})' text-anchor='middle'>{escape(y_label)}</text>")

    legend_x = right - 180
    for idx, group in enumerate(groups):
        ly = 26 + idx * 18
        parts.append(f"<rect x='{legend_x}' y='{ly - 8}' width='12' height='12' rx='3' fill='{group['color']}' />")
        parts.append(f"<text class='legend' x='{legend_x + 18}' y='{ly + 2}'>{escape(group['name'])}</text>")

    axis_labels = labels or categories
    for cat_idx, category in enumerate(categories):
        base_x = left + cat_idx * step + (step - inner) / 2
        parts.append(f"<text class='muted' x='{base_x + inner / 2:.2f}' y='{bottom + 18}' text-anchor='middle'>{escape(axis_labels[cat_idx])}</text>")
        for group_idx, group in enumerate(groups):
            value = group["values"][cat_idx]
            if value is None:
                continue
            x = base_x + group_idx * bar_width
            y = scale(float(value), y_lo, y_hi, bottom, top)
            h = bottom - y
            parts.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width * 0.82:.2f}' height='{h:.2f}' rx='6' fill='{group['color']}' opacity='0.9' />")
            parts.append(f"<text class='muted' x='{x + bar_width * 0.41:.2f}' y='{y - 6:.2f}' text-anchor='middle'>{fmt(value, 2)}</text>")

    parts.append("</svg>")
    return "".join(parts)


def downsample_points(xs: list[int], ys: list[float], max_points: int = 520) -> tuple[list[int], list[float]]:
    if len(xs) <= max_points:
        return xs, ys
    indices = sorted(set(np.linspace(0, len(xs) - 1, num=max_points, dtype=int).tolist()))
    return [xs[index] for index in indices], [ys[index] for index in indices]


def series_from_record(record: RunRecord, color: str, dashed: bool = False) -> dict:
    xs, ys = downsample_points(record.plot_eval_epochs, record.plot_eval_rewards)
    return {
        "name": record.short_name,
        "x": xs,
        "y": ys,
        "color": color,
        "dasharray": "8 6" if dashed else None,
    }


def line_chart(
    series: list[dict],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    y_domain: tuple[float, float] | None = None,
) -> str:
    width = 1120
    height = 460
    left = 78
    right = width - 26
    top = 86
    bottom = height - 56

    all_x = [value for item in series for value in item["x"]]
    all_y = [value for item in series for value in item["y"]]
    if not all_x:
        all_x = [0, 1]
        all_y = [0.0, 1.0]
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo, y_hi = y_domain if y_domain is not None else nice_domain(all_y, pad=0.08)
    if math.isclose(x_lo, x_hi):
        x_hi = x_lo + 1
    if math.isclose(y_lo, y_hi):
        y_hi = y_lo + 1

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        svg_style(),
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fff' />",
        f"<text class='title' x='26' y='34'>{escape(title)}</text>",
        f"<text class='subtitle' x='26' y='56'>{escape(subtitle)}</text>",
    ]

    for tick in ticks((y_lo, y_hi), 5):
        y = scale(tick, y_lo, y_hi, bottom, top)
        parts.append(f"<line class='grid' x1='{left}' y1='{y:.2f}' x2='{right}' y2='{y:.2f}' />")
        parts.append(f"<text class='muted' x='{left - 10}' y='{y + 4:.2f}' text-anchor='end'>{fmt(tick, 2)}</text>")

    for tick in ticks((x_lo, x_hi), 6):
        x = scale(tick, x_lo, x_hi, left, right)
        parts.append(f"<line class='grid' x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{bottom}' />")
        parts.append(f"<text class='muted' x='{x:.2f}' y='{bottom + 18}' text-anchor='middle'>{int(round(tick)):,}</text>")

    parts.append(f"<text class='muted' x='{(left + right) / 2:.2f}' y='{height - 12}' text-anchor='middle'>{escape(x_label)}</text>")
    parts.append(f"<text class='muted' x='18' y='{(top + bottom) / 2:.2f}' transform='rotate(-90, 18, {(top + bottom) / 2:.2f})' text-anchor='middle'>{escape(y_label)}</text>")

    legend_x = right - 255
    legend_y = 26
    for idx, item in enumerate(series):
        row = idx // 2
        col = idx % 2
        x = legend_x + col * 120
        y = legend_y + row * 18
        dash = f" stroke-dasharray='{item['dasharray']}'" if item.get("dasharray") else ""
        parts.append(f"<line x1='{x}' y1='{y}' x2='{x + 18}' y2='{y}' stroke='{item['color']}' stroke-width='3'{dash} />")
        parts.append(f"<text class='legend' x='{x + 24}' y='{y + 4}'>{escape(ellipsize(item['name'], 18))}</text>")

    for item in series:
        points = " ".join(
            f"{scale(x, x_lo, x_hi, left, right):.2f},{scale(y, y_lo, y_hi, bottom, top):.2f}"
            for x, y in zip(item["x"], item["y"])
        )
        dash = f" stroke-dasharray='{item['dasharray']}'" if item.get("dasharray") else ""
        parts.append(
            f"<polyline fill='none' stroke='{item['color']}' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'{dash} points='{points}' />"
        )

    parts.append("</svg>")
    return "".join(parts)


def architecture_card_svg(title: str, color: str, lines: list[str]) -> str:
    width = 360
    height = 220
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        svg_style(),
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='22' fill='#fff' />",
        f"<rect x='22' y='18' width='{width - 44}' height='42' rx='14' fill='{color}' opacity='0.94' />",
        f"<text x='36' y='45' font-size='24' fill='#fff' font-weight='700'>{escape(title)}</text>",
    ]
    y = 82
    for index, line in enumerate(lines):
        parts.append(f"<rect x='24' y='{y - 18}' width='{width - 48}' height='28' rx='12' fill='{color}' opacity='{0.12 + index * 0.03:.2f}' />")
        parts.append(f"<text x='36' y='{y}'>{escape(line)}</text>")
        if index < len(lines) - 1:
            parts.append(f"<path d='M180 {y + 12} L180 {y + 26}' stroke='{color}' stroke-width='2.2' />")
            parts.append(f"<path d='M175 {y + 22} L180 {y + 28} L185 {y + 22}' fill='none' stroke='{color}' stroke-width='2.2' />")
        y += 34
    parts.append("</svg>")
    return "".join(parts)


def reward_cell_color(value: float) -> str:
    return blend("#f7efe6", "#4caf50", min(max(value / 2.0, 0.0), 1.0), alpha=0.34)


def gap_cell_color(value: float) -> str:
    return blend("#f7efe6", "#d1495b", min(max(value / 1.0, 0.0), 1.0), alpha=0.32)


def entropy_cell_color(value: float) -> str:
    ratio = 1.0 - min(max(value / 1.0, 0.0), 1.0)
    return blend("#f7efe6", "#c05621", ratio, alpha=0.28)


def critic_loss_cell_color(value: float, scale_ref: float) -> str:
    ratio = min(max(value / max(scale_ref, 1e-9), 0.0), 1.0)
    return blend("#f7efe6", "#b83280", ratio, alpha=0.28)


def blend(base_hex: str, accent_hex: str, ratio: float, alpha: float = 1.0) -> str:
    base = hex_to_rgb(base_hex)
    accent = hex_to_rgb(accent_hex)
    mixed = tuple(int(base[idx] + (accent[idx] - base[idx]) * ratio) for idx in range(3))
    return f"rgba({mixed[0]}, {mixed[1]}, {mixed[2]}, {alpha})"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


if __name__ == "__main__":
    main()
