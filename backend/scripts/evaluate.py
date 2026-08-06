"""Evaluation report: Brier score, log-loss, calibration, accuracy by time bucket.

Usage:
    python scripts/evaluate.py --db data/nba.db --model model/win_prob_lstm.pt \
        --baseline model/baseline_logreg.joblib --split model/split.json \
        --out-md reports/evaluation_report.md --out-json reports/evaluation_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection, get_game_events, list_games
from app.model import load_model
from scripts.train import build_game_tensors

TIME_BUCKETS: list[tuple[str, float, float]] = [
    (">36 min", 36 * 60, float("inf")),
    ("12-36 min", 12 * 60, 36 * 60),
    ("3-12 min", 3 * 60, 12 * 60),
    ("<3 min", 0, 3 * 60),
]


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def log_loss_score(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))


def calibration_curve(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        out.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "predicted_mean": float(probs[in_bin].mean()),
            "observed_frequency": float(targets[in_bin].mean()),
            "count": count,
        })
    return out


def accuracy_by_time_bucket(probs: np.ndarray, targets: np.ndarray, seconds_remaining: np.ndarray) -> dict:
    result = {}
    for name, lo, hi in TIME_BUCKETS:
        in_bucket = (seconds_remaining >= lo) & (seconds_remaining < hi)
        count = int(in_bucket.sum())
        if count == 0:
            result[name] = {"accuracy": None, "count": 0}
            continue
        correct = ((probs[in_bucket] > 0.5).astype(float) == targets[in_bucket]).mean()
        result[name] = {"accuracy": float(correct), "count": count}
    return result


def _evaluate_lstm(model, test_games: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs, all_targets, all_seconds = [], [], []
    with torch.no_grad():
        for g in test_games:
            event_type_idx = g["event_type_idx"].unsqueeze(0)
            numeric_features = g["numeric_features"].unsqueeze(0)
            probs, _ = model(event_type_idx, numeric_features)
            probs = probs.squeeze(0).numpy()
            n = len(probs)
            all_probs.append(probs)
            all_targets.append(np.full(n, g["target"]))
            # numeric_features[..., 1] is seconds_remaining_scaled; undo the scale for bucketing
            from app.features import REGULATION_TOTAL_SECONDS
            all_seconds.append(g["numeric_features"][:, 1].numpy() * REGULATION_TOTAL_SECONDS)
    return np.concatenate(all_probs), np.concatenate(all_targets), np.concatenate(all_seconds)


def _evaluate_baseline(baseline_model, test_games: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_targets = [], []
    for g in test_games:
        n = g["event_type_idx"].shape[0]
        event_type_col = g["event_type_idx"].numpy().reshape(-1, 1).astype(float)
        numeric_cols = g["numeric_features"].numpy()
        X = np.hstack([numeric_cols, event_type_col])
        probs = baseline_model.predict_proba(X)[:, 1]
        all_probs.append(probs)
        all_targets.append(np.full(n, g["target"]))
    return np.concatenate(all_probs), np.concatenate(all_targets)


def generate_report(db_path: str, model_path: str, baseline_path: str, split_path: str) -> dict:
    conn = get_connection(db_path)
    games = {g["game_id"]: g for g in list_games(conn)}
    split_info = json.loads(Path(split_path).read_text())
    test_ids = split_info["test_game_ids"]

    test_games = []
    for gid in test_ids:
        events = get_game_events(conn, gid)
        tensors = build_game_tensors(events, games[gid]["home_win"])
        if tensors is not None:
            test_games.append(tensors)
    conn.close()

    lstm_model = load_model(model_path)
    baseline_model = joblib.load(baseline_path)

    lstm_probs, lstm_targets, seconds_remaining = _evaluate_lstm(lstm_model, test_games)
    baseline_probs, baseline_targets = _evaluate_baseline(baseline_model, test_games)

    report = {
        "num_test_games": len(test_games),
        "lstm": {
            "brier_score": brier_score(lstm_probs, lstm_targets),
            "log_loss": log_loss_score(lstm_probs, lstm_targets),
            "calibration": calibration_curve(lstm_probs, lstm_targets),
            "accuracy_by_time_bucket": accuracy_by_time_bucket(lstm_probs, lstm_targets, seconds_remaining),
        },
        "baseline_logreg": {
            "brier_score": brier_score(baseline_probs, baseline_targets),
            "log_loss": log_loss_score(baseline_probs, baseline_targets),
            "calibration": calibration_curve(baseline_probs, baseline_targets),
            "accuracy_by_time_bucket": accuracy_by_time_bucket(baseline_probs, baseline_targets, seconds_remaining),
        },
    }
    return report


def _report_to_markdown(report: dict) -> str:
    lines = ["# Evaluation Report", "", f"Test games: {report['num_test_games']}", ""]
    lines.append("| Metric | LSTM | Logistic Regression baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| Brier score (lower better) | {report['lstm']['brier_score']:.4f} | {report['baseline_logreg']['brier_score']:.4f} |")
    lines.append(f"| Log-loss (lower better) | {report['lstm']['log_loss']:.4f} | {report['baseline_logreg']['log_loss']:.4f} |")
    lines.append("")
    lines.append("## Accuracy by time remaining")
    lines.append("| Time remaining | LSTM accuracy | Baseline accuracy | N |")
    lines.append("|---|---|---|---|")
    for name, _, _ in TIME_BUCKETS:
        l = report["lstm"]["accuracy_by_time_bucket"][name]
        b = report["baseline_logreg"]["accuracy_by_time_bucket"][name]
        l_acc = f"{l['accuracy']:.3f}" if l["accuracy"] is not None else "n/a"
        b_acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "n/a"
        lines.append(f"| {name} | {l_acc} | {b_acc} | {l['count']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/nba.db")
    parser.add_argument("--model", default="model/win_prob_lstm.pt")
    parser.add_argument("--baseline", default="model/baseline_logreg.joblib")
    parser.add_argument("--split", default="model/split.json")
    parser.add_argument("--out-md", default="reports/evaluation_report.md")
    parser.add_argument("--out-json", default="reports/evaluation_report.json")
    args = parser.parse_args()

    report = generate_report(args.db, args.model, args.baseline, args.split)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    Path(args.out_md).write_text(_report_to_markdown(report))
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
