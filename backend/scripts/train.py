"""Trains the WinProbLSTM and a logistic-regression baseline on cached play-by-play data.

Usage:
    python scripts/train.py --db data/nba.db --epochs 15 --hidden-size 64 \
        --out-model model/win_prob_lstm.pt --out-baseline model/baseline_logreg.joblib \
        --out-split model/split.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection, get_game_events, list_games
from app.features import build_numeric_features, encode_event_type
from app.model import WinProbLSTM, masked_bce_loss, save_checkpoint


def split_games(game_ids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    ids = sorted(game_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def build_game_tensors(events: list[dict], home_win: int) -> dict | None:
    event_type_indices = []
    numeric_rows = []
    for e in events:
        feats = build_numeric_features(
            e["period"], e["clock"], e["home_score"], e["away_score"], e["possession_team"]
        )
        if feats is None:
            continue
        event_type_indices.append(encode_event_type(e["event_type"]))
        numeric_rows.append(feats)

    if not event_type_indices:
        return None

    return {
        "event_type_idx": torch.tensor(event_type_indices, dtype=torch.long),
        "numeric_features": torch.tensor(numeric_rows, dtype=torch.float32),
        "target": float(home_win),
    }


def _collate(games: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a list of per-game tensors to the batch's max length. Returns
    (event_type_idx, numeric_features, targets, mask)."""
    max_len = max(g["event_type_idx"].shape[0] for g in games)
    batch = len(games)
    event_type_idx = torch.zeros(batch, max_len, dtype=torch.long)
    numeric_features = torch.zeros(batch, max_len, 3, dtype=torch.float32)
    targets = torch.zeros(batch, max_len, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.float32)
    for i, g in enumerate(games):
        length = g["event_type_idx"].shape[0]
        event_type_idx[i, :length] = g["event_type_idx"]
        numeric_features[i, :length, :] = g["numeric_features"]
        targets[i, :length] = g["target"]
        mask[i, :length] = 1.0
    return event_type_idx, numeric_features, targets, mask


def train_model(db_path: str, epochs: int, hidden_size: int, lr: float, seed: int):
    torch.manual_seed(seed)
    conn = get_connection(db_path)
    games = list_games(conn)
    game_ids = [g["game_id"] for g in games]
    home_win_by_id = {g["game_id"]: g["home_win"] for g in games}
    train_ids, val_ids, test_ids = split_games(game_ids, seed=seed)

    def load_split(ids: list[str]) -> list[dict]:
        out = []
        for gid in ids:
            events = get_game_events(conn, gid)
            tensors = build_game_tensors(events, home_win_by_id[gid])
            if tensors is not None:
                out.append(tensors)
        return out

    train_games = load_split(train_ids)
    conn.close()

    from app.features import NUM_EVENT_TYPES

    lstm_config = {
        "num_event_types": NUM_EVENT_TYPES,
        "event_embed_dim": 8,
        "num_numeric_features": 3,
        "hidden_size": hidden_size,
        "num_layers": 1,
    }
    model = WinProbLSTM(**lstm_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train_games)
        event_type_idx, numeric_features, targets, mask = _collate(train_games)
        optimizer.zero_grad()
        win_prob, _ = model(event_type_idx, numeric_features)
        loss = masked_bce_loss(win_prob, targets, mask)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    model.eval()

    # Baseline: flatten every event of every training game into an independent row.
    baseline_X, baseline_y = [], []
    for g in train_games:
        n = g["event_type_idx"].shape[0]
        event_type_col = g["event_type_idx"].numpy().reshape(-1, 1).astype(float)
        numeric_cols = g["numeric_features"].numpy()
        baseline_X.append(np.hstack([numeric_cols, event_type_col]))
        baseline_y.append(np.full(n, g["target"]))
    baseline_X = np.vstack(baseline_X)
    baseline_y = np.concatenate(baseline_y)
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(baseline_X, baseline_y)

    split_info = {"train_game_ids": train_ids, "val_game_ids": val_ids, "test_game_ids": test_ids}
    return model, lstm_config, baseline_model, split_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/nba.db")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-model", default="model/win_prob_lstm.pt")
    parser.add_argument("--out-baseline", default="model/baseline_logreg.joblib")
    parser.add_argument("--out-split", default="model/split.json")
    args = parser.parse_args()

    model, config, baseline_model, split_info = train_model(
        args.db, args.epochs, args.hidden_size, args.lr, args.seed
    )

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, config, args.out_model)
    joblib.dump(baseline_model, args.out_baseline)
    Path(args.out_split).write_text(json.dumps(split_info, indent=2))
    print(f"Saved LSTM to {args.out_model}, baseline to {args.out_baseline}, split to {args.out_split}")


if __name__ == "__main__":
    main()
