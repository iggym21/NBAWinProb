"""Per-WebSocket-connection incremental inference — one forward pass per event,
hidden state carried across the session (spec: no full-prefix replay per event)."""
import torch

from app.features import build_numeric_features, encode_event_type
from app.model import WinProbLSTM


class InferenceSession:
    def __init__(self, model: WinProbLSTM):
        self._model = model
        self._hidden = None

    def step(self, event: dict) -> float | None:
        feats = build_numeric_features(
            event["period"], event["clock"], event["home_score"],
            event["away_score"], event["possession_team"],
        )
        if feats is None:
            return None
        event_type_idx = torch.tensor([[encode_event_type(event["event_type"])]], dtype=torch.long)
        numeric_features = torch.tensor([[list(feats)]], dtype=torch.float32)
        with torch.no_grad():
            win_prob, self._hidden = self._model(event_type_idx, numeric_features, self._hidden)
        return float(win_prob.item())
