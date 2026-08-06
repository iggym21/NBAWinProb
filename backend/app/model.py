"""PyTorch LSTM win-probability model."""
import torch
import torch.nn as nn


class WinProbLSTM(nn.Module):
    def __init__(
        self,
        num_event_types: int,
        event_embed_dim: int = 8,
        num_numeric_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_event_types = num_event_types
        self.event_embed_dim = event_embed_dim
        self.num_numeric_features = num_numeric_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.event_embedding = nn.Embedding(num_event_types, event_embed_dim)
        self.lstm = nn.LSTM(
            input_size=event_embed_dim + num_numeric_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, event_type_idx, numeric_features, hidden=None):
        embedded = self.event_embedding(event_type_idx)  # (B, T, embed_dim)
        lstm_input = torch.cat([embedded, numeric_features], dim=-1)  # (B, T, embed_dim+3)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)  # (B, T, hidden_size)
        logits = self.head(lstm_out).squeeze(-1)  # (B, T)
        win_prob = torch.sigmoid(logits)
        return win_prob, new_hidden

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


def masked_bce_loss(win_prob: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    win_prob = win_prob.clamp(eps, 1 - eps)
    per_step = -(targets * torch.log(win_prob) + (1 - targets) * torch.log(1 - win_prob))
    return (per_step * mask).sum() / mask.sum().clamp(min=1)


def save_checkpoint(model: WinProbLSTM, config: dict, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_model(checkpoint_path: str, device: str = "cpu") -> WinProbLSTM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WinProbLSTM(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
