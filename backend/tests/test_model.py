import torch
import pytest
from app.model import WinProbLSTM, masked_bce_loss, save_checkpoint, load_model
from app.features import NUM_EVENT_TYPES


def test_forward_shapes_and_range():
    model = WinProbLSTM(num_event_types=NUM_EVENT_TYPES, hidden_size=16, num_layers=1)
    batch, seq_len = 3, 5
    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (batch, seq_len))
    numeric_features = torch.randn(batch, seq_len, 3)
    win_prob, hidden = model(event_type_idx, numeric_features)
    assert win_prob.shape == (batch, seq_len)
    assert torch.all(win_prob >= 0) and torch.all(win_prob <= 1)
    assert hidden[0].shape == (1, batch, 16)


def test_forward_incremental_matches_full_sequence():
    """Feeding one timestep at a time with carried hidden state must match
    feeding the whole sequence at once — this is what /replay and /live rely on."""
    model = WinProbLSTM(num_event_types=NUM_EVENT_TYPES, hidden_size=16, num_layers=1)
    model.eval()
    seq_len = 4
    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (1, seq_len))
    numeric_features = torch.randn(1, seq_len, 3)

    with torch.no_grad():
        full_probs, _ = model(event_type_idx, numeric_features)

        hidden = None
        incremental_probs = []
        for t in range(seq_len):
            step_prob, hidden = model(
                event_type_idx[:, t : t + 1], numeric_features[:, t : t + 1, :], hidden
            )
            incremental_probs.append(step_prob)
        incremental_probs = torch.cat(incremental_probs, dim=1)

    assert torch.allclose(full_probs, incremental_probs, atol=1e-5)


def test_masked_bce_loss_ignores_padding():
    win_prob = torch.tensor([[0.9, 0.1, 0.5]])
    targets = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # last timestep is padding
    loss = masked_bce_loss(win_prob, targets, mask)
    # Only first two timesteps count: -log(0.9) and -log(0.1)
    expected = -(torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.1))) / 2
    assert torch.allclose(loss, expected, atol=1e-4)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    config = {"num_event_types": NUM_EVENT_TYPES, "event_embed_dim": 4,
              "num_numeric_features": 3, "hidden_size": 8, "num_layers": 1}
    model = WinProbLSTM(**config)
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(model, config, str(ckpt_path))

    loaded = load_model(str(ckpt_path), device="cpu")
    assert not loaded.training  # .eval() was called

    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (1, 3))
    numeric_features = torch.randn(1, 3, 3)
    with torch.no_grad():
        original_out, _ = model(event_type_idx, numeric_features)
        loaded_out, _ = loaded(event_type_idx, numeric_features)
    assert torch.allclose(original_out, loaded_out)
