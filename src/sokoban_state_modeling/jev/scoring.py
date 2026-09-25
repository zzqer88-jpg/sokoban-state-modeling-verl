"""Candidate-only probabilities and Pev-compatible KL(q || p)."""
import torch
from torch import Tensor


def log_probabilities(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != 7 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite candidate logits [cells,7]")
    return logits.float().log_softmax(-1)


def cell_kl(logits: Tensor, target: Tensor) -> Tensor:
    logp = log_probabilities(logits)
    q = target.detach().to(device=logp.device, dtype=torch.float32)
    if q.shape != logp.shape or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Invalid target distribution")
    if not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=0):
        raise ValueError("Each target distribution must sum to one")
    return (torch.xlogy(q, q) - q * logp).sum(-1)
