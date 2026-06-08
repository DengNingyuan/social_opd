from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return summed log p(label tokens) for each sequence."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def continuation_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> torch.Tensor:
    """KL(student || teacher) over aligned continuation positions."""
    student_mask = student_labels[:, 1:] != -100
    teacher_mask = teacher_labels[:, 1:] != -100
    student_logps = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    teacher_logps = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
    losses = []
    for idx in range(student_logps.shape[0]):
        student_pos = student_mask[idx].nonzero(as_tuple=False).flatten()
        teacher_pos = teacher_mask[idx].nonzero(as_tuple=False).flatten()
        steps = min(student_pos.numel(), teacher_pos.numel())
        if steps == 0:
            continue
        s = student_logps[idx, student_pos[:steps]]
        t = teacher_logps[idx, teacher_pos[:steps]]
        losses.append((s.exp() * (s - t.detach())).sum(dim=-1).mean())
    if not losses:
        return torch.zeros((), device=student_logits.device)
    return torch.stack(losses).mean()


def self_opd_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> torch.Tensor:
    return continuation_reverse_kl(student_logits, teacher_logits, student_labels, teacher_labels)
