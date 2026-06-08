from __future__ import annotations

import torch

from losses import self_opd_reverse_kl, sequence_log_probs


def test_sequence_log_probs_ignores_prompt_labels() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])

    logp = sequence_log_probs(logits, labels)

    assert logp.item() == torch.log(torch.tensor(1 / 3)).mul(2).item()


def test_self_opd_reverse_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])

    loss = self_opd_reverse_kl(logits, logits.clone(), labels, labels)

    assert loss.item() == 0.0
