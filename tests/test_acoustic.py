"""TDT likelihood: the memory-saving normaliser gives the same numbers as a full log-softmax."""

import torch

from zettlecast.podcast import acoustic
from zettlecast.podcast.acoustic import tdt_log_likelihood


def reference(joint, labels, label_lens, blank, durations):
    """The straightforward version: full float64 log-softmax over the vocabulary."""
    n_dur = len(durations)
    token_lp = torch.log_softmax(joint[..., :-n_dur].double(), -1)
    folded = torch.cat([token_lp, joint[..., -n_dur:].double()], -1)
    # token_lp already normalised: logsumexp of it is 0, so the function under test sees the same values
    return tdt_log_likelihood(folded, labels, label_lens, blank, durations)


def test_matches_full_log_softmax_across_blocks(monkeypatch):
    monkeypatch.setattr(acoustic, "LSE_FRAMES", 3)  # several normaliser blocks, last one partial
    torch.manual_seed(0)
    B, T, U, V, durations = 3, 11, 4, 9, [0, 1, 2, 3]
    joint = torch.randn(B, T, U + 1, V + 1 + len(durations))
    labels = torch.randint(0, V, (B, U))
    label_lens = torch.tensor([4, 2, 3])
    blank = V
    got = tdt_log_likelihood(joint, labels, label_lens, blank, durations)
    want = reference(joint, labels, label_lens, blank, durations)
    assert torch.allclose(got, want, atol=1e-9)
    assert torch.isfinite(got).all()
