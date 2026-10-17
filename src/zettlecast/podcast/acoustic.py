"""
Acoustic scoring: how well does a candidate transcript fit the audio?

Uses Parakeet-TDT itself as the judge: the negative log-likelihood of a text
given an audio clip (the TDT loss). A sound-alike spelling ("Gustav Iden" for
a heard "Gustav Eden") costs a few nats per changed word; a different word
("Ingebrigtsen") costs tens. That makes it a check on text corrections that
needs no human: a correction is only accepted if the audio supports it.

NeMo's TDT loss kernel needs NVVM (CUDA toolkit) and its pure-PyTorch
reference loops over every (t, u) in Python, so this module has a vectorised
forward pass: a loop over frames only, with the same-frame label chain
(duration 0) solved by a cumulative log-sum-exp.
"""

import logging
from typing import List, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


def tdt_log_likelihood(
    joint: torch.Tensor,
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    blank: int,
    durations: Sequence[int],
) -> torch.Tensor:
    """
    log P(labels | audio) for a TDT model, per batch item.

    Same recursion as nemo TDTLossPytorch (Eq. 7 of the TDT paper), with all
    items sharing the same number of frames.

    Args:
        joint: (B, T, U+1, V+1+D) raw joint outputs (token logits incl. blank, then duration logits)
        labels: (B, U) token ids, padded
        label_lens: (B,) number of real labels per item
        blank: blank token id
        durations: TDT durations, e.g. [0, 1, 2, 3, 4]; must contain 0
    """
    n_dur = len(durations)
    if 0 not in durations:
        raise ValueError("durations must include 0")
    token_lp = torch.log_softmax(joint[..., :-n_dur].double(), -1)
    dur_lp = torch.log_softmax(joint[..., -n_dur:].double(), -1)
    B, T, U1, _ = token_lp.shape
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float64, device=joint.device)

    blank_lp = token_lp[..., blank]  # (B, T, U1)
    label_lp = torch.full((B, T, U1), float("-inf"), dtype=torch.float64, device=joint.device)
    if U1 > 1:
        idx = labels[:, None, : U1 - 1, None].expand(B, T, U1 - 1, 1)
        label_lp[:, :, : U1 - 1] = token_lp[:, :, : U1 - 1].gather(-1, idx).squeeze(-1)

    zero = durations.index(0)
    alpha = torch.full((B, T, U1), float("-inf"), dtype=torch.float64, device=joint.device)
    for t in range(T):
        base = torch.full((B, U1), float("-inf"), dtype=torch.float64, device=joint.device)
        if t == 0:
            base[:, 0] = 0.0
        for n, d in enumerate(durations):
            if d == 0 or t - d < 0:
                continue
            prev = alpha[:, t - d]
            base = torch.logaddexp(base, prev + blank_lp[:, t - d] + dur_lp[:, t - d, :, n])
            base[:, 1:] = torch.logaddexp(
                base[:, 1:], prev[:, :-1] + label_lp[:, t - d, :-1] + dur_lp[:, t - d, :-1, n]
            )
        # Same-frame chain: alpha[u] = logaddexp(base[u], alpha[u-1] + e[u-1])
        e = label_lp[:, t, :-1] + dur_lp[:, t, :-1, zero]
        cum = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=joint.device), e.cumsum(1)], 1)
        alpha[:, t] = cum + torch.logcumsumexp(base - cum, dim=1)

    final = torch.full((B,), float("-inf"), dtype=torch.float64, device=joint.device)
    rows = torch.arange(B, device=joint.device)
    for n, d in enumerate(durations):
        if d == 0 or T - d < 0:
            continue
        final = torch.logaddexp(
            final,
            alpha[rows, T - d, label_lens] + blank_lp[rows, T - d, label_lens] + dur_lp[rows, T - d, label_lens, n],
        )
    return torch.where(torch.isnan(final), neg_inf, final)


class AcousticScorer:
    """Scores candidate texts against an audio clip with a Parakeet-TDT model."""

    def __init__(self, model):
        self.model = model
        self.durations = list(model.cfg.model_defaults.tdt_durations)
        self.blank = model.decoder.blank_idx

    @torch.no_grad()
    def nll(self, audio: np.ndarray, texts: List[str]) -> List[float]:
        """
        Negative log-likelihood of each text for the 16 kHz mono clip (lower fits better).
        """
        if not texts:
            return []
        m = self.model
        device = next(m.parameters()).device
        dtype = next(m.parameters()).dtype
        sig = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)[None]
        sig_len = torch.tensor([sig.shape[1]], device=device)
        feats, feat_len = m.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = m.encoder(audio_signal=feats.to(dtype), length=feat_len)
        enc = enc[..., : int(enc_len[0])].expand(len(texts), -1, -1)

        ids = [m.tokenizer.text_to_ids(t) for t in texts]
        lens = torch.tensor([len(i) for i in ids], device=device)
        tgt = torch.zeros(len(ids), max(1, max(map(len, ids))), dtype=torch.long, device=device)
        for k, i in enumerate(ids):
            tgt[k, : len(i)] = torch.tensor(i, device=device)
        dec, _, _ = m.decoder(targets=tgt, target_length=lens)
        joint = m.joint.joint(enc.transpose(1, 2), dec.transpose(1, 2))
        ll = tdt_log_likelihood(joint, tgt, lens, self.blank, self.durations)
        return (-ll).float().cpu().tolist()
