# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""VinePPO: step-boundary detection, V-hat broadcast, branch-rollout dispatch.

Ported to slime from https://github.com/Lj1ang/verl/pull/2.

VinePPO replaces the GRPO group-mean baseline with per-step value estimates
V_hat(s_t) obtained by Monte-Carlo branch rollouts off each step boundary in
the main rollout. The per-token advantage becomes (R - V_hat_t) * mask.
"""

import time
from typing import Callable, Sequence

import torch


def find_step_boundaries(
    token_ids: Sequence[int],
    tokenizer,
    separators: Sequence[str],
) -> list[int]:
    """Return token positions where the decoded suffix matches any separator.

    A "boundary" is a position b such that decoding tokens[:b] ends with a
    separator string. The branch rollout at boundary b uses tokens[:b] as the
    prefix.

    Args:
        token_ids: Response token ids (no prompt prefix).
        tokenizer: Object with `.decode(ids) -> str`.
        separators: Suffix strings that mark a reasoning-step end.

    Returns:
        Sorted list of boundary positions in [1, len(token_ids)].
    """
    boundaries: list[int] = []
    prev_decoded = ""
    for i in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:i])
        if decoded == prev_decoded:
            continue
        if any(decoded.endswith(sep) for sep in separators):
            boundaries.append(i)
        prev_decoded = decoded
    return boundaries


def broadcast_value_to_tokens(
    boundaries: Sequence[int],
    v_hat: Sequence[float],
    response_length: int,
    fallback: float,
) -> torch.Tensor:
    """Broadcast per-boundary V-hat values to per-token values.

    Token at position p is assigned V-hat[k] where k is the largest index with
    boundaries[k] <= p. Tokens before the first boundary use `fallback`
    (typically the GRPO group-mean baseline).
    """
    if len(boundaries) != len(v_hat):
        raise ValueError(f"len(boundaries)={len(boundaries)} != len(v_hat)={len(v_hat)}")
    out = torch.full((response_length,), float(fallback))
    for k, b in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else response_length
        if b < response_length:
            out[b:end] = float(v_hat[k])
    return out


def compute_vine_advantage(
    rewards: torch.Tensor,
    v_hat_per_token: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute VinePPO per-token advantages and returns.

    Slime adopts a per-sample list-of-tensors layout (see
    ``compute_advantages_and_returns`` in slime/backends/megatron_utils/loss.py).

    Args:
        rewards: shape (bs,). Scalar outcome reward per rollout.
        v_hat_per_token: per-rollout tensors of length response_length, each
            already broadcast from boundary V-hat values via
            ``broadcast_value_to_tokens``.
        loss_masks: per-rollout response-only masks, one tensor per rollout.

    Returns:
        advantages: list of (response_length,) tensors with (R - V_hat) * mask.
        returns: list of (response_length,) tensors with R * mask.
    """
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for i, (vh, mask) in enumerate(zip(v_hat_per_token, loss_masks)):
        r = float(rewards[i])
        adv = (r - vh) * mask
        ret = torch.full_like(vh, r) * mask
        advantages.append(adv)
        returns.append(ret)
    return advantages, returns


def _subsample_boundaries(boundaries: list[int], cap: int) -> list[int]:
    """Uniformly subsample boundaries down to `cap`, preserving order."""
    if len(boundaries) <= cap:
        return boundaries
    indices = torch.linspace(0, len(boundaries) - 1, cap).round().long().tolist()
    seen: set[int] = set()
    out: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(boundaries[i])
    return out


def build_branch_prompts(
    prompt_token_ids: list[list[int]],
    response_token_ids: list[list[int]],
    tokenizer,
    step_separators: Sequence[str],
    max_branches_per_rollout: int,
    num_branches: int,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Build prefix-conditioned prompts for branch sampling.

    For each main rollout i with response token ids r_i, find step boundaries
    in r_i. For each boundary b_k, the branch prefix is prompt_i ++ r_i[:b_k].
    Each (i, k) is tiled by num_branches.

    Returns:
        branch_prompt_ids: flat list of prefix token-id lists of length
            sum_i M_i * num_branches (one per branch).
        boundaries_per_rollout: per-rollout boundary positions (post-cap).
        rollout_index: source rollout id for each branch.
    """
    boundaries_per_rollout: list[list[int]] = []
    branch_prompt_ids: list[list[int]] = []
    rollout_index: list[int] = []

    for i, response_ids in enumerate(response_token_ids):
        boundaries = find_step_boundaries(response_ids, tokenizer, separators=step_separators)
        boundaries = _subsample_boundaries(boundaries, max_branches_per_rollout)
        boundaries_per_rollout.append(boundaries)

        for b in boundaries:
            prefix = list(prompt_token_ids[i]) + list(response_ids[:b])
            for _ in range(num_branches):
                branch_prompt_ids.append(prefix)
                rollout_index.append(i)

    return branch_prompt_ids, boundaries_per_rollout, rollout_index


def run_branch_rollouts(
    prompt_token_ids: list[list[int]],
    response_token_ids: list[list[int]],
    tokenizer,
    branch_generate_fn: Callable[[list[list[int]]], list[float]],
    num_branches: int,
    step_separators: Sequence[str],
    max_branches_per_rollout: int,
) -> tuple[torch.Tensor, list[list[int]], dict]:
    """Run branch rollouts and aggregate per-(rollout, boundary) V-hat.

    Args:
        prompt_token_ids: per-rollout prompt token id lists.
        response_token_ids: per-rollout response token id lists.
        tokenizer: HuggingFace tokenizer.
        branch_generate_fn: callable that takes a flat list of branch prefix
            token-id sequences and returns a list of scalar rewards (one per
            branch) after generating to completion and scoring.
        num_branches: K' branches per boundary.
        step_separators: suffix strings marking step ends.
        max_branches_per_rollout: cap M per rollout.

    Returns:
        v_hat: (bs, M_max) tensor; v_hat[i, k] = mean reward across the K'
            branches at boundary k of rollout i. Padded entries are 0.0.
        boundaries_per_rollout: list of lists of boundary positions.
        metrics: dict with v_hat_mean, v_hat_std, branches_per_rollout_mean,
            boundaries_per_rollout_mean/max, branch_rollout_time_s.
    """
    t0 = time.time()
    branch_prompts, boundaries_per_rollout, rollout_index = build_branch_prompts(
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        tokenizer=tokenizer,
        step_separators=step_separators,
        max_branches_per_rollout=max_branches_per_rollout,
        num_branches=num_branches,
    )

    bs = len(response_token_ids)
    M_max = max((len(b) for b in boundaries_per_rollout), default=0)
    v_hat = torch.zeros(bs, max(M_max, 1))

    metrics = {
        "boundaries_per_rollout_mean": float(
            sum(len(b) for b in boundaries_per_rollout) / max(bs, 1)
        ),
        "boundaries_per_rollout_max": float(M_max),
        "branches_per_rollout_mean": float(len(branch_prompts) / max(bs, 1)),
    }

    if not branch_prompts:
        metrics["v_hat_mean"] = 0.0
        metrics["v_hat_std"] = 0.0
        metrics["branch_rollout_time_s"] = time.time() - t0
        return v_hat, boundaries_per_rollout, metrics

    branch_rewards = branch_generate_fn(branch_prompts)
    branch_rewards_t = torch.as_tensor(branch_rewards, dtype=torch.float32)

    boundary_index: list[int] = []
    for bnds in boundaries_per_rollout:
        for k in range(len(bnds)):
            boundary_index.extend([k] * num_branches)
    assert len(boundary_index) == len(rollout_index) == branch_rewards_t.numel()

    sums = torch.zeros(bs, max(M_max, 1))
    counts = torch.zeros(bs, max(M_max, 1))
    for n, (i, k) in enumerate(zip(rollout_index, boundary_index)):
        sums[i, k] += float(branch_rewards_t[n])
        counts[i, k] += 1.0
    safe_counts = counts.clamp(min=1.0)
    v_hat = sums / safe_counts

    metrics["v_hat_mean"] = float(v_hat[counts > 0].mean()) if (counts > 0).any() else 0.0
    metrics["v_hat_std"] = float(v_hat[counts > 0].std()) if (counts > 0).sum() > 1 else 0.0
    metrics["branch_rollout_time_s"] = time.time() - t0
    return v_hat, boundaries_per_rollout, metrics


def assemble_v_hat_per_token(
    v_hat: torch.Tensor,
    boundaries_per_rollout: list[list[int]],
    response_lengths: list[int],
    fallback_per_rollout: torch.Tensor,
) -> list[torch.Tensor]:
    """Build the per-token V_hat tensor list expected by ``compute_vine_advantage``.

    Args:
        v_hat: (bs, M_max) from ``run_branch_rollouts``.
        boundaries_per_rollout: as returned by ``run_branch_rollouts``.
        response_lengths: per-rollout response lengths in tokens.
        fallback_per_rollout: (bs,) GRPO group-mean fallback for tokens before
            the first boundary.
    """
    out: list[torch.Tensor] = []
    for i, bnd in enumerate(boundaries_per_rollout):
        vh = v_hat[i, : len(bnd)].tolist() if len(bnd) > 0 else []
        out.append(
            broadcast_value_to_tokens(
                boundaries=bnd,
                v_hat=vh,
                response_length=response_lengths[i],
                fallback=float(fallback_per_rollout[i]),
            )
        )
    return out


def group_mean_baseline(rewards: torch.Tensor, group_indices: Sequence[int] | None) -> torch.Tensor:
    """GRPO-style group-mean fallback for V_hat when a rollout has no boundaries.

    Args:
        rewards: (bs,) scalar reward per rollout.
        group_indices: per-rollout group ids (e.g. ``Sample.group_index``). If
            None, uses a single global group.
    """
    out = rewards.clone()
    if group_indices is None:
        out[:] = rewards.mean() if rewards.numel() > 0 else 0.0
        return out
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for i, u in enumerate(group_indices):
        groups[u].append(i)
    for _, idxs in groups.items():
        m = rewards[idxs].mean()
        for i in idxs:
            out[i] = m
    return out
