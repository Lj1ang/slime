#!/usr/bin/env python3
"""
Plot GPU utilization and request timeline from slime log gpu_profile.jsonl.

Usage (script lives in repo root; profile files in log/ by default):
  python plot_gpu_profile.py
  python plot_gpu_profile.py --input log/gpu_profile.jsonl [--output-dir log]
  python plot_gpu_profile.py --input log/gpu_profile.jsonl --inference-profile log/inference_profile.jsonl

With --inference-profile, merges SGLang inference events (prefill, decode, unified) into the timeline.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Repo root = directory containing this script; default log dir is repo_root/log
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "log"


def load_profile_record(jsonl_path: Path):
    """Load the first record that has utilization samples and (optionally) request times. Rollout-only records have utilization but no request_times."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    for rec in records:
        if "gpu_profile/gpu_utilization_samples" not in rec:
            continue
        samples = rec["gpu_profile/gpu_utilization_samples"]
        if not samples:
            continue
        # Require request_times unless this is a rollout-window-only record
        gpu_request_keys = [k for k in rec if re.match(r"gpu_profile/gpu\d+_request_times", k)]
        if not gpu_request_keys and not rec.get("gpu_profile/rollout_window_only"):
            continue
        return rec
    raise ValueError(f"No record with gpu_utilization_samples (and gpu*_request_times or rollout_window_only) in {jsonl_path}")


def get_num_gpus(rec: dict) -> int:
    n = 0
    while f"gpu_profile/gpu{n}_request_times" in rec:
        n += 1
    if n == 0 and "gpu_profile/gpu_utilization_samples" in rec:
        samples = rec["gpu_profile/gpu_utilization_samples"]
        if samples:
            n = len(samples[0][1])
    # Also count GPUs from inference event keys (prefill_times, decode_times, unified_times)
    for key in rec:
        if key.startswith("gpu_profile/gpu") and "_times" in key:
            m = re.match(r"gpu_profile/gpu(\d+)_", key)
            if m:
                n = max(n, int(m.group(1)) + 1)
    return n


def normalize_time(t: float, t0: float) -> float:
    return t - t0


def draw_gpu_utilization_ax(
    rec: dict,
    t0: float,
    ax: plt.Axes,
    x_range: tuple[float, float] | None,
    cax: plt.Axes | None = None,
) -> tuple[tuple[float, float] | None, object]:
    """Draw GPU utilization. Returns (x_range, mesh). Pass cax to draw colorbar in separate axes so ax keeps same width as timeline."""
    samples = rec.get("gpu_profile/gpu_utilization_samples")
    if not samples:
        return None, None

    times = np.array([s[0] for s in samples])
    utils = np.array([s[1] for s in samples])
    times_rel = np.array([normalize_time(t, t0) for t in times])
    num_gpus = utils.shape[1]
    interval_sec = rec.get("gpu_profile/interval_ms", 200) / 1000.0
    t_edges = np.concatenate([times_rel, [times_rel[-1] + interval_sec]])
    y_edges = np.arange(num_gpus + 1)

    if x_range is None:
        x_range = (0.0, float(times_rel[-1] - times_rel[0] + interval_sec))

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)
    mesh = ax.pcolormesh(
        t_edges, y_edges, utils.T, cmap=cmap, norm=norm, shading="flat",
    )
    ax.set_ylim(num_gpus, 0)
    ax.set_yticks(np.arange(num_gpus) + 0.5)
    ax.set_yticklabels([f"GPU {g}" for g in range(num_gpus)])
    ax.set_ylabel("")
    ax.set_xlabel("relative time (s)")
    ax.set_title("GPU Utilization")
    ax.set_xlim(x_range)
    if cax is not None:
        plt.colorbar(mesh, cax=cax, label="Utilization %")
    else:
        plt.colorbar(mesh, ax=ax, shrink=0.6, label="Utilization %")

    return x_range, mesh


# Event categories for request timeline (key suffix in gpu_profile.jsonl, display label, color)
# Training (actor) events first, then SGLang inference events when --inference-profile is used
EVENT_CATEGORIES = [
    ("ref_log_probs_times", "ref_log_probs", "tab:cyan"),
    ("log_probs_times", "log_probs", "tab:blue"),
    ("request_times", "train_step", "tab:orange"),
    ("prefill_times", "prefill", "tab:blue"),
    ("decode_times", "decode", "tab:orange"),
    ("unified_times", "unified", "tab:green"),
]

OTHER_LABEL = "other (GPU busy)"
OTHER_COLOR = (0.85, 0.85, 0.85)  # light gray


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping (start, end) intervals."""
    if not intervals:
        return []
    sorted_i = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_i[0])]
    for s, e in sorted_i[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def _gaps_in_range(merged: list[tuple[float, float]], x_min: float, x_max: float) -> list[tuple[float, float]]:
    """Return (start, end) gaps inside [x_min, x_max] not covered by merged intervals."""
    out = []
    current = x_min
    for s, e in merged:
        if e <= current:
            continue
        if s > current:
            out.append((current, min(s, x_max)))
        current = max(current, e)
        if current >= x_max:
            break
    if current < x_max:
        out.append((current, x_max))
    return out


def draw_request_timeline_ax(rec: dict, t0: float, ax: plt.Axes, x_range: tuple[float, float] | None) -> None:
    """Draw request timeline on given axes. Uses x_range if provided. Draws all event categories (ref_log_probs, log_probs, train_step) with different colors. Fills time not attributed to any event as 'other (GPU busy)' when GPU utilization is present."""
    num_gpus = get_num_gpus(rec)
    if num_gpus == 0:
        return

    x_min, x_max = (0.0, 0.0)
    if x_range is not None:
        x_min, x_max = x_range

    # Per-GPU: collect all known event intervals (relative time)
    intervals_per_gpu: list[list[tuple[float, float]]] = [[] for _ in range(num_gpus)]
    for key_suffix, _label, _color in EVENT_CATEGORIES:
        for gpu in range(num_gpus):
            key = f"gpu_profile/gpu{gpu}_{key_suffix}"
            if key not in rec:
                continue
            for start, end in rec[key]:
                start_rel = normalize_time(start, t0)
                end_rel = normalize_time(end, t0)
                intervals_per_gpu[gpu].append((start_rel, end_rel))

    # Draw "other" (unaccounted) segments first so they sit behind the labeled events
    legend_used_other = False
    if x_range is not None and x_max > x_min:
        bar_height = 0.4
        for gpu in range(num_gpus):
            merged = _merge_intervals(intervals_per_gpu[gpu])
            gaps = _gaps_in_range(merged, x_min, x_max)
            y_center = num_gpus - 1 - gpu
            for start_rel, end_rel in gaps:
                if end_rel - start_rel < 0.01:
                    continue
                legend_used_other = True
                ax.barh(
                    y_center, end_rel - start_rel, left=start_rel, height=bar_height,
                    color=OTHER_COLOR, edgecolor=(0.7, 0.7, 0.7), linewidth=0.2,
                )

    all_ends = []
    legend_used = set()

    # Draw in order so train_step (many short bars) is on top
    for key_suffix, label, color in EVENT_CATEGORIES:
        edge = color
        for gpu in range(num_gpus):
            key = f"gpu_profile/gpu{gpu}_{key_suffix}"
            if key not in rec:
                continue
            times = rec[key]
            if not times:
                continue
            legend_used.add((label, color))
            y_center = num_gpus - 1 - gpu
            bar_height = 0.4
            for start, end in times:
                start_rel = normalize_time(start, t0)
                end_rel = normalize_time(end, t0)
                all_ends.append((start_rel, end_rel))
                ax.barh(y_center, end_rel - start_rel, left=start_rel, height=bar_height, color=color, edgecolor=edge, linewidth=0.3)

    # Request boundaries (vertical lines) only for train_step to avoid clutter
    for gpu in range(num_gpus):
        key = f"gpu_profile/gpu{gpu}_request_times"
        if key not in rec:
            continue
        for start, end in rec[key]:
            start_rel = normalize_time(start, t0)
            end_rel = normalize_time(end, t0)
            ax.axvline(start_rel, color="black", linewidth=0.35, alpha=0.6)
            ax.axvline(end_rel, color="black", linewidth=0.35, alpha=0.6)

    ax.set_ylim(-0.2, num_gpus)
    ax.set_yticks([num_gpus - 1 - g + 0.2 for g in range(num_gpus)])
    ax.set_yticklabels([f"GPU {g}" for g in range(num_gpus)])
    ax.set_xlabel("relative time (s)")
    ax.set_title("Request Timeline")
    if x_range is not None:
        ax.set_xlim(x_range)
    elif all_ends:
        t_min = min(e[0] for e in all_ends)
        t_max = max(e[1] for e in all_ends)
        ax.set_xlim(max(0, t_min - 1), t_max + 1)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, linewidth=8, label=label)
        for (key_suffix, label, color) in EVENT_CATEGORIES
        if (label, color) in legend_used
    ]
    if legend_used_other:
        legend_elements.append(
            Line2D([0], [0], color=OTHER_COLOR, linewidth=8, label=OTHER_LABEL),
        )
    legend_elements.append(
        Line2D([0], [0], color="black", linewidth=1, linestyle="-", label="request start/end"),
    )
    ax.legend(handles=legend_elements, loc="upper right")


def draw_combined(rec: dict, t0: float, output_path: Path, window_duration_sec: float) -> None:
    """Draw GPU utilization and request timeline in one figure. Use GridSpec so both subplots have the same x-axis width (colorbar in its own column)."""
    num_gpus = get_num_gpus(rec)
    fig = plt.figure(figsize=(12, max(6, num_gpus * 2.0)))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 0.04], height_ratios=[1, 1], hspace=0.45)
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # Single x range for both subplots (0 to window duration) so axes align
    x_range = (0.0, window_duration_sec)
    n_ticks = 8
    tick_positions = np.linspace(x_range[0], x_range[1], n_ticks)
    tick_labels = [f"{int(round(t))}" for t in tick_positions]

    draw_gpu_utilization_ax(rec, t0, ax1, x_range, cax=cax)
    draw_request_timeline_ax(rec, t0, ax2, x_range)

    # Same x-axis limits and tick positions on both subplots for alignment
    ax1.set_xlim(x_range)
    ax2.set_xlim(x_range)
    ax1.set_xticks(tick_positions)
    ax2.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax2.set_xticklabels(tick_labels)
    ax1.tick_params(axis="x", labelbottom=True)
    plt.setp(ax1.get_xticklabels(), visible=True)

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.06, top=0.94, hspace=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_and_merge_inference_profile(rec: dict, inference_path: Path) -> None:
    """Merge inference_profile.jsonl (SGLang prefill/decode/unified) into rec in-place."""
    if not inference_path.exists():
        return
    t0_candidates = []
    with open(inference_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("gpu_profile/inference_profile"):
                continue
            if "gpu_profile/inference_t0_sec" in row:
                t0_candidates.append(row["gpu_profile/inference_t0_sec"])
            for key in row:
                if not key.startswith("gpu_profile/gpu") or "_times" not in key:
                    continue
                if key in rec:
                    rec[key] = rec[key] + row[key]
                else:
                    rec[key] = list(row[key])
    if t0_candidates:
        rec["gpu_profile/inference_t0_sec"] = min(t0_candidates)


def main():
    parser = argparse.ArgumentParser(description="Plot GPU profile from gpu_profile.jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_LOG_DIR / "gpu_profile.jsonl",
        help="Input JSONL path (default: log/gpu_profile.jsonl under repo root)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--inference-profile", type=Path, default=None, help="Optional inference_profile.jsonl (SGLang prefill/decode/unified) to merge into timeline")
    args = parser.parse_args()
    input_path = args.input
    output_dir = args.output_dir or input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rec = load_profile_record(input_path)
    if args.inference_profile:
        load_and_merge_inference_profile(rec, args.inference_profile)
    num_gpus = get_num_gpus(rec)

    # Global t0 = min of utilization window start and first request start
    t0 = rec.get("gpu_profile/gpu_utilization_start_sec") or 0
    for gpu in range(num_gpus):
        key = f"gpu_profile/gpu{gpu}_first_request_start_sec"
        if key in rec:
            t0 = min(t0, rec[key])
    samples = rec.get("gpu_profile/gpu_utilization_samples") or []
    if samples:
        t0 = min(t0, samples[0][0])
    # Inference profile may have inference_t0_sec; use min for alignment
    if "gpu_profile/inference_t0_sec" in rec:
        t0 = min(t0, rec["gpu_profile/inference_t0_sec"])

    # Rollout window duration (relative time span)
    t_end = t0
    if samples:
        t_end = max(t_end, samples[-1][0])
    for gpu in range(num_gpus):
        key = f"gpu_profile/gpu{gpu}_last_request_end_sec"
        if key in rec:
            t_end = max(t_end, rec[key])
    for suffix in ("prefill_times", "decode_times", "unified_times"):
        for gpu in range(num_gpus):
            key = f"gpu_profile/gpu{gpu}_{suffix}"
            for start, end in rec.get(key, []):
                t_end = max(t_end, end)
    window_duration_sec = t_end - t0

    out_file = output_dir / "gpu_profile.png"
    draw_combined(rec, t0, out_file, window_duration_sec)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
