# Lj1ang's contribution: GPU + inference profiling for SGLang rollout

This document describes the work introduced by Lei Jiang (`Lj1ang` on GitHub) on the
`profile` branch, on top of upstream `67a21a1`. Two commits in total:

| Commit    | Title           | What it does                                                      |
| --------- | --------------- | ----------------------------------------------------------------- |
| `0c50198` | init profile    | Initial profiler module, FSDP wiring, args, plotter, geo3k tweak. |
| `ebaa013` | update profile  | Refines the profiler; adds the inference-profile path.            |

## TL;DR

Adds a **GPU + request profiler** that captures three things during slime training with
SGLang rollout, then renders them into a single PNG:

1. **GPU utilization** sampled every 200 ms via `nvidia-smi` in a background thread on
   rank 0. Drawn as a heatmap across GPUs and time.
2. **Per-request timeline events** — `train_step`, `log_probs`, `ref_log_probs` on the
   actor side; `prefill`, `decode`, `unified` on the SGLang side. Drawn as coloured bars
   per GPU lane, with grey "other (GPU busy)" fill for time that is GPU-busy but not
   attributed to any known event.
3. **Request count per GPU** for load-balance sanity-checking.

The plotter (`plot_gpu_profile.py`) consumes two JSONL files written during training:

- `log/gpu_profile.jsonl` — actor-side aggregate, written from rank 0 every
  `--gpu-profile-plot-steps` rollouts (default 30).
- `log/inference_profile.jsonl` — per-request rollout events, appended inline by the
  SGLang rollout path whenever the engine returns timing in its meta_info.

## Files changed

```
examples/geo3k_vlm_multi_turn/README.md                                   |   8 +-
examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py                 |  16 +-
log/gpu_profile.jsonl                                                     |   1 +   (sample)
log/gpu_profile.png                                                       | bin     (sample plot)
plot_gpu_profile.py                                                       | 376 +++  (new)
slime/backends/fsdp_utils/actor.py                                        | 118 ++-
slime/ray/rollout.py                                                      |  13 +-
slime/rollout/sglang_rollout.py                                           |  60 ++-
slime/utils/arguments.py                                                  |  48 +++
slime/utils/profiler.py                                                   | 297 +++  (new)
```

10 files, ~927 insertions / 10 deletions.

## Architecture

Three independent producers, one plotter:

```
        ┌──────────────────────────────────────────────────────┐
        │                       slime training                 │
        │                                                      │
        │  ┌─────────────────┐         ┌───────────────────┐   │
        │  │ FSDPTrainRayActor│        │ SGLang rollout    │   │
        │  │   (per rank)    │         │   (per request)   │   │
        │  │                 │         │                   │   │
        │  │ Profiler        │         │ write_inference_  │   │
        │  │ .record(...)    │         │  profile(events,  │   │
        │  │   train_step    │         │  t0_sec=...)      │   │
        │  │   log_probs     │         │                   │   │
        │  │   ref_log_probs │         │   prefill/decode/ │   │
        │  └────────┬────────┘         │   unified         │   │
        │           │ all_gather       └─────────┬─────────┘   │
        │           ▼                            │             │
        │  ┌─────────────────┐                   │             │
        │  │ GpuUtilization  │                   │             │
        │  │ Collector (rank 0)                  │             │
        │  │   nvidia-smi    │                   │             │
        │  │   every 200 ms  │                   │             │
        │  └────────┬────────┘                   │             │
        │           ▼                            │             │
        │  ┌─────────────────┐                   │             │
        │  │ run_gpu_profile_│                   │             │
        │  │ report (rank 0) │                   │             │
        │  └────────┬────────┘                   │             │
        └───────────┼────────────────────────────┼─────────────┘
                    ▼                            ▼
            log/gpu_profile.jsonl       log/inference_profile.jsonl
                    │                            │
                    └────────────┬───────────────┘
                                 ▼
                        plot_gpu_profile.py
                                 │
                                 ▼
                          log/gpu_profile.png
```

## Module walkthrough

### `slime/utils/profiler.py` (new)

Three classes plus three free functions; ~300 lines.

**`Profiler`** — per-rank state. Stores a dict keyed by `(rollout_id, dp_rank)` whose
values are lists of `(start_sec, end_sec, event_type)` tuples. Has a separate
`_request_count` counter that only increments on `EVENT_TRAIN_STEP` events, so it
tracks "training requests assigned to this GPU", not phase events. `clear()` is called on
all ranks after each report.

**`GpuUtilizationCollector`** — rank-0-only background thread that runs
`nvidia-smi --query-gpu=utilization.gpu` every 200 ms (`GPU_UTIL_INTERVAL_SEC`). Stop
signal is a `threading.Event`, which doubles as the sleep mechanism — calling `stop()`
exits the loop without waiting out the remainder of the interval. The collector
self-disables (logs a warning) if `nvidia-smi` is unavailable, so CPU-only nodes don't
spam errors.

**Free functions:**

- `log_request_profile_metrics(...)` — composes the JSONL line and emits to (a)
  `logging_utils.log()` (i.e. wandb / file logger) and (b) the local
  `log/gpu_profile.jsonl` append. The line schema is fixed and documented in the
  function docstring.
- `run_gpu_profile_report(...)` — thin wrapper around `log_request_profile_metrics`.
  Signature is intentionally wide to leave room for follow-up report types.
- `write_inference_profile(...)` — appends one line to
  `log/inference_profile.jsonl`. Lives in this module rather than the rollout module so
  the schema (`gpu_profile/...` keys) stays in one place.

Constants:

```python
GPU_UTIL_INTERVAL_SEC = 0.2     # 200 ms
PROFILER_LOG_DIR = "log"
PROFILER_LOG_FILENAME = "gpu_profile.jsonl"
INFERENCE_PROFILER_LOG_FILENAME = "inference_profile.jsonl"

EVENT_TRAIN_STEP = "train_step"
EVENT_LOG_PROBS = "log_probs"
EVENT_REF_LOG_PROBS = "ref_log_probs"
INFERENCE_EVENT_PREFILL = "prefill"
INFERENCE_EVENT_DECODE = "decode"
INFERENCE_EVENT_UNIFIED = "unified"
```

### `slime/backends/fsdp_utils/actor.py`

Three integration points inside `FSDPTrainRayActor`:

1. **Construction**: when `--enable-gpu-profile` is on, every rank gets a `Profiler`;
   rank 0 additionally starts a `GpuUtilizationCollector`.
2. **Phase recording**: ref-log-probs, actor-log-probs, and per-`_train_step` request
   spans are bracketed by `torch.cuda.synchronize()` + `time.time()` to capture true GPU
   durations. Each span is recorded with its event type (`EVENT_REF_LOG_PROBS`,
   `EVENT_LOG_PROBS`, default `EVENT_TRAIN_STEP`).
3. **Periodic gather + report**: every `--gpu-profile-plot-steps` rollouts (default 30):
   - All ranks call `dist.all_gather_object` with their local profiler data.
   - Rank 0 splits the flattened events into `train_step` (per-rank request_times) vs
     categorical (`event_times_per_rank_by_type`).
   - Rank 0 calls `run_gpu_profile_report` to write to wandb + `gpu_profile.jsonl`.
   - **All** ranks then call `Profiler.clear()` to bound memory growth.

`_train_step` gains a new `rollout_id: int` parameter, threaded down from the caller
solely to key the request event. The `gpu_profile_rollout_time_window` field
in `rollout_data` is consumed here: when present, the report drops training event times
and clips GPU utilization to the rollout window only (rollout-only profiling mode).

### `slime/ray/rollout.py`

Three small touches on `RolloutManager`:

- `_get_rollout_data` now returns a third value: a `(rollout_start_sec, rollout_end_sec)`
  tuple wrapping the `call_rollout_fn` span on the live path, or `None` on the
  debug-load path.
- `.generate()` forwards that tuple into the train data dict as
  `gpu_profile_rollout_time_window` only when `--enable-gpu-profile` is on.
- `_split_train_data_by_dp` duplicates the window into every DP partition's payload —
  the window is the same for all partitions but each train-side actor reads it from its
  own `rollout_data`.

### `slime/rollout/sglang_rollout.py`

Adds `_record_inference_profile_if_present(args, state, request_start_sec, meta_info)`.
The function tries two timing schemes from SGLang's `meta_info`:

1. **Explicit phase boundaries** — `prefill_start_sec` / `prefill_end_sec` /
   `decode_*_sec` / `unified_*_sec`.
2. **Duration shorthand** — `first_token_time` (prefill duration) + `total_time`
   (end-to-end), anchored at `request_start_sec`.

SGLang upstream currently only ships shape (2); shape (1) is forward-looking. The result
is built into the per-rank list-of-lists shape the plotter wants and forwarded to
`write_inference_profile`. The hook lives at the end of `generate(...)` and is gated on
`enable_inference_profile`.

### `slime/utils/arguments.py`

Five new CLI flags + one validation rule:

| Flag                            | Default | Meaning                                                              |
| ------------------------------- | ------- | -------------------------------------------------------------------- |
| `--enable-gpu-profile`          | False   | Master switch. Turns on the FSDP-actor profiling and rollout window. |
| `--gpu-profile-plot-steps`      | 30      | Emit a report every N rollout steps.                                 |
| `--gpu-profile-output-dir`      | None    | Override `log/`; falls back to `GPU_PROFILE_DIR` env var.            |
| `--gpu-profile-heatmap-steps`   | 20      | Reserved for future heatmap rendering — currently unused.            |
| `--gpu-profile-rollout-events`  | True    | Sub-switch for inference-side events. Use `--no-...` to disable.     |
| `--enable-inference-profile`    | False   | Stand-alone switch for the rollout-side recorder.                    |

Validation rule: when `--enable-gpu-profile` is on AND `--gpu-profile-rollout-events` is
on (default), `enable_inference_profile` is force-set to True. Reason: the actor-side
profile relies on the rollout-side writer for `prefill`/`decode` timestamps, so letting
the user enable one without the other would silently produce a report missing the
inference timeline. The `--no-gpu-profile-rollout-events` flag is the escape hatch.

### `plot_gpu_profile.py` (new)

Standalone CLI that lives at the repo root and consumes the JSONL files. Two stacked
panels share an x-axis:

1. **GPU utilization heatmap** (top): rows = GPUs, cells = 200 ms util %, RdYlGn colormap.
2. **Per-request timeline** (bottom): coloured bars per GPU lane, with grey "other (GPU busy)"
   fill for time covered by GPU utilization but not attributed to any known event.

Key functions:

- `load_profile_record` — picks the **first** record with both utilization samples and
  (request times OR `rollout_window_only`). Slime appends one such record every
  `plot_steps` rollouts.
- `load_and_merge_inference_profile` — concatenates per-(gpu, event_type) lists across
  all inference-profile rows, then takes `min(t0_sec)` to anchor the merged events on
  the earliest-seen request.
- `EVENT_CATEGORIES` — defines the (jsonl-key-suffix, label, colour) tuples. Iteration
  order matters: `train_step` is drawn last so its many short bars sit on top of the
  broader `log_probs` / `prefill` spans. Two categories share colours intentionally
  (log_probs ≡ prefill in blue; train_step ≡ decode in orange) — the actor and inference
  profiles are never drawn for the same GPU lane in practice.

Usage:

```bash
# With the defaults (looks for log/gpu_profile.jsonl):
python plot_gpu_profile.py

# Merging in the inference profile too:
python plot_gpu_profile.py \
    --input log/gpu_profile.jsonl \
    --inference-profile log/inference_profile.jsonl
```

### Examples

**`examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py`** — adds a `profile_args`
block (`--enable-gpu-profile --gpu-profile-plot-steps 1 --enable-inference-profile`),
switches `tensor-model-parallel-size` from 4 to 2, drops `--num-rollout` from 3000 to 1
(profiling demo), bumps `--rollout-max-response-len` to 20000, sets
`--sglang-attention-backend fa3`. Comments out a stale `--ckpt-step 499` line.

**`examples/geo3k_vlm_multi_turn/README.md`** — updates the run command to use `nohup …
&` with a timestamped log file. **WARNING**: the README change as committed includes
a hardcoded `WANDB_API_KEY`. **This is a credential leak — please rotate the key** and
remove it from the file before pushing the fork to a public remote.

## How to enable

```bash
# In your slime launch script:
python -m slime.training.main \
    ... \
    --enable-gpu-profile \
    --gpu-profile-plot-steps 30 \
    [--no-gpu-profile-rollout-events]   # only if you want to skip inference events
```

Output ends up in `log/` (or wherever `--gpu-profile-output-dir` / `GPU_PROFILE_DIR`
points). Render with `python plot_gpu_profile.py` after the run.

## Process model recap

- **Per-rank**: each `FSDPTrainRayActor` owns its own `Profiler`; data is collected
  locally and gathered via `dist.all_gather_object` every `plot_steps` rollouts.
- **Rank-0-only**: the `GpuUtilizationCollector` thread (one node-wide nvidia-smi
  reader is enough) and the report writer.
- **Per-request**: `write_inference_profile` is called inline on the rollout path with
  no cross-rank coordination. Each request appends one line.

## Notes / caveats

- The plotter assumes `log/gpu_profile.jsonl` is at the slime repo root or that
  `--input` is passed explicitly. Default `log_dir` resolution is
  `args.gpu_profile_output_dir` → `GPU_PROFILE_DIR` env → `"log"/`.
- `gpu_profile_heatmap_steps` is currently unused; it's reserved for a follow-up
  heatmap report type.
- The committed `log/gpu_profile.jsonl` and `log/gpu_profile.png` are sample artifacts
  from a development run, not consumed by code — treat them as documentation, not
  fixtures.
