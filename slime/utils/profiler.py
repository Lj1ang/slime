"""
gpu-profile: collect latencies per (step, worker). request-profile: request count per GPU.
Per-request specific times (start_sec, end_sec) are recorded per step; latencies = end - start.
GPU utilization is sampled every 200 ms on a background thread (when enabled).

Relationship between rank and worker:
- Rank = distributed process rank (e.g. torch.distributed.get_rank()). One process per GPU in FSDP.
- Worker = unit of work / GPU in the profiler. In FSDP, worker_id is set to dp_rank, so one worker
  per rank (1:1). In other backends, worker could map to a different dimension (e.g. rollout engine id).
"""

import json
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GPU_UTIL_INTERVAL_SEC = 0.2  # 200 ms

# Folder for local profiler log files (JSON lines, one object per report)
PROFILER_LOG_DIR = "log"
PROFILER_LOG_FILENAME = "gpu_profile.jsonl"

# (start_time_sec, end_time_sec) wall-clock per request
RequestTime = tuple[float, float]

# (start_time_sec, end_time_sec, event_type) for categorized events
EventTime = tuple[float, float, str]

# Event types for timeline (train_step = one microbatch forward+backward; others = phase spans)
EVENT_TRAIN_STEP = "train_step"
EVENT_LOG_PROBS = "log_probs"
EVENT_REF_LOG_PROBS = "ref_log_probs"

# Inference (SGLang) event types for prefill/decode/unified timeline
INFERENCE_EVENT_PREFILL = "prefill"
INFERENCE_EVENT_DECODE = "decode"
INFERENCE_EVENT_UNIFIED = "unified"
INFERENCE_PROFILER_LOG_FILENAME = "inference_profile.jsonl"


class Profiler:
    """
    Records per-request (start_time, end_time, event_type) per (step, worker) for gpu-profile;
    tracks request count for request-profile (train_step only).
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        # (step, worker_id) -> list of (start_sec, end_sec, event_type)
        self._samples: dict[tuple[int, int], list[EventTime]] = defaultdict(list)
        # Number of train_step records (request-profile: requests bound to this GPU)
        self._request_count: int = 0

    def record(
        self,
        step: int,
        worker_id: int,
        start_time_sec: float,
        end_time_sec: float,
        event_type: str = EVENT_TRAIN_STEP,
    ) -> None:
        self._samples[(step, worker_id)].append((start_time_sec, end_time_sec, event_type))
        if event_type == EVENT_TRAIN_STEP:
            self._request_count += 1

    def merge_from_rank(
        self,
        rank_data: dict[tuple[int, int], list[EventTime]],
        request_count: int = 0,
    ) -> None:
        """Merge samples and request count from another rank (for distributed gather)."""
        for key, vals in rank_data.items():
            self._samples[key].extend(vals)
        self._request_count += request_count

    def get_local_data_for_gather(
        self,
    ) -> tuple[dict[tuple[int, int], list[EventTime]], int]:
        """Return (samples with (start, end, event_type), request_count) for all_gather_object."""
        return dict(self._samples), self._request_count

    def clear(self) -> None:
        self._samples.clear()
        self._request_count = 0


class GpuUtilizationCollector:
    """
    Background thread that samples GPU utilization (percent) every 200 ms for all GPUs
    via the nvidia-smi Linux command. Call start() to begin, get_and_clear() to consume
    samples (e.g. at report time).
    """

    def __init__(self) -> None:
        self._samples: list[tuple[float, list[int]]] = []  # (timestamp_sec, [util_pct per gpu])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return
            # One integer per line, one line per GPU
            utils = [int(line.strip()) for line in result.stdout.strip().splitlines()]
            with self._lock:
                self._samples.append((time.time(), utils))
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
            logger.debug("GpuUtilizationCollector sample failed: %s", e)

    def _run(self) -> None:
        while not self._stop.wait(GPU_UTIL_INTERVAL_SEC):
            self._sample_once()

    def start(self) -> None:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("nvidia-smi not available, GPU utilization profiling disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_and_clear(self) -> list[tuple[float, list[int]]]:
        """Return list of (timestamp_sec, [util_pct per gpu]) and clear the buffer."""
        with self._lock:
            out = self._samples
            self._samples = []
        return out


def _request_times_to_latencies(times: list[RequestTime]) -> list[float]:
    """Convert (start, end) list to duration list in seconds."""
    return [end - start for start, end in times]


def log_request_profile_metrics(
    args: Any,
    step_key_value: int,
    request_count_per_rank: list[int] | None = None,
    latency_per_rank: list[list[float]] | None = None,
    request_times_per_rank: list[list[RequestTime]] | None = None,
    gpu_utilization_samples: list[tuple[float, list[int]]] | None = None,
    event_times_per_rank_by_type: dict[str, list[list[RequestTime]]] | None = None,
) -> None:
    """Log request-profile metrics: per-request (start_time, end_time) per GPU, request counts, optional GPU utilization, and optional per-event-type times (e.g. log_probs, ref_log_probs)."""
    from slime.utils import logging_utils

    step_key = "rollout/step"
    log_dict: dict[str, Any] = {
        "gpu_profile/interval_ms": 200,
        step_key: step_key_value,
    }
    if request_count_per_rank is not None:
        for rank, count in enumerate(request_count_per_rank):
            log_dict[f"request_profile/requests_gpu{rank}"] = count
        log_dict["request_profile/total_requests"] = sum(request_count_per_rank)
    if request_times_per_rank is not None:
        for rank, times in enumerate(request_times_per_rank):
            log_dict[f"gpu_profile/gpu{rank}_request_times"] = [[start_sec, end_sec] for start_sec, end_sec in times]
            if times:
                starts = [s for s, _ in times]
                ends = [e for _, e in times]
                log_dict[f"gpu_profile/gpu{rank}_first_request_start_sec"] = min(starts)
                log_dict[f"gpu_profile/gpu{rank}_last_request_end_sec"] = max(ends)
    if event_times_per_rank_by_type is not None:
        for event_type, times_per_rank in event_times_per_rank_by_type.items():
            key_suffix = f"{event_type}_times"
            for rank, times in enumerate(times_per_rank):
                if times:
                    log_dict[f"gpu_profile/gpu{rank}_{key_suffix}"] = [[s, e] for s, e in times]
    if gpu_utilization_samples is not None and gpu_utilization_samples:
        # Each entry: (timestamp_sec, [util_pct for gpu0, gpu1, ...])
        log_dict["gpu_profile/gpu_utilization_samples"] = [[t, utils] for t, utils in gpu_utilization_samples]
        log_dict["gpu_profile/gpu_utilization_start_sec"] = gpu_utilization_samples[0][0]
        log_dict["gpu_profile/gpu_utilization_end_sec"] = gpu_utilization_samples[-1][0]

    logging_utils.log(args, log_dict, step_key=step_key)

    # Save all profiler log data locally under log/
    log_dir = Path(getattr(args, "gpu_profile_output_dir", None) or os.environ.get("GPU_PROFILE_DIR", PROFILER_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / PROFILER_LOG_FILENAME
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_dict) + "\n")
    except OSError as e:
        logger.warning("Failed to write profiler log to %s: %s", log_file, e)


def run_gpu_profile_report(
    profiler: Profiler | None,
    args: Any,
    rollout_id: int,
    is_primary_rank: bool,
    step_key_value: int,
    heatmap_max_steps: int = 20,
    request_count_per_rank: list[int] | None = None,
    latency_per_rank: list[list[float]] | None = None,
    request_times_per_rank: list[list[RequestTime]] | None = None,
    gpu_utilization_samples: list[tuple[float, list[int]]] | None = None,
    event_times_per_rank_by_type: dict[str, list[list[RequestTime]]] | None = None,
) -> None:
    """
    Log gpu-profile and request-profile metrics at report cadence.
    Call from rank 0. Per-request (start_sec, end_sec) are recorded per step; GPU utilization sampled every 200 ms.
    event_times_per_rank_by_type: optional dict of event_type -> list of (start, end) per rank for extra categories (e.g. log_probs, ref_log_probs).
    """
    log_request_profile_metrics(
        args,
        step_key_value,
        request_count_per_rank,
        latency_per_rank,
        request_times_per_rank,
        gpu_utilization_samples,
        event_times_per_rank_by_type=event_times_per_rank_by_type,
    )


def write_inference_profile(
    event_times_per_rank_by_type: dict[str, list[list[RequestTime]]],
    step_key_value: int,
    log_dir: str | Path | None = None,
    t0_sec: float | None = None,
) -> None:
    """
    Write one line to inference_profile.jsonl for SGLang inference events (prefill, decode, unified).

    Call this from the rollout path when SGLang returns per-phase timings in the response.
    event_times_per_rank_by_type: dict with keys "prefill", "decode", "unified" (or subset),
        each value = list of (start_sec, end_sec) per GPU rank.
    step_key_value: rollout/step value for alignment with actor profile.
    log_dir: directory for inference_profile.jsonl (default: PROFILER_LOG_DIR).
    t0_sec: optional reference time for relative-time alignment with actor profile.
    """
    log_dir = Path(log_dir or PROFILER_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dict: dict[str, Any] = {
        "gpu_profile/interval_ms": 200,
        "rollout/step": step_key_value,
        "gpu_profile/inference_profile": True,
    }
    if t0_sec is not None:
        log_dict["gpu_profile/inference_t0_sec"] = t0_sec
    for event_type, times_per_rank in event_times_per_rank_by_type.items():
        key_suffix = f"{event_type}_times"
        for rank, times in enumerate(times_per_rank):
            if times:
                log_dict[f"gpu_profile/gpu{rank}_{key_suffix}"] = [[s, e] for s, e in times]
    log_file = log_dir / INFERENCE_PROFILER_LOG_FILENAME
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_dict) + "\n")
    except OSError as e:
        logger.warning("Failed to write inference profile to %s: %s", log_file, e)
