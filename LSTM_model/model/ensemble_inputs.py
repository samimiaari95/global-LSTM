"""Stream prepared feature bundles using the ensemble's fitted scalers."""

import datetime as dt
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np

from LSTM_model.model.config import BATCH_SIZE, FEATURES_FILES, LOOKBACK, ensemble_member_paths

LOGGER = logging.getLogger("era5land_preprocessing")
DYNAMIC_VARIABLES = ("tp", "vpd", "swvl3")
DYNAMIC_FILES = (
    "tp_1995_2016.npy",
    "vpd_1995_2016.npy",
    "swvl3_1995_2016.npy",
)
SCALER_PREFIXES = ("TOT_PREC", "vpd", "soilmoisture")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _print_progress(message: str) -> None:
    print(f"[ERA5-Land preprocessing] {message}", flush=True)


def _feature_scaler_prefix(filename: str) -> str:
    if filename in DYNAMIC_FILES:
        return SCALER_PREFIXES[DYNAMIC_FILES.index(filename)]
    return Path(filename).stem


def load_feature_scalers(member: int) -> Tuple[np.ndarray, np.ndarray]:
    _, scaler_path = ensemble_member_paths(member)
    with scaler_path.open("rb") as scaler_file:
        scalers = pickle.load(scaler_file)

    means = np.empty(len(FEATURES_FILES), dtype=np.float32)
    standard_deviations = np.empty(len(FEATURES_FILES), dtype=np.float32)
    for index, filename in enumerate(FEATURES_FILES):
        prefix = _feature_scaler_prefix(filename)
        mean_key = f"{prefix}mean"
        std_key = f"{prefix}std"
        if mean_key not in scalers or std_key not in scalers:
            raise KeyError(
                f"{scaler_path} is missing scaler values {mean_key!r} "
                f"and/or {std_key!r}"
            )
        mean = np.asarray(scalers[mean_key])
        standard_deviation = np.asarray(scalers[std_key])
        if mean.size != 1 or standard_deviation.size != 1:
            raise ValueError(
                f"{scaler_path} must contain scalar means and standard "
                f"deviations for {filename}"
            )
        means[index] = np.float32(mean.item())
        standard_deviations[index] = np.float32(standard_deviation.item())
    if not np.isfinite(means).all():
        raise ValueError(f"{scaler_path} contains non-finite means")
    if (
        not np.isfinite(standard_deviations).all()
        or np.any(standard_deviations == 0)
    ):
        raise ValueError(f"{scaler_path} contains invalid standard deviations")
    return means, standard_deviations


def iter_member_input_batches(
    prepared_directory: Path,
    member: int,
    batch_size: int = BATCH_SIZE,
    rank: int = 0,
    world_size: int = 1,
    with_indices: bool = False,
) -> Iterator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Yield standardized chunks in global time-major pixel order."""
    if member < 0:
        raise ValueError("member must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}); got rank={rank}"
        )

    prepared_directory = Path(prepared_directory)
    metadata_path = prepared_directory / "preprocessing.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("feature_files") != list(FEATURES_FILES):
        raise ValueError("Prepared feature order does not match model config")
    if metadata.get("lookback_days") != LOOKBACK:
        raise ValueError("Prepared lookback does not match model config")

    dynamic = np.load(
        prepared_directory / "dynamic_features.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    static = np.load(
        prepared_directory / "static_features.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dates = np.load(
        prepared_directory / "context_dates.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected_pixels = np.load(
        prepared_directory / "selected_pixels.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if selected_pixels.ndim != 2 or selected_pixels.shape[1] != 2:
        raise ValueError(
            f"Selected pixels must have shape (n, 2); got {selected_pixels.shape}"
        )
    pixel_count = len(selected_pixels)
    if pixel_count == 0:
        raise ValueError("Prepared inputs contain no selected pixels")
    expected_dynamic_shape = (
        len(DYNAMIC_VARIABLES),
        len(dates),
        pixel_count,
    )
    expected_static_shape = (len(FEATURES_FILES) - len(DYNAMIC_FILES), pixel_count)
    if dynamic.shape != expected_dynamic_shape or static.shape != expected_static_shape:
        raise ValueError(
            f"Prepared feature shapes do not match metadata: "
            f"dynamic={dynamic.shape}, static={static.shape}"
        )
    if dates.ndim != 1 or len(dates) <= LOOKBACK:
        raise ValueError("Prepared dates do not contain a complete lookback window")

    means, standard_deviations = load_feature_scalers(member)
    windows = np.lib.stride_tricks.sliding_window_view(
        dynamic, LOOKBACK, axis=1
    )
    target_count = len(dates) - LOOKBACK
    sample_count = target_count * pixel_count
    rank_sample_count = (
        sample_count + world_size - 1 - rank
    ) // world_size
    batch_count = (rank_sample_count + batch_size - 1) // batch_size
    progress_interval = max(1, batch_count // 20)
    batches_started = time.monotonic()
    _print_progress(
        f"streaming member {member}, rank {rank}/{world_size}: "
        f"{rank_sample_count:,} rank-local samples in "
        f"{batch_count:,} batches of at most {batch_size:,}; "
        f"each full float32 batch is about "
        f"{batch_size * LOOKBACK * len(FEATURES_FILES) * 4 / (1024**2):.1f} MiB"
    )
    for local_start in range(0, rank_sample_count, batch_size):
        local_end = min(local_start + batch_size, rank_sample_count)
        sample_indices = (
            rank + np.arange(local_start, local_end, dtype=np.int64) * world_size
        )
        feature_block = np.empty(
            (len(sample_indices), LOOKBACK, len(FEATURES_FILES)),
            dtype=np.float32,
        )
        time_indices, pixel_indices = np.divmod(sample_indices, pixel_count)
        feature_block[:, :, : len(DYNAMIC_VARIABLES)] = np.stack(
            [
                windows[
                    feature_index, time_indices, pixel_indices, :
                ]
                for feature_index in range(len(DYNAMIC_VARIABLES))
            ],
            axis=-1,
        )
        for static_index in range(static.shape[0]):
            feature_block[
                :, :, len(DYNAMIC_VARIABLES) + static_index
            ] = static[static_index, pixel_indices][:, None]
        feature_block -= means[None, None, :]
        feature_block /= standard_deviations[None, None, :]
        batch_index = local_start // batch_size + 1
        if (
            batch_index == 1
            or batch_index % progress_interval == 0
            or batch_index == batch_count
        ):
            elapsed = time.monotonic() - batches_started
            estimated_total = elapsed / batch_index * batch_count
            batch_end = min(local_start + len(sample_indices), rank_sample_count)
            _print_progress(
                f"member {member}, rank {rank}/{world_size}: input batch "
                f"{batch_index}/{batch_count} "
                f"({batch_end:,}/{rank_sample_count:,} rank-local samples, "
                f"{batch_end / rank_sample_count:.1%}); elapsed "
                f"{_format_duration(elapsed)}, ETA about "
                f"{_format_duration(estimated_total - elapsed)}"
            )
        if with_indices:
            yield sample_indices, feature_block
        else:
            yield feature_block
