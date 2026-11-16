"""Prepare masked ERA5-Land inputs and stream member-scaled LSTM batches."""

import calendar
import datetime as dt
import json
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Iterator, Sequence, Tuple, Union

import numpy as np

from LSTM_model.model.config import (
    BATCH_SIZE,
    FEATURES_FILES,
    INPUTPATH,
    LOOKBACK,
    MODEL_NAME,
    OUTPUTPATH,
    get_root_dir,
)


LOGGER = logging.getLogger("era5land_preprocessing")
FIRST_CONTEXT_YEAR = 1995
FIRST_TARGET_YEAR = 1996
LAST_TARGET_YEAR = 2012
OBSERVATION_FILENAME = (
    "grid_measured_wtd_a_data_1996_2012_EUR11_YM_v_20210407_"
    "with_station_anomalies.nc"
)
DYNAMIC_VARIABLES = ("tp", "vpd", "swvl3")
DYNAMIC_FILES = (
    "tp_1995_2016.npy",
    "vpd_1995_2016.npy",
    "swvl3_1995_2016.npy",
)
SCALER_PREFIXES = ("TOT_PREC", "vpd", "soilmoisture")
OBSERVATION_VARIABLE = "wtd_a"
ENSEMBLE_DIRECTORY = Path(OUTPUTPATH) / "validation_ERA5" / "ensemble_400px"
ERA5_DIRECTORY = Path(INPUTPATH) / "raw" / "ERA5-Land"
INPUT_DIRECTORY = Path(INPUTPATH)
OBSERVATION_FILE = Path(get_root_dir()).parent / OBSERVATION_FILENAME
OUTPUT_DIRECTORY = Path(INPUTPATH) / "era5land_inputs_1996_2012"
PIXEL_BATCH_SIZE = 256


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


def _all_dates(start_year: int, end_year: int) -> Sequence[dt.date]:
    return tuple(
        dt.date(year, month, day)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if not (month == 2 and day == 29)
    )


def _read_valid_pixels(
    observation_path: Path, grid_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    try:
        import netCDF4
    except ImportError as error:
        raise RuntimeError(
            "netCDF4 is required to read the water-table observation file"
        ) from error

    with netCDF4.Dataset(observation_path) as dataset:
        if OBSERVATION_VARIABLE not in dataset.variables:
            raise KeyError(
                f"{observation_path} does not contain "
                f"{OBSERVATION_VARIABLE!r}"
            )
        observations = dataset.variables[OBSERVATION_VARIABLE]
        if observations.ndim != 3:
            raise ValueError(
                f"{OBSERVATION_VARIABLE} must have time, y, x dimensions; "
                f"got {observations.shape}"
            )
        observation_shape = tuple(observations.shape[1:])
        if "lat" not in dataset.variables or "lon" not in dataset.variables:
            raise KeyError(f"{observation_path} must contain lat and lon variables")
        if dataset.variables["lat"].shape != (observation_shape[0],):
            raise ValueError("Observation latitude dimension does not match wtd_a")
        if dataset.variables["lon"].shape != (observation_shape[1],):
            raise ValueError("Observation longitude dimension does not match wtd_a")
        if any(
            grid_size < observation_size
            or (grid_size - observation_size) % 2
            for grid_size, observation_size in zip(grid_shape, observation_shape)
        ):
            raise ValueError(
                f"Cannot center observation grid {observation_shape} in "
                f"model grid {grid_shape}"
            )

        valid_observations = np.zeros(observation_shape, dtype=bool)
        time_chunk_size = 12
        chunk_count = (
            observations.shape[0] + time_chunk_size - 1
        ) // time_chunk_size
        for chunk_index, start in enumerate(
            range(0, observations.shape[0], time_chunk_size), start=1
        ):
            block = observations[start : start + time_chunk_size]
            block = np.ma.filled(block, np.nan)
            valid_observations |= np.isfinite(block).any(axis=0)
            _print_progress(
                f"scanned observation time blocks {chunk_index}/{chunk_count}"
            )

    valid_grid = np.zeros(grid_shape, dtype=bool)
    row_offset = (grid_shape[0] - observation_shape[0]) // 2
    col_offset = (grid_shape[1] - observation_shape[1]) // 2
    # The observation grid is centered within the larger model grid.
    valid_grid[
        row_offset : row_offset + observation_shape[0],
        col_offset : col_offset + observation_shape[1],
    ] = valid_observations
    y_indices, x_indices = np.where(valid_grid)
    if y_indices.size == 0:
        raise ValueError(
            f"No finite {OBSERVATION_VARIABLE} observations in {observation_path}"
        )
    _print_progress(
        f"selected {y_indices.size:,} valid pixels from grid {grid_shape}; "
        f"observation-grid offset is rows={row_offset}, columns={col_offset}"
    )
    return y_indices, x_indices, (row_offset, col_offset)


def _load_static_values(
    input_dir: Path,
    filename: str,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    path = input_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required static predictor is missing: {path}")

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    cell_count = int(np.prod(grid_shape))
    if data.ndim == 1 and data.size == cell_count:
        selected = data[y_indices * grid_shape[1] + x_indices]
    elif data.ndim == 2 and tuple(data.shape) == grid_shape:
        selected = data[y_indices, x_indices]
    elif data.ndim > 2 and tuple(data.shape[-2:]) == grid_shape:
        selected = data[0, y_indices, x_indices]
    else:
        raise ValueError(
            f"{path} does not match model grid {grid_shape}; got {data.shape}"
        )
    return np.nan_to_num(
        np.asarray(selected, dtype=np.float32),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _load_dynamic_month(
    era5_directory: Path,
    variable: str,
    year: int,
    month: int,
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    path = (
        era5_directory
        / "npy_regridded_to_EUR11"
        / f"{variable}_{year}{month:02d}_EU.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Required ERA5-Land input is missing: {path}")

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 3 or tuple(data.shape[1:]) != grid_shape:
        raise ValueError(
            f"{path} must have shape (days, {grid_shape[0]}, {grid_shape[1]}); "
            f"got {data.shape}"
        )
    month_dates = [
        dt.date(year, month, day)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]
    retained_indices = np.asarray(
        [
            index
            for index, date in enumerate(month_dates)
            if not (date.month == 2 and date.day == 29)
        ],
        dtype=np.intp,
    )
    if data.shape[0] not in {len(month_dates), len(retained_indices)}:
        raise ValueError(
            f"{path} contains {data.shape[0]} days; expected "
            f"{len(month_dates)} or {len(retained_indices)}"
        )
    if data.shape[0] == len(retained_indices) and len(retained_indices) != len(
        month_dates
    ):
        retained_indices = np.arange(data.shape[0], dtype=np.intp)

    return data, retained_indices


def _feature_scaler_prefix(filename: str) -> str:
    if filename in DYNAMIC_FILES:
        return SCALER_PREFIXES[DYNAMIC_FILES.index(filename)]
    return Path(filename).stem


def _load_member_scalers(member: int) -> Tuple[np.ndarray, np.ndarray]:
    member_directory = ENSEMBLE_DIRECTORY / f"400px_member_{member}"
    scaler_path = (
        member_directory
        / f"meanstd_400px_member_{member}_{MODEL_NAME}.pkl"
    )
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


def prepare(
    era5_directory: Path,
    input_dir: Path,
    observation_path: Path,
    output_dir: Path,
    pixel_batch_size: int,
) -> None:
    if len(FEATURES_FILES) < len(DYNAMIC_FILES):
        raise ValueError("config.FEATURES_FILES is missing dynamic predictors")
    if tuple(FEATURES_FILES[:3]) != DYNAMIC_FILES:
        raise ValueError(
            "Expected config.FEATURES_FILES to begin with "
            f"{DYNAMIC_FILES}, got {tuple(FEATURES_FILES[:3])}"
        )
    if pixel_batch_size < 1:
        raise ValueError("pixel_batch_size must be positive")

    probe_path = (
        era5_directory / "npy_regridded_to_EUR11" / "tp_199501_EU.npy"
    )
    if not probe_path.is_file():
        raise FileNotFoundError(f"Required ERA5-Land input is missing: {probe_path}")
    probe = np.load(probe_path, mmap_mode="r", allow_pickle=False)
    if probe.ndim != 3:
        raise ValueError(f"{probe_path} must be a 3D daily grid; got {probe.shape}")
    grid_shape = tuple(probe.shape[1:])
    del probe

    y_indices, x_indices, observation_offset = _read_valid_pixels(
        observation_path, grid_shape
    )
    dates = _all_dates(FIRST_CONTEXT_YEAR, LAST_TARGET_YEAR)
    target_dates = dates[LOOKBACK:]
    expected_target_count = (LAST_TARGET_YEAR - FIRST_TARGET_YEAR + 1) * 365
    if (
        target_dates[0] != dt.date(FIRST_TARGET_YEAR, 1, 1)
        or target_dates[-1] != dt.date(LAST_TARGET_YEAR, 12, 31)
        or len(target_dates) != expected_target_count
    ):
        raise ValueError("Date sequence does not align with target years")
    _print_progress(
        f"date window: {dates[0]} to {dates[-1]} ({len(dates):,} days, "
        f"Feb 29 excluded); model samples cover {target_dates[0]} to "
        f"{target_dates[-1]} ({len(target_dates):,} days)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_path = output_dir / "dynamic_features.npy"
    static_path = output_dir / "static_features.npy"
    metadata_path = output_dir / "preprocessing.json"
    pixels_path = output_dir / "selected_pixels.npy"
    dates_path = output_dir / "context_dates.npy"
    target_dates_path = output_dir / "target_dates.npy"
    output_paths = (
        dynamic_path,
        static_path,
        metadata_path,
        pixels_path,
        dates_path,
        target_dates_path,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite prepared inputs: "
            + ", ".join(str(path) for path in existing)
        )

    pixel_count = len(y_indices)
    dynamic_bytes = (
        len(DYNAMIC_VARIABLES)
        * len(dates)
        * pixel_count
        * np.dtype(np.float32).itemsize
    )
    static_count = len(FEATURES_FILES) - len(DYNAMIC_FILES)
    static_bytes = (
        static_count * pixel_count * np.dtype(np.float32).itemsize
    )
    required_bytes = dynamic_bytes + static_bytes
    available_bytes = shutil.disk_usage(output_dir).free
    _print_progress(
        f"grid={grid_shape}, valid_pixels={pixel_count:,}, "
        f"features={len(FEATURES_FILES)} ({len(DYNAMIC_VARIABLES)} daily, "
        f"{static_count} static), lookback={LOOKBACK} days"
    )
    _print_progress(
        f"staging requires {required_bytes / (1024**3):.2f} GiB "
        f"({dynamic_bytes / (1024**3):.2f} GiB daily + "
        f"{static_bytes / (1024**3):.2f} GiB static); "
        f"{available_bytes / (1024**3):.2f} GiB free in {output_dir}"
    )
    LOGGER.info(
        "Preparing %d context days and %d target days for %d/%d grid cells; "
        "staging requires %.2f GiB (%.2f GiB free)",
        len(dates),
        len(target_dates),
        pixel_count,
        int(np.prod(grid_shape)),
        required_bytes / (1024**3),
        available_bytes / (1024**3),
    )
    if available_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space in {output_dir}: need at least "
            f"{required_bytes / (1024**3):.2f} GiB, have "
            f"{available_bytes / (1024**3):.2f} GiB"
        )

    dynamic_features = np.lib.format.open_memmap(
        dynamic_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(DYNAMIC_VARIABLES), len(dates), pixel_count),
    )
    date_offsets = {date: index for index, date in enumerate(dates)}
    total_months = (
        len(DYNAMIC_VARIABLES) * (LAST_TARGET_YEAR - FIRST_CONTEXT_YEAR + 1) * 12
    )
    completed_months = 0
    staging_started = time.monotonic()
    for feature_index, variable in enumerate(DYNAMIC_VARIABLES):
        for year in range(FIRST_CONTEXT_YEAR, LAST_TARGET_YEAR + 1):
            for month in range(1, 13):
                month_dates = [
                    dt.date(year, month, day)
                    for day in range(
                        1, calendar.monthrange(year, month)[1] + 1
                    )
                ]
                retained_dates = [
                    date
                    for date in month_dates
                    if not (date.month == 2 and date.day == 29)
                ]
                start = date_offsets[retained_dates[0]]
                data, retained_indices = _load_dynamic_month(
                    era5_directory,
                    variable,
                    year,
                    month,
                    grid_shape,
                )
                if len(retained_indices) != len(retained_dates):
                    raise ValueError(
                        f"Unexpected retained-day count for "
                        f"{variable} {year}-{month:02d}"
                    )
                elapsed = time.monotonic() - staging_started
                eta = (
                    _format_duration(
                        elapsed
                        / completed_months
                        * (total_months - completed_months)
                    )
                    if completed_months
                    else "estimating after first month"
                )
                _print_progress(
                    f"staging month {completed_months + 1}/{total_months} "
                    f"({(completed_months + 1) / total_months:.1%}): "
                    f"{variable} {year}-{month:02d}, "
                    f"{len(retained_dates)} days, {pixel_count:,} pixels; "
                    f"pixel batch size={pixel_batch_size}, "
                    f"elapsed {_format_duration(elapsed)}, ETA {eta}"
                )
                pixel_starts = range(0, pixel_count, pixel_batch_size)
                pixel_batch_count = (
                    pixel_count + pixel_batch_size - 1
                ) // pixel_batch_size
                for pixel_batch_index, pixel_start in enumerate(
                    pixel_starts, start=1
                ):
                    pixel_end = min(pixel_start + pixel_batch_size, pixel_count)
                    selected = data[
                        retained_indices[:, None],
                        y_indices[None, pixel_start:pixel_end],
                        x_indices[None, pixel_start:pixel_end],
                    ]
                    selected = np.nan_to_num(
                        np.asarray(selected, dtype=np.float32),
                        copy=False,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    dynamic_features[
                        feature_index,
                        start : start + len(retained_dates),
                        pixel_start:pixel_end,
                    ] = selected
                    if (
                        pixel_batch_count > 64
                        and pixel_batch_index == (pixel_batch_count + 1) // 2
                    ):
                        _print_progress(
                            f"  halfway through {variable} {year}-{month:02d}: "
                            f"{pixel_end:,}/{pixel_count:,} pixels staged"
                        )
                del data
                completed_months += 1
        dynamic_features.flush()
    del dynamic_features

    _print_progress("daily predictors staged; starting configured static predictors")
    static_features = np.lib.format.open_memmap(
        static_path,
        mode="w+",
        dtype=np.float32,
        shape=(static_count, pixel_count),
    )
    for feature_index, filename in enumerate(FEATURES_FILES[3:]):
        _print_progress(
            f"staging static predictor {feature_index + 1}/{static_count}: "
            f"{filename}"
        )
        for pixel_start in range(0, pixel_count, pixel_batch_size):
            pixel_end = min(pixel_start + pixel_batch_size, pixel_count)
            static_features[feature_index, pixel_start:pixel_end] = (
                _load_static_values(
                    input_dir,
                    filename,
                    y_indices[pixel_start:pixel_end],
                    x_indices[pixel_start:pixel_end],
                    grid_shape,
                )
            )
    static_features.flush()
    del static_features

    np.save(
        pixels_path,
        np.column_stack((y_indices, x_indices)).astype(np.int32, copy=False),
    )
    np.save(dates_path, np.asarray(dates, dtype="datetime64[D]"))
    np.save(target_dates_path, np.asarray(target_dates, dtype="datetime64[D]"))
    _print_progress(
        f"wrote coordinates and dates; preparing metadata in {output_dir}"
    )
    metadata = {
        "observation_file": str(observation_path.resolve()),
        "observation_variable": OBSERVATION_VARIABLE,
        "era5_directory": str(era5_directory.resolve()),
        "input_directory": str(input_dir.resolve()),
        "feature_files": list(FEATURES_FILES),
        "dynamic_variables": list(DYNAMIC_VARIABLES),
        "dynamic_feature_path": str(dynamic_path.resolve()),
        "static_feature_path": str(static_path.resolve()),
        "selected_pixels_path": str(pixels_path.resolve()),
        "context_dates_path": str(dates_path.resolve()),
        "target_dates_path": str(target_dates_path.resolve()),
        "grid_shape": list(grid_shape),
        "observation_grid_center_offset": list(observation_offset),
        "selected_pixel_count": pixel_count,
        "context_start_date": dates[0].isoformat(),
        "context_end_date": dates[-1].isoformat(),
        "target_start_date": target_dates[0].isoformat(),
        "target_end_date": target_dates[-1].isoformat(),
        "excluded_calendar_dates": ["02-29"],
        "lookback_days": LOOKBACK,
        "dtype": "float32",
        "feature_layout": "time_major_pixel_minor_lookback_feature",
        "input_shape_per_member": [
            len(target_dates) * pixel_count,
            LOOKBACK,
            len(FEATURES_FILES),
        ],
        "staged_feature_shapes": {
            "dynamic": [len(DYNAMIC_VARIABLES), len(dates), pixel_count],
            "static": [static_count, pixel_count],
        },
        "pixel_batch_size": pixel_batch_size,
        "member_scaling": (
            "Apply the requested member's saved training means and standard "
            "deviations using iter_member_input_batches."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    _print_progress(
        f"preprocessing complete: {len(target_dates) * pixel_count:,} logical "
        f"samples per member; staged inputs are in {output_dir}"
    )
    LOGGER.info("Preprocessing complete; staged inputs at %s", output_dir)


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

    means, standard_deviations = _load_member_scalers(member)
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _print_progress("starting input validation and preprocessing")
    _print_progress(f"ERA5-Land source: {ERA5_DIRECTORY}")
    _print_progress(f"observation mask: {OBSERVATION_FILE}")
    _print_progress(f"output directory: {OUTPUT_DIRECTORY}")
    prepare(
        ERA5_DIRECTORY,
        INPUT_DIRECTORY,
        OBSERVATION_FILE,
        OUTPUT_DIRECTORY,
        PIXEL_BATCH_SIZE,
    )
