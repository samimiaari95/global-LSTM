"""Predict daily WTD at the prepared valid observation-grid cells."""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from LSTM_model.model.ensemble_inputs import iter_member_input_batches
from LSTM_model.model.config import (
    BATCH_SIZE,
    DROPOUT,
    FEATURES_FILES,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LOOKBACK,
    ENSEMBLE_PREDICTION_DIRECTORY,
    INPUTPATH,
    NUM_LAYERS,
    OUTPUT_SIZE,
    OUTPUTPATH,
    AwesomeLSTM,
    ensemble_member_paths,
)


LOGGER = logging.getLogger("era5land_observation_prediction")
FIRST_TARGET_DATE = np.datetime64("1996-01-01", "D")
LAST_TARGET_DATE = np.datetime64("2012-12-31", "D")
TARGET_DAY_COUNT = 17 * 365
EXPECTED_WORLD_SIZE = 4
DEFAULT_PREPARED_DIRECTORY = Path(INPUTPATH) / "era5land_inputs_1996_2012"
OUTPUT_FILENAME = "sim_ERA5on_observed_valid_pixels_1996_2012.npy"
NATIVE_EUROPE_GRID = "native ERA5-Land 0.1 degree"
NATIVE_EUROPE_OUTPUT = Path(OUTPUTPATH) / "era5land_arco_1996_2012" / "predictions"


def log_progress(message: str, rank: int) -> None:
    tagged_message = f"member={os.environ.get('ENS_MEMBER', '?')} rank={rank}: {message}"
    LOGGER.info(tagged_message)


def initialize_distributed(
    expected_world_size: int,
) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} DDP processes, but launcher started "
            f"{world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for daily WTD prediction")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Local rank {local_rank} has no visible GPU "
            f"(visible GPU count: {torch.cuda.device_count()})"
        )

    torch.cuda.set_device(local_rank)
    print(
        f"member={os.environ.get('ENS_MEMBER', '?')} rank={rank}/{world_size} "
        f"local_rank={local_rank}: initializing NCCL on "
        f"{torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def load_target_scalers(member: int) -> Tuple[float, float, Path]:
    _, scaler_path = ensemble_member_paths(member)
    with scaler_path.open("rb") as scaler_file:
        scalers: Dict[str, Any] = pickle.load(scaler_file)

    values = []
    for key in ("wtdmean", "wtdstd"):
        if key not in scalers:
            raise KeyError(f"{scaler_path} is missing {key!r}")
        value = np.asarray(scalers[key], dtype=np.float32)
        if value.size != 1:
            raise ValueError(f"{scaler_path} must contain a scalar {key}")
        values.append(float(value.item()))
    mean, standard_deviation = values
    if not np.isfinite(mean):
        raise ValueError(f"{scaler_path} contains a non-finite wtdmean")
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError(f"{scaler_path} contains an invalid wtdstd")
    return mean, standard_deviation, scaler_path


def validate_prepared_inputs(
    prepared_directory: Path, conus: bool = False,
) -> Tuple[int, int]:
    with (prepared_directory / "preprocessing.json").open(
        encoding="utf-8"
    ) as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("feature_files") != list(FEATURES_FILES):
        raise ValueError("Prepared feature order does not match model config")
    if metadata.get("lookback_days") != LOOKBACK:
        raise ValueError("Prepared lookback does not match model config")

    dates = np.load(
        prepared_directory / "target_dates.npy", mmap_mode="r", allow_pickle=False
    )
    pixels = np.load(
        prepared_directory / "selected_pixels.npy", mmap_mode="r", allow_pickle=False
    )
    native_europe = not conus and metadata.get("grid") == NATIVE_EUROPE_GRID
    if conus and metadata.get("region") != "CONUS":
        raise ValueError("Expected CONUS preparation metadata")
    if conus or native_europe:
        context = np.load(prepared_directory / "context_dates.npy", allow_pickle=False)
        if (dates.ndim != 1 or not dates.size or dates.dtype.kind != "M"
                or context.ndim != 1 or context.dtype.kind != "M"
                or len(context) != LOOKBACK + len(dates) or np.isnat(context).any()
                or not np.array_equal(context[LOOKBACK:], dates)):
            raise ValueError("Target dates must follow a complete lookback")
        expected = np.arange(context[0], context[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
        expected = expected[~np.char.endswith(expected.astype(str), "-02-29")]
        if not np.array_equal(context, expected):
            raise ValueError("Context dates must form one continuous no-leap calendar")
        if (str(dates[0]) != metadata.get("target_start_date")
                or str(dates[-1]) != metadata.get("target_end_date")):
            raise ValueError("Target dates disagree with preparation metadata")
    if not conus and dates.shape != (TARGET_DAY_COUNT,):
        raise ValueError(
            f"Expected {TARGET_DAY_COUNT} target dates for 1996-2012; got "
            f"{dates.shape}"
        )
    if not conus and (dates[0] != FIRST_TARGET_DATE or dates[-1] != LAST_TARGET_DATE):
        raise ValueError(
            f"Target dates must span 1996-01-01 through 2012-12-31; got "
            f"{dates[0]} through {dates[-1]}"
        )
    date_strings = dates.astype("datetime64[D]").astype(str)
    if np.any(np.char.endswith(date_strings, "-02-29")):
        raise ValueError("Prepared target dates must exclude February 29")
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) == 0:
        raise ValueError(
            "Prepared selected_pixels.npy must contain non-empty (y, x) pairs"
        )
    if conus or native_europe:
        if len(pixels) != metadata.get("selected_pixel_count") or len(np.unique(pixels, axis=0)) != len(pixels):
            raise ValueError("Pixel IDs must be unique and match the prepared cell count")
    if native_europe:
        shape = np.asarray(metadata["grid_shape"])
        if (shape.shape != (2,) or pixels.dtype.kind not in "iu"
                or np.any(pixels < 0) or np.any(pixels >= shape)):
            raise ValueError("Pixel IDs must be integer indices inside the ERA5-Land grid")
    return len(dates), len(pixels)


def create_prediction_file(
    temporary_path: Path,
    prediction_path: Path,
    target_day_count: int,
    pixel_count: int,
    rank: int,
) -> None:
    status = [None]
    log_progress(f"preparing output array {temporary_path}", rank)
    if rank == 0:
        try:
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            if prediction_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite predictions: {prediction_path}"
                )
            if prediction_path.with_suffix(".json").exists():
                raise FileExistsError(
                    "Refusing to overwrite prediction metadata: "
                    f"{prediction_path.with_suffix('.json')}"
                )
            if temporary_path.exists():
                raise FileExistsError(
                    f"An incomplete prediction file already exists: {temporary_path}"
                )
            output = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(target_day_count, pixel_count),
            )
            output[:] = np.nan
            output.flush()
            del output
        except Exception as error:
            LOGGER.exception("Could not initialize the shared prediction file")
            status[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Prediction-file initialization failed: {status[0]}")
    torch.distributed.barrier()
    log_progress("shared output array is ready", rank)


def finalize_prediction(
    temporary_path: Path,
    prediction_path: Path,
    metadata: Dict[str, Any],
    rank: int,
) -> None:
    status = [None]
    log_progress("finalizing prediction array and metadata", rank)
    if rank == 0:
        metadata_path = prediction_path.with_suffix(".json")
        staged_directory = temporary_path.parent != prediction_path.parent
        temporary_metadata_path = (
            temporary_path.parent / metadata_path.name if staged_directory
            else metadata_path.with_suffix(".json.partial")
        )
        try:
            with temporary_metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
            if staged_directory:
                # Publish the native-grid array and metadata together.
                if prediction_path.parent.exists():
                    raise FileExistsError(prediction_path.parent)
                temporary_path.parent.rename(prediction_path.parent)
            else:
                os.replace(temporary_path, prediction_path)
                os.replace(temporary_metadata_path, metadata_path)
            LOGGER.info("Saved daily predictions to %s", prediction_path)
        except Exception as error:
            LOGGER.exception("Could not finalize the prediction outputs")
            status[0] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Prediction finalization failed: {status[0]}")
    torch.distributed.barrier()
    log_progress(f"prediction run complete; saved {prediction_path}", rank)


def run_prediction(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    run_started = time.monotonic()
    prepared_directory = args.prepared_directory.resolve()
    log_progress(f"validating prepared inputs in {prepared_directory}", rank)
    with (prepared_directory / "preprocessing.json").open(encoding="utf-8") as metadata_file:
        prepared_metadata = json.load(metadata_file)
    conus = prepared_metadata.get("region") == "CONUS"
    native_europe = not conus and prepared_metadata.get("grid") == NATIVE_EUROPE_GRID
    output_directory = args.output_directory
    if native_europe and output_directory is None:
        output_directory = NATIVE_EUROPE_OUTPUT
    if conus and output_directory is None:
        raise ValueError("CONUS prediction requires --output-directory")
    target_day_count, pixel_count = validate_prepared_inputs(prepared_directory, conus)
    target_dates = np.load(prepared_directory / "target_dates.npy", mmap_mode="r",
                           allow_pickle=False)
    log_progress(
        f"validated {target_day_count:,} dates x {pixel_count:,} valid pixels "
        f"({target_day_count * pixel_count:,} predictions)",
        rank,
    )
    log_progress(f"loading target scaler from member {args.member}", rank)
    target_mean, target_standard_deviation, scaler_path = load_target_scalers(
        args.member
    )
    log_progress(
        f"loaded target scaler {scaler_path} "
        f"(wtdmean={target_mean:.6g}, wtdstd={target_standard_deviation:.6g})",
        rank,
    )

    model_path, _ = ensemble_member_paths(args.member)
    device = torch.device("cuda", local_rank)
    model_started = time.monotonic()
    log_progress(f"loading checkpoint {model_path}", rank)
    model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    log_progress(
        f"model loaded and wrapped with DDP in {time.monotonic() - model_started:.1f}s",
        rank,
    )

    if output_directory is not None:
        prediction_path = output_directory / f"member_{args.member:03d}" / "daily_wtd.npy"
    else:
        prediction_path = ENSEMBLE_PREDICTION_DIRECTORY / f"400px_member_{args.member}" / OUTPUT_FILENAME
    if native_europe:
        attempt = [uuid4().hex if rank == 0 else None]
        torch.distributed.broadcast_object_list(attempt, src=0)
        stage = prediction_path.parent.with_name(f".member_{args.member:03d}.{attempt[0]}.partial")
        temporary_path = stage / prediction_path.name
    else:
        temporary_path = Path(str(prediction_path) + ".partial")
    create_prediction_file(
        temporary_path,
        prediction_path,
        target_day_count,
        pixel_count,
        rank,
    )
    predictions = np.lib.format.open_memmap(
        temporary_path, mode="r+", dtype=np.float32
    )
    flat_predictions = predictions.reshape(-1)
    sample_count = target_day_count * pixel_count
    rank_sample_count = (
        sample_count + world_size - 1 - rank
    ) // world_size
    batch_count = (rank_sample_count + args.batch_size - 1) // args.batch_size
    progress_interval = max(1, batch_count // 20)
    prediction_started = time.monotonic()
    print(
        f"member={args.member} rank={rank}/{world_size}: predicting "
        f"{rank_sample_count:,} samples in {batch_count:,} batches "
        f"(batch size {args.batch_size})",
        flush=True,
    )

    with torch.inference_mode():
        for batch_index, (sample_indices, input_batch) in enumerate(
            iter_member_input_batches(
                prepared_directory,
                args.member,
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                with_indices=True,
            ),
            start=1,
        ):
            input_tensor = torch.from_numpy(input_batch).to(
                device=device, non_blocking=True
            )
            standardized_predictions = model(input_tensor).reshape(-1).cpu().numpy()
            if not np.isfinite(standardized_predictions).all():
                raise ValueError(
                    f"Member {args.member} generated non-finite predictions"
                )
            destandardized_predictions = (
                standardized_predictions * target_standard_deviation + target_mean
            )
            if not np.isfinite(destandardized_predictions).all():
                raise ValueError(
                    f"Member {args.member} generated non-finite WTD values"
                )
            flat_predictions[sample_indices] = destandardized_predictions
            if (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == batch_count
            ):
                elapsed = time.monotonic() - prediction_started
                completed = min(batch_index * args.batch_size, rank_sample_count)
                estimated_total = elapsed / batch_index * batch_count
                log_progress(
                    f"prediction batch {batch_index:,}/{batch_count:,} "
                    f"({completed:,}/{rank_sample_count:,} samples, "
                    f"{completed / rank_sample_count:.1%}); "
                    f"elapsed {elapsed / 60:.1f} min, "
                    f"ETA {(estimated_total - elapsed) / 60:.1f} min",
                    rank,
                )

    log_progress("flushing prediction data to disk", rank)
    predictions.flush()
    del flat_predictions, predictions
    torch.distributed.barrier()
    log_progress(
        f"inference finished in {(time.monotonic() - prediction_started) / 60:.1f} "
        "min; all ranks reached the output barrier",
        rank,
    )

    metadata = {
        "member": args.member,
        "model": str(model_path.resolve()),
        "scaler": str(scaler_path.resolve()),
        "prepared_inputs": str(prepared_directory),
        "prediction_file": str(prediction_path.resolve()),
        "target_dates_file": str(
            (prepared_directory / "target_dates.npy").resolve()
        ),
        "selected_pixels_file": str(
            (prepared_directory / "selected_pixels.npy").resolve()
        ),
        "shape": [target_day_count, pixel_count],
        "date_start": str(target_dates[0]),
        "date_end": str(target_dates[-1]),
        "units": (
            "Native training WTD units, destandardized with this member's wtdmean/wtdstd"
            if conus or native_europe
            else "wtd_a units, destandardized with this member's wtdmean/wtdstd"
        ),
        "feature_standardization": (
            "Per-member feature means and standard deviations from the "
            "training scaler pickle."
        ),
    }
    if native_europe:
        metadata.update(
            quantity="water_table_depth", grid=NATIVE_EUROPE_GRID,
            grid_shape=prepared_metadata["grid_shape"],
        )
    finalize_prediction(temporary_path, prediction_path, metadata, rank)
    log_progress(
        f"total run time {(time.monotonic() - run_started) / 60:.1f} min",
        rank,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict daily ERA5-Land WTD for valid observation-grid cells "
            "using one pretrained ensemble member."
        )
    )
    parser.add_argument(
        "--member",
        type=int,
        default=int(os.environ.get("ENS_MEMBER", "-1")),
        help="Pretrained ensemble member (0-99).",
    )
    parser.add_argument(
        "--prepared-directory",
        type=Path,
        default=DEFAULT_PREPARED_DIRECTORY,
        help="Output directory from preprocess_era5land_observation_inputs.py.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--expected-world-size", type=int, default=EXPECTED_WORLD_SIZE
    )
    parser.add_argument("--output-directory", type=Path,
                        help="Write daily WTD arrays and metadata in per-member subdirectories here.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = build_parser().parse_args()
    if not 0 <= args.member < 100:
        raise ValueError("--member must be an integer from 0 to 99")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.expected_world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(
            f"This pipeline requires exactly {EXPECTED_WORLD_SIZE} GPU processes"
        )
    if not (args.prepared_directory / "preprocessing.json").is_file():
        raise FileNotFoundError(
            "Prepared inputs are missing; first run "
            "preprocess_era5land_observation_inputs.py"
        )

    os.environ["ENS_MEMBER"] = str(args.member)
    print(
        f"Starting daily WTD prediction: member={args.member}, "
        f"prepared_directory={args.prepared_directory}, "
        f"batch_size={args.batch_size}, expected_gpus={args.expected_world_size}",
        flush=True,
    )
    rank, local_rank, world_size = initialize_distributed(
        args.expected_world_size
    )
    try:
        run_prediction(args, rank, local_rank, world_size)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
