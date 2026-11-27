"""Prepare and verify CONUS monthly coordinate sensitivity inputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from LSTM_model.conusmonthly.conusmonthly_evaluation import MONTHLY_GRID, MONTHLY_OUTPUTS
from LSTM_model.model.config import FEATURES_FILES, TRAINING_PERIOD, ensemble_member_paths, get_root_dir
from LSTM_model.model.ensemble_inputs import DYNAMIC_FILES, iter_member_input_batches, load_feature_scalers
from LSTM_model.model.predict_era5land_observations import load_target_scalers, validate_prepared_inputs
from LSTM_model.utils.observation_evaluation import MEMBERS

ROOT = Path(get_root_dir())
RUN = ROOT / "outputs/coordinate_sensitivity"
TRAINING = ROOT.parents[1] / "spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px"
COORDINATES = {"longitude": "lon2D_ts.npy", "latitude": "lat2D_ts.npy"}
SCENARIOS = {"A0": (), "A1": ("longitude",), "A2": ("latitude",),
             "A3": ("longitude", "latitude")}
STATIC = "static_features.npy"
SHARED = ("preprocessing.json", "context_dates.npy", "target_dates.npy", "selected_pixels.npy",
          "dynamic_features.npy", "selected_wtd.csv", "stations_with_static.csv", "selection.json")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def static_row(coordinate):
    return FEATURES_FILES.index(COORDINATES[coordinate]) - len(DYNAMIC_FILES)


def coordinate_statistics(values):
    return dict(mean=float(np.mean(values, dtype=np.float64)),
                min=float(np.min(values)), max=float(np.max(values)),
                count=int(values.size))


def european_coordinates(training):
    """Pool the static coordinate columns from all original training members."""
    columns = {coordinate: [] for coordinate in COORDINATES}
    for member in range(MEMBERS):
        source = training / f"400px_member_{member}"
        means, stds = load_feature_scalers(member)
        for coordinate, filename in COORDINATES.items():
            raw = np.load(source / filename, mmap_mode="r", allow_pickle=False)
            if raw.ndim != 2 or raw.shape[1] != 400 or len(raw) < TRAINING_PERIOD:
                raise ValueError(f"Unexpected European training coordinate shape: {source / filename}")
            train = raw[:TRAINING_PERIOD]
            first = train[0].astype(np.float64)
            if not np.isfinite(first).all() or not np.all(train == first):
                raise ValueError(f"Expected finite and time-invariant training coordinates: {source / filename}")
            index = FEATURES_FILES.index(filename)
            np.testing.assert_allclose(
                [first.mean(), first.std()], [means[index], stds[index]],
                rtol=2e-6, atol=5e-5, err_msg=f"Training scaler mismatch for {source / filename}",
            )
            columns[coordinate].append(first)
    return {coordinate: coordinate_statistics(np.concatenate(values))
            for coordinate, values in columns.items()}


def replacement_static(baseline, replacements, europe):
    static = baseline.copy()
    for coordinate in replacements:
        static[static_row(coordinate)] = europe[coordinate]["mean"]
    return static


def check_training_artifacts(models):
    for member, recorded in enumerate(models):
        for name, path in zip(("model", "scaler"), ensemble_member_paths(member), strict=True):
            if str(path.resolve()) != recorded[name]["path"] or sha256(path) != recorded[name]["sha256"]:
                raise ValueError(f"Training artifact changed for member {member}: {path}")


def verify(run, prepared_root=None, check_models=True):
    prepared = prepared_root or run / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    shared = prepared / "shared"
    for name, checksum in manifest["shared_sha256"].items():
        if sha256(shared / name) != checksum:
            raise ValueError(f"Experiment input changed: {shared / name}")
    baseline = np.load(shared / STATIC, allow_pickle=False)
    for scenario, replacements in SCENARIOS.items():
        branch = prepared / scenario
        for name in SHARED:
            if (branch / name).resolve() != (shared / name).resolve():
                raise ValueError(f"{branch / name} does not share the experiment input")
        actual = np.load(branch / STATIC, allow_pickle=False)
        expected = replacement_static(baseline, replacements, manifest["europe"])
        np.testing.assert_array_equal(actual, expected, err_msg=f"Unexpected static features in {scenario}")
    if check_models:
        check_training_artifacts(manifest["models"])
    return manifest


def verify_normalized_inputs(prepared, europe):
    # Raw verification covers every cell and date. Also exercise the inference reader.
    for member in (0, MEMBERS - 1):
        means, stds = load_feature_scalers(member)
        baseline = next(iter_member_input_batches(prepared / "A0", member, batch_size=8))
        for scenario, replacements in SCENARIOS.items():
            actual = next(iter_member_input_batches(prepared / scenario, member, batch_size=8))
            expected = baseline.copy()
            for coordinate in replacements:
                index = FEATURES_FILES.index(COORDINATES[coordinate])
                raw = np.float32(europe[coordinate]["mean"])
                expected[:, :, index] = (raw - means[index]) / stds[index]
            np.testing.assert_array_equal(actual, expected, err_msg=f"{scenario}, member {member}")


def validate_baseline(source, predictions, shape, models):
    """Check every member of the archived A0 ensemble against the confirmed inputs."""
    for member, artifacts in enumerate(models):
        path = predictions / f"member_{member:03d}/daily_wtd.npy"
        info = json.loads(path.with_suffix(".json").read_text())
        if (info["member"] != member or info["shape"] != list(shape)
                or Path(info["prepared_inputs"]).resolve() != source
                or Path(info["target_dates_file"]).resolve() != (source / "target_dates.npy").resolve()
                or Path(info["selected_pixels_file"]).resolve() != (source / "selected_pixels.npy").resolve()):
            raise ValueError(f"Archived baseline metadata does not match its inputs: {path}")
        for key in ("model", "scaler"):
            recorded = Path(info[key])
            current = Path(artifacts[key]["path"])
            # A0 predates the model directory move. The ensemble identity is confirmed.
            if recorded.name != current.name or (recorded.is_file() and sha256(recorded) != artifacts[key]["sha256"]):
                raise ValueError(f"Archived baseline member {member:03d} used a different {key}: {recorded}")
        daily = np.load(path, mmap_mode="r", allow_pickle=False)
        if daily.shape != shape or not np.isfinite(daily).all():
            raise ValueError(f"Invalid archived baseline predictions: {path}")


def baseline_inputs(manifest):
    source = Path(manifest["original_conus_prepared"])
    predictions = Path(manifest["baseline_archive"])
    for name, checksum in manifest["shared_sha256"].items():
        if sha256(source / name) != checksum:
            raise ValueError(f"Baseline input changed since preparation: {source / name}")
    validate_baseline(source, predictions, tuple(manifest["prediction_shape"]), manifest["models"])
    return source, predictions


def prepare(run, source, training, predictions):
    source, training, predictions = (path.resolve() for path in (source, training, predictions))
    if (run / "prepared").exists():
        raise FileExistsError(f"Experiment inputs already exist: {run / 'prepared'}")
    shape = validate_prepared_inputs(source, conus=True)
    europe = european_coordinates(training)
    models = []
    for member in range(MEMBERS):
        load_target_scalers(member)
        models.append({key: {"path": str(path.resolve()), "sha256": sha256(path)}
                       for key, path in zip(("model", "scaler"), ensemble_member_paths(member), strict=True)})
    validate_baseline(source, predictions, shape, models)
    run.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=run, prefix=".prepare-") as scratch:
        staged = Path(scratch) / "prepared"
        shared = staged / "shared"
        shared.mkdir(parents=True)
        for name in (*SHARED, STATIC):
            shutil.copyfile(source / name, shared / name)
        static = np.load(shared / STATIC, allow_pickle=False)
        if static.dtype != np.float32 or static.shape != (len(FEATURES_FILES) - len(DYNAMIC_FILES), shape[1]):
            raise ValueError(f"Unexpected static feature matrix: {static.shape}, {static.dtype}")
        dynamic = np.load(shared / "dynamic_features.npy", mmap_mode="r", allow_pickle=False)
        if dynamic.dtype != np.float32 or not np.isfinite(dynamic).all() or not np.isfinite(static).all():
            raise ValueError("Expected finite float32 CONUS features")
        for scenario, replacements in SCENARIOS.items():
            branch = staged / scenario
            branch.mkdir()
            for name in SHARED:
                (branch / name).symlink_to(Path("..") / "shared" / name)
            np.save(branch / STATIC, replacement_static(static, replacements, europe), allow_pickle=False)
        conus = {coordinate: coordinate_statistics(static[static_row(coordinate)])
                 for coordinate in COORDINATES}
        for coordinate, stats in conus.items():
            values = static[static_row(coordinate)]
            stats["below_europe"] = int((values < europe[coordinate]["min"]).sum())
            stats["above_europe"] = int((values > europe[coordinate]["max"]).sum())
            europe[coordinate]["replacement_float32"] = float(np.float32(europe[coordinate]["mean"]))
        manifest = dict(
            original_training_directory=str(training), training_days=TRAINING_PERIOD,
            original_conus_prepared=str(source), prediction_shape=list(shape),
            europe=europe, conus=conus, models=models, scenarios=SCENARIOS,
            shared_sha256={name: sha256(shared / name) for name in (*SHARED, STATIC)},
            baseline_archive=str(predictions),
        )
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(run, staged, check_models=False)
        verify_normalized_inputs(staged, europe)
        staged.rename(run / "prepared")
    print(json.dumps({"europe": europe, "conus": conus, "baseline_archive": manifest["baseline_archive"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--source", type=Path, default=ROOT / MONTHLY_GRID / "prepared")
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--baseline", type=Path, default=ROOT / MONTHLY_OUTPUTS / "predictions")
    parser.add_argument("--check-baseline", action="store_true",
                        help="Also validate the archived A0 ensemble during verification.")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run.resolve(), args.source, args.training, args.baseline)
    else:
        manifest = verify(args.run.resolve())
        print("Verified shared inputs, coordinate substitutions, checkpoints and scalers.")
        if args.check_baseline:
            _, predictions = baseline_inputs(manifest)
            print(f"Verified archived A0 predictions at {predictions}.")


if __name__ == "__main__":
    main()
