"""Put CONUS groundwater observations and static predictors on the native ERA5-Land grid."""

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

from LSTM_model.utils.era5land_arco import GROUPS, arco_headers, grid_indices, open_arco_group
from LSTM_model.utils.file_io import save_csv, save_json
from LSTM_model.conusmonthly.conus_inputs import CONUS2_SHAPE
from LSTM_model.conusmonthly.conus_observations import (
    MONTHLY_END, MONTHLY_INPUTS, MONTHLY_START, connect_hydrodata, monthly_wtd, nearest_wells, read_sites,
)
from LSTM_model.model.config import get_root_dir


TCL = Path(get_root_dir()) / "sample_MapIndicatorToSoilProperties.tcl"
SOIL_PARAMETERS = {
    "permeability_x": "Perm.Value",
    "porosity": "Porosity.Value",
    "vg_alpha": "Saturation.Alpha",
    "vg_n": "Saturation.N",
}
SURFACE_FIELDS = (*SOIL_PARAMETERS, "sres")
SOIL_CODES = range(1, 14)
OBSERVATION_MAPPING = "Nearest eligible well to each assigned ERA5-Land centre"
GEOLOGY_CODES = range(19, 29)
CONUS2_ORIGIN = (-2208000.30881173, -1668999.65483222)
CONUS2_PROJECTION = (
    "+proj=lcc +lat_1=30 +lat_2=60 +lon_0=-97 +lat_0=40.0000076294444 "
    "+a=6370000 +b=6370000 +units=m"
)


def download_surface_parameters(root):
    """Cache the actual CONUS2 hydraulic fields at the top layer."""
    folder = root / "static" / "conus2" / "surface_parameters"
    folder.mkdir(parents=True, exist_ok=True)
    pending = [name for name in SURFACE_FIELDS if not (folder / f"{name}.npy").is_file()]
    if not pending:
        return
    hf = connect_hydrodata()
    height, width = CONUS2_SHAPE

    def fetch(name):
        print(f"Downloading CONUS2 surface {name}", flush=True)
        field = np.asarray(hf.get_gridded_data(
            dataset="conus2_domain", grid="conus2", temporal_resolution="static",
            variable=name, z=9, grid_bounds=[0, 0, width, height],
        ))
        if field.shape == (1, height, width):
            field = field[0]
        if field.shape != CONUS2_SHAPE:
            raise ValueError(f"Unexpected surface shape for {name}: {field.shape}")
        path = folder / f"{name}.npy"
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.save(stream, field)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, pending))


def reference_soils(path):
    values = dict(re.findall(r"^[ \t]*pfset[ \t]+([\w.]+)[ \t]+([^\s]+)", path.read_text(), re.M))
    records = []
    prefix = "GeomInput."
    for key, indicator in values.items():
        if not key.startswith(prefix) or not key.endswith(".Value"):
            continue
        geometry = key[len(prefix):-len(".Value")]
        keys = {name: f"Geom.{geometry}.{field}" for name, field in SOIL_PARAMETERS.items()}
        if re.fullmatch(r"TC\d+", geometry):
            material = "soil"
        elif re.fullmatch(r"BGR\d+", geometry):
            material = "bedrock"
            keys = {"permeability_x": keys["permeability_x"]}
        else:
            continue
        if not all(tag in values for tag in keys.values()):
            raise ValueError(f"Incomplete hydraulic properties for {geometry} in {path}")
        records.append({
            "geometry": geometry, "soil_indicator": int(indicator), "material": material,
            **{name: float(values[tag]) for name, tag in keys.items()},
        })
    if not records:
        raise ValueError(f"No soil or bedrock indicators in {path}")
    reference = pd.DataFrame(records).sort_values("soil_indicator", ignore_index=True)
    if reference.soil_indicator.duplicated().any():
        raise ValueError(f"Invalid hydraulic classes in {path}")
    return reference


def match_material(pixels, reference, names):
    # Each field contributes in units of its spread across the reference classes.
    # Log scaling treats multiplicative differences in conductivity and alpha consistently.
    def features(table):
        parameters = table[names].copy()
        for name in parameters.columns.intersection(["permeability_x", "vg_alpha"]):
            if not parameters[name].gt(0).all():
                raise ValueError(f"Non-positive {name} in hydraulic matching inputs")
            parameters[name] = np.log10(parameters[name])
        return parameters.to_numpy()

    target = features(reference)
    scale = target.std(axis=0)
    varies = scale > 0
    if not varies.any() or not np.isfinite(target).all():
        raise ValueError(f"Reference hydraulic classes cannot be distinguished using {names}")
    distances = np.sqrt(np.mean((
        (features(pixels)[:, None, varies] - target[None, :, varies]) / scale[varies]
    ) ** 2, axis=2))
    closest = distances.argmin(axis=1)
    return pixels.assign(
        soil_indicator=reference.soil_indicator.to_numpy()[closest],
        soil_geometry=reference.geometry.to_numpy()[closest],
        soil_match_distance=distances[np.arange(len(pixels)), closest],
        soil_match_method="hydraulic soil properties" if len(names) > 1 else "bedrock permeability",
    )


def classify_soils(pixels, reference):
    soils = match_material(pixels.loc[pixels.pf_indicator.isin(SOIL_CODES)],
                           reference.loc[reference.material == "soil"], list(SOIL_PARAMETERS))
    geology = match_material(pixels.loc[pixels.pf_indicator.isin(GEOLOGY_CODES)],
                             reference.loc[reference.material == "bedrock"], ["permeability_x"])
    return pd.concat([soils, geology]).sort_index()


def nearest_conus2_static(pixels, folder):
    project = Proj(CONUS2_PROJECTION)
    x, y = project(pixels.longitude.to_numpy(), pixels.latitude.to_numpy())
    origin_x, origin_y = CONUS2_ORIGIN
    # The published origin marks the lower-left edge. Cell centers sit 500 m inside.
    cols = np.rint((x - origin_x - 500) / 1000).astype(int)
    rows = np.rint((y - origin_y - 500) / 1000).astype(int)
    height, width = CONUS2_SHAPE
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    result = pixels.assign(conus2_x=cols, conus2_y=rows, inside_static_grid=in_bounds)
    for name in ("slope_x", "slope_y", "pf_indicator", "latitude", "longitude", "mask", *SURFACE_FIELDS):
        path = folder / f"{name}.npy"
        if name in SURFACE_FIELDS:
            path = folder / "surface_parameters" / path.name
        grid = np.load(path, mmap_mode="r", allow_pickle=False)
        if name == "pf_indicator":
            grid = grid[9]
        if grid.shape != CONUS2_SHAPE:
            raise ValueError(f"Unexpected CONUS2 surface grid in {path}: {grid.shape}")
        values = np.full(len(pixels), np.nan)
        values[in_bounds] = grid[rows[in_bounds], cols[in_bounds]]
        output_name = f"conus2_{name}" if name in ("latitude", "longitude") else name
        result[output_name] = values
    return result


def assign_wells(wells, grid):
    rows, cols = grid_indices(grid, wells)
    latitude, longitude = grid["latitude"][:], grid["longitude"][:]
    # Normalize longitude for the model and for CONUS2 projection.
    signed_lon = (longitude[cols] + 180) % 360 - 180
    return wells.assign(y=rows, x=cols, grid_latitude=latitude[rows], grid_longitude=signed_lon)


def write_observations(output, wells, observations):
    selected = nearest_wells(wells, observations)
    if len(selected) != len(wells.drop_duplicates(["y", "x"])):
        raise ValueError("At least one assigned observation cell has no eligible well")
    cell_index = pd.MultiIndex.from_frame(selected[["y", "x"]])
    values = observations.loc[:, selected.index].set_axis(cell_index, axis=1)
    values.index = values.index.to_timestamp().rename("date")
    metadata = selected[["y", "x", "well_distance_km", "candidate_well_count"]].reset_index()
    gridded = (values.stack(["y", "x"], future_stack=True).rename("wtd_m").reset_index()
               .merge(metadata, on=["y", "x"], validate="many_to_one").assign(well_count=1))
    save_csv(selected, output / "selected_wells_by_cell.csv")
    save_csv(gridded, output / "observations_gridded.csv", index=False)
    return len(selected)


def observation_months(path):
    return monthly_wtd(path, pd.period_range(MONTHLY_START, MONTHLY_END, freq="M"))


def build_grid(root, output, tcl):
    metadata = read_sites(root / "selected_stations.csv")
    observations = observation_months(root / "selected_wtd.csv")
    server, group_name, _ = GROUPS[0]
    grid = open_arco_group(server, group_name, arco_headers())
    wells = assign_wells(metadata, grid)
    pixels = (wells[["y", "x", "grid_latitude", "grid_longitude"]].drop_duplicates(["y", "x"])
              .rename(columns={"grid_latitude": "latitude", "grid_longitude": "longitude"})
              .sort_values(["y", "x"], ignore_index=True))
    output.mkdir(parents=True, exist_ok=True)
    save_csv(wells, output / "well_era5land_cells.csv")
    n_selected = write_observations(output, wells, observations)

    sampled = nearest_conus2_static(pixels, root / "static" / "conus2")
    usable = (
        sampled.inside_static_grid & sampled["mask"].gt(0)
        & sampled.pf_indicator.isin([*SOIL_CODES, *GEOLOGY_CODES])
        & sampled.slope_x.between(-10, 10) & sampled.slope_y.between(-10, 10)
        & sampled.permeability_x.gt(0) & sampled.vg_alpha.gt(0)
        & sampled.vg_n.gt(1) & sampled.porosity.between(0, 1)
        & np.isfinite(sampled[list(SOIL_PARAMETERS)]).all(axis=1)
    )
    if not usable.any():
        raise ValueError("No ERA5-Land cell has a usable nearest CONUS2 static cell")
    reference = reference_soils(tcl)
    static = classify_soils(sampled.loc[usable], reference)
    retained = wells.join(static.set_index(["y", "x"])[["conus2_y", "conus2_x", "soil_indicator"]],
                          on=["y", "x"], how="inner")
    save_csv(reference, output / "tcl_reference_soils.csv", index=False)
    save_csv(static, output / "soil_mapping.csv", index=False)
    save_csv(sampled.loc[~usable], output / "excluded_static_pixels.csv", index=False)
    save_csv(static, output / "static_pixels.csv", index=False)
    save_csv(retained, output / "stations_with_static.csv")
    for name in ("selection.json", "selected_wtd.csv"):
        shutil.copyfile(root / name, output / name)
    summary = {
        "grid": "native ERA5-Land 0.1 degree",
        "wells": len(wells), "observation_cells": len(pixels),
        "gridded_observations_file": "observations_gridded.csv",
        "observation_mapping": OBSERVATION_MAPPING,
        "selected_observation_wells": n_selected,
        "wells_with_static": len(retained), "cells_with_static": len(static),
        "static_source": "HydroData conus2_domain nearest cell center in Lambert conformal projection",
        "soil_layer": 9, "soil_matching_parameters": list(SOIL_PARAMETERS),
        "bedrock_matching_parameters": ["permeability_x"],
        "source_tcl": str(tcl.resolve()),
    }
    save_json(summary, output / "gridding.json")
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download-soil", "grid", "observations"])
    parser.add_argument("--root", type=Path, default=MONTHLY_INPUTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--soil-tcl", type=Path, default=TCL)
    args = parser.parse_args()
    if args.stage == "download-soil":
        download_surface_parameters(args.root)
    elif args.stage == "observations":
        output = args.output or args.root / "era5land_grid"
        selected = write_observations(output, read_sites(output / "well_era5land_cells.csv"),
                                      observation_months(output / "selected_wtd.csv"))
        manifest = output / "gridding.json"
        if manifest.is_file():
            info = json.loads(manifest.read_text())
            info.update(observation_mapping=OBSERVATION_MAPPING, selected_observation_wells=selected)
            save_json(info, manifest)
        print(f"Gridded observations using {selected} nearest wells in {output}")
    else:
        build_grid(args.root, args.output or args.root / "era5land_grid", args.soil_tcl)


if __name__ == "__main__":
    main()
