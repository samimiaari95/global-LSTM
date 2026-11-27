# CONUS observation and prediction workflow

CONUSmonthly uses HydroData USGS water-table depth observations, ParFlow CONUS2
static fields, and ERA5-Land forcing. It selects wells with continuous monthly
coverage and compares monthly anomalies derived from observations and daily
predictions from the shared pretrained ensemble.

The [CONUSmonthly pipeline](conusmonthly_pipeline.md) covers observation retrieval,
well selection, the soil-indicator crosswalk, forcing, prediction, and evaluation.

Use the registered HydroData account with `CONUS_HYDRODATA_PIN` or enter the
PIN when prompted. The soil-property definitions used by the crosswalk are in
[`sample_MapIndicatorToSoilProperties.tcl`](../sample_MapIndicatorToSoilProperties.tcl).
