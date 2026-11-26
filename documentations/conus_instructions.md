# CONUS observation and prediction workflows

The CONUS workflows use HydroData USGS water-table depth observations, ParFlow
CONUS2 static fields, and ERA5-Land forcing. CONUSmonthly selects wells with
continuous monthly coverage and compares monthly anomalies. CONUSdaily
compares daily water-table depths over a continuous period. Both use the shared
pretrained ensemble for daily predictions.

- [CONUSmonthly pipeline](conusmonthly_pipeline.md) covers observation retrieval,
  well selection, the soil-indicator crosswalk, forcing, prediction, and evaluation.
- [CONUSdaily pipeline](conusdaily_pipeline.md) covers daily input preparation,
  importing or regenerating predictions, and daily metrics.

Use the registered HydroData account with `CONUS_HYDRODATA_PIN` or enter the
PIN when prompted. The soil-property definitions used by the crosswalk are in
[`sample_MapIndicatorToSoilProperties.tcl`](../sample_MapIndicatorToSoilProperties.tcl).
