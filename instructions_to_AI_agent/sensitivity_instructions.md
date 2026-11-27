## Experiment: Sensitivity of CONUS WTD predictions to geographical coordinates

### Objective

Investigate whether the predictive performance of the LSTM ensemble transferred from Europe to CONUS is influenced by longitude and latitude values that fall outside the training data range.

### Instructions

1. Inspect the existing CONUS inference and evaluation workflow to understand how static attributes are prepared, normalized, and passed to the trained LSTM ensemble.

2. Extract the mean longitude and latitude from the original European training data.

3. Implement the following four inference scenarios:

| Scenario                   | Longitude              | Latitude               |
| -------------------------- | ---------------------- | ---------------------- |
| A0 – Baseline              | Original CONUS         | Original CONUS         |
| A1 – Longitude sensitivity | European training mean | Original CONUS         |
| A2 – Latitude sensitivity  | Original CONUS         | European training mean |
| A3 – Combined sensitivity  | European training mean | European training mean |

4. Apply the coordinate modifications before normalization. Use the original training normalization parameters in all scenarios.

5. Keep everything else identical across scenarios:

   - Trained LSTM weights and architecture.
   - Dynamic input variables from ERA5.
   - All remaining static attributes.
   - Spatial and temporal extent.
   - Ensemble members and aggregation method.
   - Evaluation reference data and valid evaluation mask.

6. Run inference for all ensemble members for each scenario. Reuse the existing baseline predictions if they were generated under identical conditions.

7. Evaluate each scenario using the existing performance metrics, including Pearson correlation, RMSE, KGE, and NSE.

8. Quantify the differences between each modified scenario and the baseline at every evaluated pixel:

   - Difference in each performance metric.
   - RMSE between the predicted WTD anomalies time series of the modified scenario and the baseline.

9. Generate the following figures using consistent plotting conventions and axis limits:

   - Comparative boxplots of performance metrics across the four scenarios.
   - Spatial maps of changes in RMSE, Pearson correlation, KGE and NSE relative to the baseline.
   - Comparative WTD anomalies time series at representative locations exhibiting low, moderate, and high coordinate sensitivity.

10. Generate a summary table containing the median and interquartile range of each performance metric and its change relative to the baseline.

### Implementation requirements

- Do not retrain or modify the trained models.
- Do not modify or overwrite existing input data, predictions, or evaluation results.
- Reuse existing helper functions and scripts wherever possible.
- Store the experiment scripts, predictions, evaluation metrics, and figures as a new project named `coordinate_sensitivity`.
- Document the methodology, implementation, coordinate ranges, normalization procedure, results, and interpretation in a `.md` file in the documentation directory.
- Verify that all scenarios differ exclusively in their longitude and/or latitude inputs.

### Expected interpretation

Determine whether the ensemble's predictive accuracy and predicted WTD anomalies dynamics remain robust when the out-of-range geographical coordinates are replaced with values representative of the European training domain.

Distinguish between changes in predictive accuracy and changes in the predictions themselves. Similar performance metrics do not necessarily imply similar predictions.
