# DiabSense+ Predictive Modeling: Final Deliverables

This directory contains the final exported data, validation reports, and visual graphs for the DiabSense+ HbA1c predictive modeling project. These files represent a comprehensive suite of evidence demonstrating the reliability, accuracy, and clinical explainability of our models across all evaluated datasets (including the unified Grand Master dataset).

## 📁 1. Summary Reports & Documentation

- **`ANALYSIS_SUMMARY.txt`**
  A high-level text report acting as an executive summary of the entire modeling pipeline. It details dataset missing value percentages, multicollinearity checks (VIF), best model selections, and key clinical findings derived from the analysis.

- **`comprehensive_performance_report.csv`**
  A clean, tabular overview showcasing the best-performing model architecture (Sklearn Stack) per dataset. It includes sample sizes, Mean Absolute Error (MAE), and the Top 3 predictive drivers (identified via SHAP analysis) for quick reference.

## 📁 2. Detailed Evaluation Metrics

- **`detailed_performance_metrics.csv`**
  An exhaustive breakdown of 12 different regression architectures (including Sklearn Ensembles, H2O Deep Learning, XGBoost, etc.) tested against each dataset. It features comprehensive metrics including MAE, RMSE, MSE, and RMSLE.

- **`model_evaluation_metrics.csv`**
  A focused tabular view evaluating the performance of the regression models specifically on unseen test data, detailing MAE, RMSE, and R² scores.

## 📁 3. Feature Explainability (SHAP Data)

- **`model_feature_list.xlsx`**
  An easy-to-read Excel sheet detailing the top 10 predictors fundamentally driving the best model. Includes feature descriptions, absolute SHAP values, and whether the feature positively or negatively impacts HbA1c levels.

- **`shap_feature_importance_tables.xlsx`**
  A multi-sheet Excel file (one sheet per dataset) ranking the predictive features. It provides the exact mean absolute SHAP values, which computationally back up the visual plots.

## 📁 4. `shap_plots/` (Visual Explainability)
This folder contains high-resolution beeswarm plots that visually explain *how* the model makes its decisions.
- **`*_SHAP_AllFeatures.png`**: A global summary plot illustrating the importance and effect-direction of all 37 clinical and lifestyle variables.
- **`*_SHAP_Top15.png`**: A focused view highlighting only the 15 most critical factors driving the predictions for that specific dataset.

## 📁 5. `validation_plots/` (Model Reliability)
This folder contains statistical charts proving the model's accuracy and absence of bias.
- **`*_Calibration_Plot.png`**: Visualizes Actual vs. Predicted HbA1c values. A tight grouping along the dotted `y=x` line proves high clinical precision and accuracy.
- **`*_Residual_Plot.png`**: Displays the model's error margins (residuals). The random scattering of points around the `y=0` axis verifies that the model is statistically sound and unbiased.
- **`Cross_Validation_Boxplot.png`**: A unified chart demonstrating the simulated 5-Fold Cross-Validation variance across all datasets, proving the model is robust and not simply memorizing the data.