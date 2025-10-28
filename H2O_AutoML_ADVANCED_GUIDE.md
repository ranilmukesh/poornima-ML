# H2O AutoML Advanced Script - User Guide

## Overview
This is a comprehensive refactoring of the H2O AutoML script with advanced analysis, validation, and modular execution control.

## 🎯 Key Improvements

### 1. **Focused AutoML Strategy (Primary)**
- **Algorithms**: XGBoost, GBM, StackedEnsemble only (removed GLM, DRF, DeepLearning)
- **Max Models**: Increased from 30 to 50
- **Runtime**: 1800s (30 min) for individual datasets, optional 3600s (60 min) for combinations
- **Rationale**: Previous runs showed XGBoost/GBM consistently outperforming others

### 2. **Integrated Advanced Analysis**
Each model training now includes:
- **Feature Importance**: Top 20 features with relative importance scores
- **SHAP Analysis**: Contribution plots with intelligent error handling
- **Detailed Metrics**: MAE, RMSE, R², MSE, Mean Residual Deviance
- **Performance Timing**: Track training duration

### 3. **External Validation**
- Cross-dataset testing (e.g., Model trained on Dataset 1, tested on Dataset 2)
- Uses same 80/20 split methodology (seed=42) for consistency
- Evaluates generalization capability across different patient populations

### 4. **Two-Dataset Combinations**
New training runs on pairwise combinations:
- Dataset 1 + Dataset 2
- Dataset 1 + Dataset 3  
- Dataset 2 + Dataset 3

Rationale: May capture unique patterns not present in individual or full combination

### 5. **Optional Advanced Strategies**
- **Strategy 2**: Yeo-Johnson preprocessing (power transformation)
- **Strategy 3**: Manual stacked ensemble with custom metalearner
- Only triggered if no model achieves MAE < 0.5

### 6. **Modular Execution Control**
Top-of-script flags for easy customization:
```python
RUN_INDIVIDUAL_DATASETS = True      # Models 1-3
RUN_THREE_DATASET_COMBO = True      # Model 4
RUN_TWO_DATASET_COMBOS = True       # Models 5-7
RUN_EXTERNAL_VALIDATION = True      # Cross-dataset testing
RUN_PREPROCESSING_STRATEGY = False  # Optional Strategy 2
RUN_MANUAL_STACKING = False         # Optional Strategy 3
```

## 📂 File Requirements

Ensure these files exist in `./final_imputed_data/` (or `/content/final_imputed_data/` for Colab):
- `nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv`
- `nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv`
- `PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv`

**Note**: The script now correctly uses the `_final_imputed.csv` files (post-MICE imputation).

## 🚀 Execution Phases

### Phase 1: Individual Datasets (Models 1-3)
Trains separate models on each of the three datasets.
- **Runtime**: 1800s each (30 minutes × 3 = 90 minutes)
- **Output**: 3 models + analysis for each

### Phase 2: Three-Dataset Combination (Model 4)
Combines all three datasets and trains a single model.
- **Runtime**: 3600s (60 minutes) if `USE_EXTENDED_RUNTIME_FOR_COMBOS = True`
- **Output**: 1 model + analysis

### Phase 3: Two-Dataset Combinations (Models 5-7)
Trains on pairwise dataset combinations.
- **Runtime**: 3600s each if extended runtime enabled (180 minutes total)
- **Output**: 3 models + analysis for each

### Phase 4: External Validation
Validates models across different datasets.
- **Example**: Model trained on Dataset 1, tested on Dataset 2's test split
- **Output**: External MAE, RMSE, R² for generalization assessment

### Phase 5: Optional Strategies (Conditional)
Only runs if no model achieves MAE < 0.5:
- Applies to the best-performing dataset
- Tests preprocessing and/or manual stacking

## 📊 Output Summary

### Console Output Includes:
1. **Per-Model Analysis**:
   - Leaderboard (top 15 models)
   - CV metrics (10-fold)
   - Test metrics (20% holdout)
   - Feature importance (top 20)
   - SHAP analysis
   - Clinical goal achievement status

2. **Final Summary Table**:
   - All models compared side-by-side
   - Sorted by Test MAE (ascending)
   - Goal achievement indicators (✅/❌)

3. **Feature Comparison**:
   - Top 5 features for each model
   - Identify consistent predictors

4. **External Validation Table**:
   - Cross-dataset performance
   - Generalization assessment

5. **Aggregate Statistics**:
   - Average/Median/Best MAE
   - Total training time
   - Goal achievement rate

### Saved Models
All models saved to `./models/` (or `/content/models/`):
- `h2o_model_nmbfinalDiabetes/`
- `h2o_model_nmbfinalnewDiabetes/`
- `h2o_model_PrePostFinal/`
- `h2o_model_Combined_3_Datasets/`
- `h2o_model_nmbfinalDiabetes_nmbfinalnewDiabetes/`
- etc.

## ⏱️ Estimated Total Runtime

With default settings:
- Individual datasets: 3 × 30 min = 90 min
- 3-dataset combo: 60 min
- 2-dataset combos: 3 × 60 min = 180 min
- External validation: ~10 min
- **Total**: ~5-6 hours

To reduce runtime:
```python
RUN_TWO_DATASET_COMBOS = False  # Saves 180 minutes
USE_EXTENDED_RUNTIME_FOR_COMBOS = False  # Reduces combo runtime to 30 min each
```

## 🎛️ Customization Options

### Quick Test Run (Fast Mode)
```python
RUN_INDIVIDUAL_DATASETS = True
RUN_THREE_DATASET_COMBO = True
RUN_TWO_DATASET_COMBOS = False  # Skip
RUN_EXTERNAL_VALIDATION = False  # Skip
STANDARD_RUNTIME = 600  # 10 minutes
USE_EXTENDED_RUNTIME_FOR_COMBOS = False
```

### Aggressive Optimization (If Close to Goal)
```python
RUN_TWO_DATASET_COMBOS = True
RUN_PREPROCESSING_STRATEGY = True
RUN_MANUAL_STACKING = True
EXTENDED_RUNTIME = 7200  # 2 hours for combos
```

### Focus on Best Dataset Only
```python
RUN_INDIVIDUAL_DATASETS = False
RUN_THREE_DATASET_COMBO = True  # Only this
RUN_TWO_DATASET_COMBOS = False
EXTENDED_RUNTIME = 10800  # 3 hours
```

## 🔍 Interpreting Results

### Feature Importance
- **High Importance**: Strong predictive power
- **Consistent Across Models**: Robust predictors
- **Dataset-Specific**: Unique to that population

### SHAP Analysis
- **Positive Contributions**: Feature increases HbA1c prediction
- **Negative Contributions**: Feature decreases HbA1c prediction
- **Magnitude**: Larger absolute value = stronger effect

### External Validation
- **Lower External MAE**: Good generalization
- **Similar MAE to Test**: Model is robust
- **Much Higher External MAE**: Overfitting to training distribution

### Clinical Goal (MAE < 0.5)
- **✅ Achieved**: Model is clinically useful
- **Close (0.5-0.7)**: May be acceptable with calibration
- **Far (>0.7)**: Requires significant improvement

## 🐛 Troubleshooting

### H2O Initialization Fails
```python
# Try manually starting H2O first
import h2o
h2o.init(max_mem_size='8G')  # Reduce if system has limited RAM
```

### SHAP Analysis Times Out
- Expected for large datasets or StackedEnsembles
- Script automatically limits to 500 rows
- Non-critical - models still train successfully

### File Not Found Errors
- Verify `_final_imputed.csv` files exist
- Check `data_dir` path matches your environment
- For Colab: Upload files to `/content/final_imputed_data/`

### Memory Issues
```python
h2o.init(max_mem_size='32G')  # Increase if available
# Or reduce dataset size before training
```

## 📝 Next Steps After Running

1. **Review Final Summary Table**: Identify best model(s)
2. **Check Feature Importance**: Validate with domain knowledge
3. **Analyze External Validation**: Assess generalization
4. **If Goal Not Met**:
   - Enable `RUN_PREPROCESSING_STRATEGY = True`
   - Enable `RUN_MANUAL_STACKING = True`
   - Increase `EXTENDED_RUNTIME` to 10800s (3 hours)
5. **Deploy Best Model**: Use saved model path for inference

## 🏆 Expected Improvements Over Baseline

- **Focused Algorithms**: 10-20% faster training, similar/better accuracy
- **Increased Models**: Higher chance of finding optimal hyperparameters
- **External Validation**: Reveals true generalization capability
- **2-Dataset Combos**: May discover optimal data mix for MAE < 0.5
- **Advanced Analysis**: Better understanding of model behavior

## 📞 Support

For issues or questions:
1. Check console output for specific error messages
2. Verify data file integrity with `data_validation.py`
3. Review imputation quality with `imputation_quality_evaluation.py`
4. Consider ensemble of multiple models if single model struggles

---
**Version**: 2.0 - Advanced Analysis & Validation  
**Last Updated**: 2025-10-27  
**Compatible with**: H2O 3.x, Python 3.8+
