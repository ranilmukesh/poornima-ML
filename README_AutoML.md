# PyCaret AutoML Stack Ridge for Diabetes HbA1c Prediction

## 🎯 Project Overview

This project implements a **PyCaret AutoML Stack Ridge ensemble** for predicting HbA1c levels (PostBLHBA1C) in diabetes patients. The goal is to achieve **MAE < 0.5** for clinical accuracy using automated machine learning.

### 🔬 Clinical Objective
- **Target Variable**: PostBLHBA1C (Post Blood Glucose HbA1c levels)
- **Clinical Goal**: MAE < 0.5 for accurate diabetes management
- **Approach**: AutoML with Stack Ridge ensemble using PyCaret

## 📊 Datasets

The project processes three diabetes datasets:

1. **nmbfinalDiabetes_4**: Primary diabetes dataset with comprehensive features
2. **nmbfinalnewDiabetes_3**: Secondary diabetes dataset with extended features  
3. **PrePostFinal_3**: Pre/Post intervention diabetes dataset

## 🤖 AutoML Pipeline

### Core Components

1. **PyCaret Setup**: Automated data preprocessing and environment configuration
2. **Model Comparison**: AutoML selection of best performing algorithms
3. **Hyperparameter Tuning**: Automated optimization using Optuna
4. **Stack Models**: Ridge ensemble with automated stacking
5. **Feature Importance**: Interpretability analysis for clinical insights
6. **Model Evaluation**: Comprehensive performance assessment

### Key Features

- ✅ **Fully Automated**: End-to-end AutoML pipeline
- ✅ **Ridge Stacking**: Meta-learner ensemble for robust predictions
- ✅ **Clinical Metrics**: MAE-focused optimization for medical accuracy
- ✅ **Feature Engineering**: Automated preprocessing and transformation
- ✅ **Model Interpretability**: Feature importance for clinical understanding
- ✅ **Multi-Dataset**: Processes all three datasets automatically

## 🚀 Usage

### Quick Start

```bash
# Run the complete AutoML pipeline
python ml_pycaret_automl.py
```

### What It Does

1. **Data Loading**: Validates and loads all three datasets
2. **AutoML Setup**: Configures PyCaret regression environment
3. **Model Selection**: Compares Ridge, RF, ET, GBR, LightGBM, XGBoost, CatBoost
4. **Optimization**: Tunes hyperparameters automatically
5. **Stacking**: Creates Ridge ensemble with best models
6. **Evaluation**: Provides clinical performance metrics
7. **Saving**: Exports models and results for deployment

## 📈 Expected Output

```
🚀 PyCaret AutoML Stack Ridge Pipeline
================================================================================
🎯 Objective: Predict PostBLHBA1C with MAE < 0.5
🤖 Method: AutoML with Stack Ridge ensemble

🔥 PROCESSING DATASET 1/3: nmbfinalDiabetes_4
📂 Loading nmbfinalDiabetes_4 from: final_imputed_data/...
   ✅ Shape: (X, Y)
   📊 Target statistics: Mean: X.XX, Std: X.XX

🤖 Starting PyCaret AutoML for nmbfinalDiabetes_4
🔧 Setting up AutoML environment...
   ✅ AutoML environment configured

🏁 Comparing models with AutoML...
   ✅ Model comparison completed
   🎯 Top models selected based on MAE performance

⚡ Hyperparameter tuning with AutoML...
   ✅ Hyperparameter tuning completed

🚀 Creating Stack Ridge ensemble with AutoML...
   ✅ Stack Ridge ensemble created
   🎯 Meta-learner: Ridge Regression

📊 Evaluating Stack Ridge model...
   📈 Performance Metrics:
      • MAE: 0.XXXX
      • RMSE: 0.XXXX
      • R²: 0.XXXX
   🏥 Clinical Assessment:
      • Status: 🎯 EXCELLENT - Target achieved!
      • Grade: A

🔍 Analyzing feature importance...
   🔬 Top 10 Most Important Features:
      1. Feature_A: 0.XXXX
      2. Feature_B: 0.XXXX
      ...

💾 Saving model and results...
   ✅ Model saved: models/stack_ridge_automl_...

🏆 FINAL AUTOML SUMMARY - ALL DATASETS COMPARISON
📊 AutoML Model Performance Comparison:
   • Dataset_1: MAE=0.XXXX, R²=0.XXXX, Grade=A - 🎯 TARGET MET
   • Dataset_2: MAE=0.XXXX, R²=0.XXXX, Grade=B - 📈 NEEDS IMPROVEMENT
   • Dataset_3: MAE=0.XXXX, R²=0.XXXX, Grade=A - 🎯 TARGET MET

🥇 Best AutoML Model: Dataset_1
   • MAE: 0.XXXX
   • Clinical Grade: A

💡 Clinical Recommendations:
   ✅ AutoML model ready for clinical validation
   🏥 Consider deployment for diabetes management
```

## 📁 File Structure

```
poornima-ML/
├── ml_pycaret_automl.py          # Main AutoML pipeline
├── final_imputed_data/           # Input datasets
│   ├── nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv
│   ├── nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv
│   └── PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv
├── models/                       # Generated models and results
│   ├── stack_ridge_automl_*.pkl  # Saved AutoML models
│   └── results_*.json           # Performance metrics
└── README.md                    # This documentation
```

## 🔧 Technical Requirements

### Dependencies
```bash
pip install pycaret pandas numpy scikit-learn
```

### Optional (for advanced models)
```bash
pip install xgboost lightgbm catboost optuna
```

## 🏥 Clinical Interpretation

### Performance Grades
- **Grade A (MAE < 0.5)**: Excellent - Ready for clinical use
- **Grade B (MAE < 0.75)**: Good - Clinically acceptable
- **Grade C (MAE < 1.0)**: Fair - Needs improvement
- **Grade D (MAE ≥ 1.0)**: Poor - Requires significant work

### Clinical Value
- **MAE < 0.5**: Provides reliable HbA1c predictions for diabetes management
- **Feature Importance**: Identifies key factors influencing HbA1c levels
- **Model Interpretability**: Supports clinical decision-making
- **Automated Pipeline**: Reduces manual ML expertise requirements

## 🔄 AutoML Advantages

### vs Manual Approach
1. **Automation**: No manual model selection or hyperparameter tuning
2. **Optimization**: Systematic search across model space
3. **Consistency**: Reproducible results across datasets
4. **Efficiency**: Faster development and deployment
5. **Best Practices**: Built-in ML best practices and validation

### PyCaret Benefits
- **Low-Code**: Minimal coding required for complex ML pipelines
- **Comprehensive**: Handles preprocessing, modeling, and evaluation
- **Stack Models**: Advanced ensemble methods with simple API
- **Interpretability**: Built-in model explanation capabilities
- **Production Ready**: Easy model deployment and saving

## 🎯 Next Steps

1. **Validation**: Test AutoML models on holdout clinical data
2. **Deployment**: Deploy best performing model for clinical use
3. **Monitoring**: Track model performance in production
4. **Iteration**: Retrain with new data periodically
5. **Integration**: Connect with clinical decision support systems

## 📚 References

- [PyCaret Documentation](https://pycaret.org/)
- [Stack Models Guide](https://pycaret.gitbook.io/docs/get-started/functions/optimize#stack_models)
- [AutoML Best Practices](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog)

---

**🎉 Ready to run AutoML for diabetes prediction with clinical-grade accuracy!**