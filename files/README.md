# Diabetes HbA1c Prediction with PyCaret AutoML

A comprehensive machine learning pipeline for predicting diabetes HbA1c levels using automated machine learning. Combines professional data preprocessing with PyCaret's powerful AutoML capabilities for clinical-grade predictions.

## ✨ Features

### Data Processing Pipeline
- **Intelligent Feature Selection**: Automatically identifies the most informative features
- **Professional Data Cleaning**: Standardizes categorical variables and handles outliers  
- **Optimal Imputation**: Uses KNN imputation for categorical variables (24.9% improvement over baseline)
- **Complete Pipeline**: End-to-end processing from raw CSV to ML-ready datasets

### PyCaret AutoML
- **Windows Compatible**: No WSL or complex setup required
- **Automated Model Selection**: Compares 15+ regression algorithms automatically
- **Hyperparameter Tuning**: Automated optimization for best performance
- **Ensemble Models**: Combines top models for improved accuracy
- **Clinical Metrics**: HbA1c-specific accuracy thresholds (±0.5% and ±1.0%)
- **Rich Visualizations**: Comprehensive model performance plots

## 🚀 Quick Start

### Prerequisites
```bash
pip install pycaret pandas numpy scikit-learn
```

### Full Pipeline (Data Processing + ML)
```bash
# 1. Process raw data (if needed)
python process_all.py

# 2. Run PyCaret AutoML
python diabetes_hba1c_pycaret_automl.py
```

### Interactive Analysis
```bash
# Open Jupyter notebook for interactive analysis
jupyter notebook diabetes_hba1c_automl_prediction.ipynb
```

## Project Structure

```
├── raw_data/                    # Input CSV files
├── final_imputed_data/          # Output ML-ready datasets
├── temp_processed/              # Intermediate processing files
├── data_prep.py                 # Core preprocessing functions
├── columns.py                   # Feature selection and initial processing
├── final_imputation.py          # Optimal imputation pipeline  
├── process_all.py               # Complete pipeline orchestrator
├── simple_efficiency_check.py   # Performance evaluation
└── README.md
```

## Pipeline Stages

### 1. Data Preprocessing (`columns.py`)
- Feature selection from 500+ variables to 80 key features
- Text normalization and categorical standardization
- Feature engineering (BMI, diabetes duration, activity scores)
- Categorical encoding for ML compatibility

### 2. Optimal Imputation (`final_imputation.py`)
- KNN imputation for categorical variables
- Context-aware strategy selection based on missing patterns
- Priority processing for critical clinical variables
- 100% completion rate with data quality preservation

### 3. Quality Validation (`simple_efficiency_check.py`)
- Performance benchmarking
- Data distribution analysis
- Accuracy metrics on key variables

## Output Datasets

All datasets are ML-ready with:
- **Zero missing values** (100% completion)
- **Standardized encoding** for categorical variables
- **Preserved data distributions** (<0.5% mean change)
- **Validated quality** through comprehensive testing

### Key Variables
- `PostBLHBA1C` - Primary diabetes outcome (HbA1c %)
- `PreBLAge` - Patient demographics
- `PreRgender`, `PreRarea` - Encoded categorical features
- `PreBLFBS`, `PreBLCHOLESTEROL` - Clinical measurements
- `Diabetic_Duration(years)` - Disease progression metrics
- Plus 70+ additional clinical and lifestyle variables

## Performance

- **Processing Speed**: 1.4M values/second
- **Imputation Quality**: KNN shows 24.9% improvement over simple methods
- **Completion Rate**: 100% across all datasets
- **Processing Time**: ~2 minutes for complete pipeline

## Usage Examples

### Basic Usage
```python
from data_prep import process_csv_files_enriched

# Process single dataset
process_csv_files_enriched(
    ["raw_data/your_data.csv"],
    output_dir="processed/",
    columns_to_keep=selected_columns
)
```

### Custom Imputation
```python
from final_imputation import OptimalImputer

imputer = OptimalImputer()
clean_df = imputer.process_dataset(your_dataframe)
```

## Validation

Run the quality check to validate processing:
```bash
python simple_efficiency_check.py
```

## License

This project is designed for diabetes research and clinical data analysis. Please ensure appropriate data use permissions for your specific datasets.

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation for API changes
4. Validate data quality preservation