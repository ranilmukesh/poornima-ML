# Poornima-ML: Diabetes Dataset Preprocessing

This project processes diabetes datasets for machine learning using advanced feature selection and enrichment.

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### Step 2: Run the Processing Pipeline
```bash
python columns.py
```

This single command will:
- Read raw CSV files from `raw data/` folder
- Clean and preprocess the data
- Select top 50% most important features for enrichment
- Fill missing values using KNN imputation with important features
- Encode categorical variables (gender, area) to numeric values
- Save processed files to `cleaned data/` folder

## 📁 Project Structure

```
poornima-ML/
├── raw data/                    # Original CSV files (input)
│   ├── nmbfinalDiabetes (4).csv
│   ├── nmbfinalnewDiabetes (3).csv
│   └── PrePostFinal (3).csv
├── cleaned data/               # Processed files (output)
│   ├── nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv
│   ├── nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv
│   └── PrePostFinal (3)_selected_columns_cleaned_processed.csv
├── backup/                     # Backup files
├── data_prep.py               # Core preprocessing functions
├── columns.py                 # Main processing script (RUN THIS)
├── describe_csv.py            # CSV analysis utility
├── null_value_analysis.py     # Null value analysis utility
└── README.md                  # This file
```

## 🔧 What the Pipeline Does

1. **Feature Selection**: Uses mutual information and Random Forest to identify the most important features from 500+ columns
2. **Data Enrichment**: Fills missing values in target columns using relationships with important features
3. **Categorical Encoding**: Converts text categories to numbers (Male=1, Female=0, Urban=1, Rural=0)
4. **Data Cleaning**: Normalizes text, calculates derived variables (BMI, diabetes duration, etc.)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ranilmukesh/poornima-ML.git
   cd poornima-ML
   ```

## 📊 Output

The processed files contain 80 carefully selected columns with:
- Encoded categorical variables ready for ML
- Missing values intelligently filled using feature relationships
- Derived health indicators (current_smoking, current_alcohol)
- Calculated metrics (BMI, waist-hip ratio, MET scores)

## 🛠️ Advanced Usage

### Process Individual Files
```python
from data_prep import process_csv_files_enriched

# Process specific files
process_csv_files_enriched(
    [r"raw data\nmbfinalDiabetes (4).csv"],
    output_dir="my_output",
    enrich_with_features=True,
    top_feature_percent=0.5  # Use top 50% features
)
```

### Analyze Data Quality
```python
from null_value_analysis import save_null_and_unique

# Analyze nulls and unique values
save_null_and_unique("cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv")
```

### Describe CSV Statistics
```bash
python describe_csv.py "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
```

## 📋 Requirements

- Python 3.8+
- pandas
- numpy  
- scikit-learn

## 🎯 Next Steps

After running the pipeline, your data is ready for:
- Machine learning model training
- Statistical analysis
- Data visualization
- Further feature engineering

The processed files are optimized for ML algorithms with proper encoding and minimal missing values.