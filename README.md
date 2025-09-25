# Poornima-ML: Diabetes Dataset Preprocessing for Machine Learning

This repository contains Python scripts for preprocessing diabetes-related datasets in preparation for machine learning tasks. The pipeline includes data cleaning, feature encoding, null value analysis, and KNN imputation for handling missing values.

## Project Overview

The project processes three main datasets:
- `nmbfinalDiabetes (4).csv`
- `nmbfinalnewDiabetes (3).csv`
- `PrePostFinal (3).csv`

These datasets contain pre- and post-treatment data for diabetes patients, including demographic information, lifestyle factors, and clinical measurements.

## Data Pipeline

1. **Raw Data**: Located in `raw data/` folder.
2. **Prepared Data**: Initial cleaning and column selection (via `data_prep.py` and `columns.py`).
3. **Cleaned Data**: Further processing and feature engineering.
4. **KNN Ready Data**: Categorical features encoded to numeric values for KNN compatibility.
5. **Imputed Data**: (Future step) Missing values filled using KNN imputer.

## Scripts Overview

- `data_prep.py`: Core preprocessing functions for normalizing text, mapping categories, and feature engineering.
- `encode_features.py`: Encodes categorical columns (`PreRgender`, `PreRarea`) to numeric values for KNN compatibility.
- `null_value_analysis.py`: Analyzes null values and unique values in datasets, saves summaries to CSV/JSON.
- `describe_csv.py`: Provides detailed statistics and samples for CSV files.
- `columns.py`: Selects and saves specific columns from prepared data to cleaned data.
- `test_dietary.py`: Example script to run the full preprocessing pipeline.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/ranilmukesh/poornima-ML.git
   cd poornima-ML
   ```

2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn
   ```

3. Ensure Python 3.8+ is installed.

## Usage

### Running the Full Pipeline

To process all datasets from raw to cleaned:

```python
from data_prep import process_csv_files

process_csv_files([
    r"raw data\nmbfinalDiabetes (4).csv",
    r"raw data\nmbfinalnewDiabetes (3).csv",
    r"raw data\PrePostFinal (3).csv",
])
```

### Encoding Features for KNN

Run the encoding script to prepare data for imputation:

```
python encode_features.py
```

This will create `knn_ready_data/` folder with encoded CSVs.

### Analyzing Null Values

To analyze nulls and uniques for a specific file:

```python
from null_value_analysis import save_null_and_unique

save_null_and_unique("path/to/your/file.csv", output_dir="analysis_output")
```

### Describing a CSV

Get detailed stats:

```
python describe_csv.py "path/to/file.csv" --samples 5
```

## Key Features

- **Text Normalization**: Handles variations in gender, area, and other categorical fields.
- **Ordinal Encoding**: Converts sleep quality, dietary habits, and education to ordinal scales.
- **Binary Conversion**: Transforms yes/no responses to 0/1.
- **Derived Features**: Creates indicators like `current_smoking` and `current_alcohol`.
- **KNN Imputation Ready**: Encodes categoricals to numeric for sklearn's KNNImputer.

## Folder Structure

- `raw data/`: Original CSV files.
- `prepared data/`: Initial cleaned CSVs with selected columns.
- `cleaned data/`: Processed CSVs with feature engineering.
- `knn_ready_data/`: Encoded CSVs ready for imputation.
- `backup/`: Backup of processed files and analyses.
- `null_unique_analysis/`: Null and unique value summaries.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and test thoroughly.
4. Submit a pull request.

## License

This project is for educational/research purposes. Please check data usage permissions.

## Contact

For questions, contact the repository owner or open an issue.