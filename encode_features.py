import pandas as pd
import os

# Define input and output directories
input_dir = "cleaned data"
output_dir = "knn_ready_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the files to process
files = [
    "nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv",
    "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv",
    "PrePostFinal (3)_selected_columns_cleaned_processed.csv"
]

# Encoding mappings
gender_mappings = {
    "nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv": {"Male": 1, "Female": 0, "Transgender": 2},
    "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv": {"Male": 1, "Female": 0},
    "PrePostFinal (3)_selected_columns_cleaned_processed.csv": {"Male": 1, "Female": 0, "Missing": pd.NA}
}

area_mappings = {
    "nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv": {"Urban": 1, "Rural": 0},
    "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv": {"Urban": 1, "Rural": 0},
    "PrePostFinal (3)_selected_columns_cleaned_processed.csv": {"Urban": 1, "Rural": 0, "Missing": pd.NA}
}

for file in files:
    input_path = os.path.join(input_dir, file)
    base_name = os.path.splitext(file)[0]
    output_file = f"{base_name}_knn_ready.csv"
    output_path = os.path.join(output_dir, output_file)
    
    # Load the CSV
    df = pd.read_csv(input_path)
    
    # Encode PreRgender
    if "PreRgender" in df.columns:
        df["PreRgender"] = df["PreRgender"].map(gender_mappings[file]).astype("Int64")
    
    # Encode PreRarea
    if "PreRarea" in df.columns:
        df["PreRarea"] = df["PreRarea"].map(area_mappings[file]).astype("Int64")
    
    # Save the encoded DataFrame
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")