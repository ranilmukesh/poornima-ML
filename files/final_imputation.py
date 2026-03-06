"""Optimal Imputation Pipeline for Diabetes Datasets

Applies IterativeImputer (MICE) with RandomForestRegressor for robust
handling of missing data with non-linear feature dependencies.
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

class OptimalImputer:
    """MICE-based imputation strategy using IterativeImputer with Random Forest."""
    
    def __init__(self):
        self.encoders = {}
        self.categorical_columns = []
        self.imputer = None
        
    def process_dataset(self, df):
        """Process entire dataset with IterativeImputer (MICE) using Random Forest."""
        result_df = df.copy()
        original_columns = result_df.columns.tolist()
        
        print(f"[INFO] Starting MICE imputation with Random Forest...")
        print(f"[INFO] Original missing values: {result_df.isnull().sum().sum():,}")
        
        # Step 1: Identify categorical columns
        self.categorical_columns = []
        for col in result_df.columns:
            if result_df[col].dtype == 'object':
                self.categorical_columns.append(col)
        
        print(f"[INFO] Identified {len(self.categorical_columns)} categorical columns")
        
        # Step 2: Encode categorical columns (fit only on non-null values)
        df_encoded = result_df.copy()
        for col in self.categorical_columns:
            non_null_values = df_encoded[col].dropna()
            if len(non_null_values) > 0:
                encoder = LabelEncoder()
                encoder.fit(non_null_values.astype(str))
                self.encoders[col] = encoder
                
                # Transform non-null values, leave NaN as NaN
                mask = df_encoded[col].notna()
                df_encoded.loc[mask, col] = encoder.transform(df_encoded.loc[mask, col].astype(str))
                
                print(f"[INFO] Encoded {col}: {len(encoder.classes_)} unique categories")
        
        # Step 3: Configure and apply IterativeImputer with Random Forest
        print(f"[INFO] Applying IterativeImputer with Random Forest...")
        
        rf_estimator = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
        
        self.imputer = IterativeImputer(
            estimator=rf_estimator,
            max_iter=10,
            random_state=42,
            verbose=0
        )
        
        # Fit and transform
        imputed_array = self.imputer.fit_transform(df_encoded)
        df_imputed = pd.DataFrame(imputed_array, columns=original_columns, index=result_df.index)
        
        # Step 4: Decode categorical columns back to original labels
        print(f"[INFO] Decoding categorical columns...")
        for col in self.categorical_columns:
            if col in self.encoders:
                encoder = self.encoders[col]
                n_classes = len(encoder.classes_)
                
                # Round to nearest integer
                imputed_values = np.round(df_imputed[col].values).astype(int)
                
                # Clip to valid range
                imputed_values = np.clip(imputed_values, 0, n_classes - 1)
                
                # Inverse transform back to original labels
                df_imputed[col] = encoder.inverse_transform(imputed_values)
                
                print(f"[INFO] Decoded {col}")
        
        print(f"[INFO] Final missing values: {df_imputed.isnull().sum().sum():,}")
        
        return df_imputed

def process_all_datasets():
    """Process all preprocessed datasets with optimal imputation."""
    
    print("[PROCESSING] FINAL IMPUTATION PIPELINE")
    print("="*60)
    
    input_files = [
        "temp_processed/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv",
        "temp_processed/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv", 
        "temp_processed/PrePostFinal (3)_selected_columns_cleaned_processed.csv"
    ]
    
    output_dir = "final_imputed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️ File not found: {input_file}")
            continue
            
        print(f"\n[INFO] Processing: {os.path.basename(input_file)}")
        print("-" * 50)
        
        # Load dataset
        df = pd.read_csv(input_file)
        original_missing = df.isnull().sum().sum()
        
        print(f"[DATA] Original dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"[DATA] Total missing values: {original_missing:,}")
        
        # Apply optimal imputation
        imputer = OptimalImputer()
        imputed_df = imputer.process_dataset(df)
        
        # Check results
        final_missing = imputed_df.isnull().sum().sum()
        print(f"\n[SUCCESS] Final missing values: {final_missing:,}")
        print(f"[SUCCESS] Completion rate: {((original_missing-final_missing)/original_missing)*100:.1f}%")
        
        # Save imputed dataset
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_final_imputed.csv'))
        imputed_df.to_csv(output_file, index=False)
        print(f"[SAVED] {output_file}")
        
        results[input_file] = {
            'original_missing': original_missing,
            'final_missing': final_missing,
            'completion_rate': ((original_missing-final_missing)/original_missing)*100 if original_missing > 0 else 100,
            'method': 'IterativeImputer (MICE) with Random Forest',
            'categorical_columns': len(imputer.categorical_columns),
            'output_file': output_file
        }
    
    return results

def create_imputation_report(results):
    """Create summary report of imputation results"""
    
    print(f"\n\n[REPORT] FINAL IMPUTATION REPORT")
    print("="*60)
    
    total_original = sum([r['original_missing'] for r in results.values()])
    total_final = sum([r['final_missing'] for r in results.values()])
    overall_completion = ((total_original-total_final)/total_original)*100 if total_original > 0 else 100
    
    print(f"[STATS] OVERALL STATISTICS:")
    print(f"   Total missing values processed: {total_original:,}")
    print(f"   Total remaining missing values: {total_final:,}")
    print(f"   Overall completion rate: {overall_completion:.1f}%")
    
    print(f"\n[STATS] PER-DATASET RESULTS:")
    for file_path, result in results.items():
        filename = os.path.basename(file_path)
        print(f"   {filename}: {result['completion_rate']:.1f}% complete")
        print(f"      Categorical columns encoded: {result['categorical_columns']}")
    
    print(f"\n[METHOD] IMPUTATION METHOD:")
    print(f"   IterativeImputer (MICE) with Random Forest Regressor")
    print(f"   - Algorithm: Multiple Imputation by Chained Equations")
    print(f"   - Estimator: Random Forest (n_estimators=10, max_depth=5)")
    print(f"   - Iterations: 10 (max_iter=10)")
    print(f"   - Handles non-linear feature dependencies")
    print(f"   - Categorical columns: Label-encoded → imputed → decoded")
    
    print(f"\n[ADVANTAGES] OVER PREVIOUS METHODS:")
    print(f"   - Captures complex feature interactions (Random Forest)")
    print(f"   - No manual priority queue needed (MICE handles dependencies)")
    print(f"   - Better uncertainty modeling through iterative refinement")
    print(f"   - Proven superior for tree-based downstream models")

def main():
    """Main execution function"""
    
    try:
        # Process all datasets
        results = process_all_datasets()
        
        # Generate report
        create_imputation_report(results)
        
        print(f"\n[SUCCESS] IMPUTATION COMPLETED SUCCESSFULLY!")
        print(f"[INFO] Check 'final_imputed_data/' folder for results")
        
    except Exception as e:
        print(f"[ERROR] Error during imputation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  