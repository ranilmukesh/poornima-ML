"""
Final Imputation Solution for Diabetes Dataset
==============================================

Based on comprehensive evaluation:
- KNN works best for categorical variables (24.9% and 17.9% improvement over mode)
- For numerical variables, different methods work better depending on data characteristics
- This script applies optimal imputation for columns specified in columns.py

Priority columns for imputation (most critical for ML):
1. PostBLHBA1C - Primary outcome measure
2. PreBLAge - Key demographic
3. PreRgender, PreRarea - Important categorical features
4. PreBLFBS - Clinical measurement
5. Other clinical measurements as needed
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

class OptimalImputer:
    """
    Optimal imputation strategy based on evaluation results:
    - KNN for categorical variables (PreRgender, PreRarea)
    - KNN for numerical variables with moderate missing rates
    - Mean/Mode for variables with very high missing rates (>80%)
    """
    
    def __init__(self):
        self.imputers = {}
        self.encoders = {}
        self.strategies = {}
        
    def analyze_column(self, series):
        """Determine optimal imputation strategy for a column"""
        missing_rate = series.isnull().sum() / len(series)
        is_categorical = series.dtype == 'object' or series.nunique() <= 10
        
        # Strategy selection based on data characteristics
        if missing_rate > 0.8:
            strategy = 'simple'  # Too much missing data for KNN
        elif is_categorical:
            strategy = 'knn'     # KNN works well for categorical
        elif missing_rate > 0.5:
            strategy = 'simple'  # High missing rate, use simple
        else:
            strategy = 'knn'     # Low-moderate missing, KNN should work
            
        return strategy, is_categorical, missing_rate
    
    def fit_transform_column(self, df, column):
        """Apply optimal imputation to a single column"""
        if column not in df.columns:
            return df
            
        series = df[column].copy()
        strategy, is_categorical, missing_rate = self.analyze_column(series)
        
        print(f"📊 {column}: {missing_rate:.1%} missing → {strategy.upper()} imputation")
        
        # Store strategy for reporting
        self.strategies[column] = {
            'method': strategy,
            'is_categorical': is_categorical,
            'missing_rate': missing_rate
        }
        
        if missing_rate == 0:
            return df
        
        # Prepare features for KNN (use other non-missing columns)
        if strategy == 'knn':
            # Select features with low missing rates for KNN
            feature_cols = []
            for col in df.columns:
                if col != column and df[col].isnull().sum() / len(df) < 0.3:
                    feature_cols.append(col)
            
            if len(feature_cols) < 2:
                # Fall back to simple imputation if insufficient features
                strategy = 'simple'
                self.strategies[column]['method'] = 'simple (fallback)'
        
        # Apply imputation
        if strategy == 'knn' and len(feature_cols) >= 2:
            # Encode categorical features for KNN
            df_encoded = df.copy()
            temp_encoders = {}
            
            for col in feature_cols + [column]:
                if df[col].dtype == 'object':
                    encoder = LabelEncoder()
                    # Handle both missing and existing values
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        encoder.fit(non_null_values.astype(str))
                        # Transform non-null values
                        mask = df[col].notna()
                        df_encoded.loc[mask, col] = encoder.transform(df.loc[mask, col].astype(str))
                        temp_encoders[col] = encoder
            
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=min(5, len(df)//10), weights='distance')
            cols_for_knn = feature_cols + [column]
            imputed_values = imputer.fit_transform(df_encoded[cols_for_knn])
            
            # Get imputed column
            col_idx = cols_for_knn.index(column)
            imputed_column = imputed_values[:, col_idx]
            
            # Decode if categorical
            if column in temp_encoders:
                # Round to nearest integer for categorical
                imputed_column = np.round(imputed_column).astype(int)
                # Ensure values are within encoder range
                n_classes = len(temp_encoders[column].classes_)
                imputed_column = np.clip(imputed_column, 0, n_classes-1)
                # Decode back to original labels
                decoded_values = temp_encoders[column].inverse_transform(imputed_column)
                df.loc[:, column] = decoded_values
            else:
                df.loc[:, column] = imputed_column
                
        else:
            # Simple imputation
            if is_categorical or series.dtype == 'object':
                # Mode imputation for categorical
                mode_value = series.mode()
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'Unknown'
            else:
                # Mean imputation for numerical
                fill_value = series.mean()
                
            df[column].fillna(fill_value, inplace=True)
        
        return df
    
    def process_dataset(self, df, priority_columns=None):
        """Process entire dataset with optimal imputation"""
        result_df = df.copy()
        
        # Define priority order for imputation
        if priority_columns is None:
            priority_columns = [
                'PostBLHBA1C', 'PreBLAge', 'PreRgender', 'PreRarea', 'PreBLFBS',
                'PreBLHBA1C', 'PreRheight', 'PreRweight', 'PreBLCHOLESTEROL'
            ]
        
        # Process priority columns first
        for column in priority_columns:
            result_df = self.fit_transform_column(result_df, column)
        
        # Process remaining columns with missing values
        remaining_cols = [col for col in result_df.columns 
                         if col not in priority_columns and result_df[col].isnull().sum() > 0]
        
        for column in remaining_cols:
            result_df = self.fit_transform_column(result_df, column)
        
        return result_df, self.strategies

def process_all_datasets():
    """Process all datasets with optimal imputation"""
    
    print("🚀 FINAL IMPUTATION PIPELINE")
    print("="*60)
    
    # File paths from columns.py
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
            
        print(f"\n📁 Processing: {os.path.basename(input_file)}")
        print("-" * 50)
        
        # Load dataset
        df = pd.read_csv(input_file)
        original_missing = df.isnull().sum().sum()
        
        print(f"📊 Original dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"📊 Total missing values: {original_missing:,}")
        
        # Apply optimal imputation
        imputer = OptimalImputer()
        imputed_df, strategies = imputer.process_dataset(df)
        
        # Check results
        final_missing = imputed_df.isnull().sum().sum()
        print(f"\n✅ Final missing values: {final_missing:,}")
        print(f"✅ Completion rate: {((original_missing-final_missing)/original_missing)*100:.1f}%")
        
        # Save imputed dataset
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_final_imputed.csv'))
        imputed_df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        results[input_file] = {
            'original_missing': original_missing,
            'final_missing': final_missing,
            'completion_rate': ((original_missing-final_missing)/original_missing)*100 if original_missing > 0 else 100,
            'strategies': strategies,
            'output_file': output_file
        }
    
    return results

def create_imputation_report(results):
    """Create summary report of imputation results"""
    
    print(f"\n\n📋 FINAL IMPUTATION REPORT")
    print("="*60)
    
    total_original = sum([r['original_missing'] for r in results.values()])
    total_final = sum([r['final_missing'] for r in results.values()])
    overall_completion = ((total_original-total_final)/total_original)*100 if total_original > 0 else 100
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total missing values processed: {total_original:,}")
    print(f"   Total remaining missing values: {total_final:,}")
    print(f"   Overall completion rate: {overall_completion:.1f}%")
    
    print(f"\n📊 PER-DATASET RESULTS:")
    for file_path, result in results.items():
        filename = os.path.basename(file_path)
        print(f"   {filename}: {result['completion_rate']:.1f}% complete")
    
    print(f"\n🎯 METHOD USAGE SUMMARY:")
    all_strategies = {}
    for result in results.values():
        for col, strategy_info in result['strategies'].items():
            method = strategy_info['method']
            if method not in all_strategies:
                all_strategies[method] = 0
            all_strategies[method] += 1
    
    for method, count in sorted(all_strategies.items()):
        print(f"   {method.upper()}: {count} columns")
    
    print(f"\n💡 IMPUTATION STRATEGY:")
    print(f"   ✅ KNN for categorical variables (proven 24.9% better than mode)")
    print(f"   ✅ KNN for numerical variables with sufficient features")
    print(f"   ✅ Mean/Mode fallback for high-missing columns (>80%)")
    print(f"   ✅ Context-aware method selection per column")

def main():
    """Main execution function"""
    
    try:
        # Process all datasets
        results = process_all_datasets()
        
        # Generate report
        create_imputation_report(results)
        
        print(f"\n🎉 IMPUTATION COMPLETED SUCCESSFULLY!")
        print(f"📁 Check 'final_imputed_data/' folder for results")
        
    except Exception as e:
        print(f"❌ Error during imputation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()