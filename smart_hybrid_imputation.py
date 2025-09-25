"""
Smart Hybrid Imputation System
==============================

Based on our analysis:
- KNN: Best for categorical variables (16-39% better than median)
- Simple methods: Competitive for numerical variables  
- SDV: Needs optimization for this specific dataset

Solution: Create an intelligent imputation system that chooses the best method per variable type and context.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

class SmartHybridImputer:
    """Intelligent imputation system that chooses optimal methods per variable."""
    
    def __init__(self):
        self.imputation_strategies = {}
        self.fitted_imputers = {}
        self.label_encoders = {}
        
    def analyze_and_recommend_strategy(self, df, target_col):
        """Analyze a column and recommend the best imputation strategy."""
        
        if target_col not in df.columns:
            return None
            
        # Get basic statistics
        total_values = len(df)
        missing_values = df[target_col].isnull().sum()
        missing_rate = missing_values / total_values
        unique_values = df[target_col].nunique()
        
        # Determine data type
        if df[target_col].dtype in ['object', 'category']:
            data_type = 'categorical'
        else:
            data_type = 'numerical'
        
        # Recommend strategy based on analysis
        strategy = {
            'column': target_col,
            'data_type': data_type,
            'missing_rate': missing_rate,
            'unique_values': unique_values,
            'total_values': total_values
        }
        
        if data_type == 'categorical':
            if unique_values <= 5 and missing_rate < 0.3:
                strategy['method'] = 'knn'  # KNN works well for low-cardinality categorical
                strategy['reason'] = 'Low cardinality categorical with moderate missingness'
            elif missing_rate > 0.5:
                strategy['method'] = 'mode'  # Too sparse for KNN
                strategy['reason'] = 'High missingness rate, mode imputation safer'
            else:
                strategy['method'] = 'knn'
                strategy['reason'] = 'Categorical variable suitable for KNN'
        
        else:  # numerical
            if missing_rate < 0.1:
                strategy['method'] = 'knn'  # KNN for low missingness
                strategy['reason'] = 'Low missingness, KNN can preserve relationships'
            elif missing_rate > 0.4:
                strategy['method'] = 'median'  # Too sparse for KNN
                strategy['reason'] = 'High missingness, median imputation more robust'
            else:
                # Check if normally distributed
                skewness = df[target_col].skew()
                if abs(skewness) < 1:
                    strategy['method'] = 'mean'
                    strategy['reason'] = 'Normally distributed, mean imputation appropriate'
                else:
                    strategy['method'] = 'median'
                    strategy['reason'] = 'Skewed distribution, median more robust'
        
        return strategy
    
    def fit_and_impute_column(self, df, target_col, feature_cols=None, k_neighbors=5):
        """Fit and impute a specific column using the recommended strategy."""
        
        strategy = self.analyze_and_recommend_strategy(df, target_col)
        if not strategy:
            return df.copy()
        
        print(f"🎯 {target_col} ({strategy['data_type']})")
        print(f"   Strategy: {strategy['method']} - {strategy['reason']}")
        print(f"   Missing: {strategy['missing_rate']*100:.1f}% ({df[target_col].isnull().sum():,} values)")
        
        df_result = df.copy()
        method = strategy['method']
        
        if method == 'knn':
            # KNN imputation with feature selection
            if feature_cols is None:
                # Auto-select features
                if strategy['data_type'] == 'categorical':
                    # For categorical targets, use other categorical + some numerical
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    
                    feature_cols = [col for col in cat_cols if col != target_col][:5]
                    feature_cols.extend([col for col in num_cols if col != target_col][:5])
                else:
                    # For numerical targets, prefer numerical features
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    feature_cols = [col for col in num_cols if col != target_col][:10]
            
            # Prepare data for KNN
            if len(feature_cols) == 0:
                print(f"   ⚠️  No suitable features found, falling back to mode/median")
                method = 'mode' if strategy['data_type'] == 'categorical' else 'median'
            else:
                print(f"   📊 Using {len(feature_cols)} features for KNN imputation")
                
                # Handle categorical encoding for KNN
                work_df = df[feature_cols + [target_col]].copy()
                
                # Encode categorical columns
                categorical_features = work_df.select_dtypes(include=['object', 'category']).columns
                
                for col in categorical_features:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        
                    # Fit on non-null values
                    non_null_mask = work_df[col].notna()
                    if non_null_mask.sum() > 0:
                        self.label_encoders[col].fit(work_df.loc[non_null_mask, col].astype(str))
                        work_df.loc[non_null_mask, col] = self.label_encoders[col].transform(
                            work_df.loc[non_null_mask, col].astype(str)
                        )
                
                # Apply KNN imputation
                try:
                    imputer = KNNImputer(n_neighbors=min(k_neighbors, len(work_df.dropna())-1))
                    imputed_data = imputer.fit_transform(work_df)
                    imputed_df = pd.DataFrame(imputed_data, columns=work_df.columns, index=work_df.index)
                    
                    # Decode categorical target if needed
                    if target_col in categorical_features:
                        # Round and clip to valid categories
                        imputed_values = np.round(imputed_df[target_col]).astype(int)
                        n_categories = len(self.label_encoders[target_col].classes_)
                        imputed_values = np.clip(imputed_values, 0, n_categories-1)
                        
                        # Decode back to original categories
                        decoded_values = self.label_encoders[target_col].inverse_transform(imputed_values)
                        df_result[target_col] = decoded_values
                    else:
                        df_result[target_col] = imputed_df[target_col]
                    
                    print(f"   ✅ KNN imputation completed")
                    
                except Exception as e:
                    print(f"   ❌ KNN failed: {str(e)[:50]}... Falling back to simple method")
                    method = 'mode' if strategy['data_type'] == 'categorical' else 'median'
        
        # Simple imputation methods (fallback or primary choice)
        if method in ['mean', 'median', 'mode']:
            missing_mask = df_result[target_col].isnull()
            
            if method == 'mean':
                fill_value = df_result[target_col].mean()
            elif method == 'median':
                fill_value = df_result[target_col].median()
            elif method == 'mode':
                mode_result = df_result[target_col].mode()
                fill_value = mode_result.iloc[0] if len(mode_result) > 0 else 'Unknown'
            
            df_result.loc[missing_mask, target_col] = fill_value
            print(f"   ✅ {method.title()} imputation: filled with {fill_value}")
        
        # Store strategy for reference
        self.imputation_strategies[target_col] = strategy
        
        return df_result
    
    def process_dataset(self, df, priority_columns=None):
        """Process entire dataset with smart imputation strategies."""
        
        print("🧠 SMART HYBRID IMPUTATION SYSTEM")
        print("=" * 60)
        
        result_df = df.copy()
        
        # Determine processing order
        if priority_columns:
            columns_to_process = [col for col in priority_columns if col in df.columns]
            # Add remaining columns
            remaining_cols = [col for col in df.columns if col not in columns_to_process and df[col].isnull().sum() > 0]
            columns_to_process.extend(remaining_cols)
        else:
            # Process by missing rate (least missing first)
            missing_rates = df.isnull().sum() / len(df)
            columns_to_process = missing_rates[missing_rates > 0].sort_values().index.tolist()
        
        print(f"📋 Processing {len(columns_to_process)} columns with missing values")
        
        for col in columns_to_process:
            result_df = self.fit_and_impute_column(result_df, col)
        
        # Summary
        original_missing = df.isnull().sum().sum()
        final_missing = result_df.isnull().sum().sum()
        
        print(f"\n📊 IMPUTATION SUMMARY:")
        print(f"   Original missing values: {original_missing:,}")
        print(f"   Final missing values: {final_missing:,}")
        print(f"   Imputation rate: {((original_missing - final_missing) / original_missing * 100):.1f}%")
        
        return result_df, self.imputation_strategies

def test_hybrid_imputation():
    """Test the hybrid imputation system."""
    
    print("🎯 TESTING SMART HYBRID IMPUTATION")
    print("=" * 60)
    
    # Load test data
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found.")
        return
    
    df = pd.read_csv(test_file)
    
    # Test on key columns
    priority_columns = ['PostBLHBA1C', 'PreBLAge', 'PreRgender', 'PreRarea']
    
    # Initialize hybrid imputer
    imputer = SmartHybridImputer()
    
    # Process dataset
    result_df, strategies = imputer.process_dataset(df, priority_columns)
    
    # Analyze results
    print(f"\n📋 STRATEGIES USED:")
    print("-" * 30)
    
    for col, strategy in strategies.items():
        print(f"{col}:")
        print(f"   Method: {strategy['method']}")
        print(f"   Reason: {strategy['reason']}")
        print(f"   Missing rate: {strategy['missing_rate']*100:.1f}%")
    
    return result_df, strategies

if __name__ == "__main__":
    test_hybrid_imputation()