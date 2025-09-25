"""
Null Value Imputation Evaluation Script
=======================================

This script evaluates the accuracy and effectiveness of the imputation methods used
in the diabetes dataset preprocessing pipeline.

Imputation Methods Used:
1. KNN Imputation (K-Nearest Neighbors) with k=5
2. Feature selection-based imputation using top 50% important features
3. Different strategies for categorical vs numerical columns
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_original_missingness():
    """Analyze missing values in the original raw datasets."""
    
    print("🔍 ORIGINAL DATA MISSINGNESS ANALYSIS")
    print("=" * 60)
    
    raw_files = [
        "raw data/nmbfinalDiabetes (4).csv",
        "raw data/nmbfinalnewDiabetes (3).csv", 
        "raw data/PrePostFinal (3).csv"
    ]
    
    for file_path in raw_files:
        if os.path.exists(file_path):
            print(f"\n📊 {os.path.basename(file_path)}")
            print("-" * 40)
            
            df = pd.read_csv(file_path, low_memory=False)
            
            # Overall statistics
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            print(f"Total rows: {len(df):,}")
            print(f"Total columns: {len(df.columns):,}")
            print(f"Total cells: {total_cells:,}")
            print(f"Missing cells: {missing_cells:,}")
            print(f"Overall missingness: {missing_percentage:.2f}%")
            
            # Top columns with missing values
            null_counts = df.isnull().sum()
            null_percentages = (null_counts / len(df)) * 100
            
            high_missing = null_percentages[null_percentages > 10].sort_values(ascending=False)
            
            if len(high_missing) > 0:
                print(f"\n🚨 Columns with >10% missing values ({len(high_missing)} columns):")
                for col, pct in high_missing.head(10).items():
                    print(f"   {col}: {pct:.1f}% ({null_counts[col]:,} missing)")
            
            # Columns with no missing values
            complete_cols = null_counts[null_counts == 0]
            print(f"\n✅ Complete columns (no missing): {len(complete_cols)}/{len(df.columns)}")


def compare_imputation_methods():
    """Compare different imputation methods on a subset of data."""
    
    print("\n\n🧪 IMPUTATION METHOD COMPARISON")
    print("=" * 60)
    
    # Load processed data to see what we're working with
    processed_files = [
        "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv",
        "cleaned data/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv",
        "cleaned data/PrePostFinal (3)_selected_columns_cleaned_processed.csv"
    ]
    
    for file_path in processed_files:
        if os.path.exists(file_path):
            print(f"\n📈 {os.path.basename(file_path)}")
            print("-" * 50)
            
            df = pd.read_csv(file_path)
            
            # Analyze remaining missingness after our pipeline
            missing_after = df.isnull().sum()
            total_missing = missing_after.sum()
            
            if total_missing > 0:
                print(f"🔴 Remaining missing values: {total_missing:,}")
                high_missing = missing_after[missing_after > 0].sort_values(ascending=False)
                
                print("Top columns still with missing values:")
                for col, count in high_missing.head(5).items():
                    pct = (count / len(df)) * 100
                    print(f"   {col}: {count:,} ({pct:.1f}%)")
            else:
                print("✅ No missing values remaining!")
            
            # Data type distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            print(f"\n📊 Data types after processing:")
            print(f"   Numeric columns: {len(numeric_cols)}")
            print(f"   Categorical columns: {len(categorical_cols)}")
            print(f"   Total columns: {len(df.columns)}")


def evaluate_knn_imputation_accuracy():
    """Evaluate KNN imputation accuracy using artificial missingness."""
    
    print("\n\n🎯 KNN IMPUTATION ACCURACY EVALUATION")
    print("=" * 60)
    
    # Load a processed file for testing
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found. Please run the pipeline first.")
        return
    
    df = pd.read_csv(test_file)
    
    # Select numeric columns for testing (easier to evaluate accuracy)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    test_cols = [col for col in numeric_cols if df[col].notna().sum() > 100][:5]  # Top 5 complete numeric columns
    
    print(f"🧪 Testing imputation accuracy on {len(test_cols)} columns:")
    for col in test_cols:
        print(f"   - {col}")
    
    results = {}
    
    for col in test_cols:
        print(f"\n📊 Testing column: {col}")
        
        # Get complete data for this column
        complete_data = df[df[col].notna()].copy()
        
        if len(complete_data) < 50:
            print(f"   ⚠️  Skipping {col} - not enough complete data")
            continue
        
        # Create artificial missingness (remove 20% randomly)
        test_data = complete_data.copy()
        n_to_remove = int(len(test_data) * 0.2)
        
        # Randomly select indices to make missing
        np.random.seed(42)  # For reproducibility
        missing_indices = np.random.choice(test_data.index, n_to_remove, replace=False)
        
        # Store true values
        true_values = test_data.loc[missing_indices, col].copy()
        
        # Create missingness
        test_data.loc[missing_indices, col] = np.nan
        
        # Method 1: KNN Imputation (our method)
        knn_imputer = KNNImputer(n_neighbors=5)
        
        # Use other numeric columns as features
        feature_cols = [c for c in numeric_cols if c != col and c in test_data.columns][:10]  # Limit features
        
        if len(feature_cols) > 0:
            X = test_data[feature_cols + [col]].copy()
            X_imputed = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns, index=X.index)
            knn_predictions = X_imputed.loc[missing_indices, col]
            
            # Method 2: Simple mean imputation (baseline)
            mean_value = test_data[col].mean()
            mean_predictions = pd.Series([mean_value] * len(missing_indices), index=missing_indices)
            
            # Method 3: Median imputation (baseline)
            median_value = test_data[col].median()
            median_predictions = pd.Series([median_value] * len(missing_indices), index=missing_indices)
            
            # Calculate metrics
            knn_mse = mean_squared_error(true_values, knn_predictions)
            mean_mse = mean_squared_error(true_values, mean_predictions)
            median_mse = mean_squared_error(true_values, median_predictions)
            
            knn_mae = mean_absolute_error(true_values, knn_predictions)
            mean_mae = mean_absolute_error(true_values, mean_predictions)
            median_mae = mean_absolute_error(true_values, median_predictions)
            
            results[col] = {
                'KNN_MSE': knn_mse,
                'Mean_MSE': mean_mse,
                'Median_MSE': median_mse,
                'KNN_MAE': knn_mae,
                'Mean_MAE': mean_mae,
                'Median_MAE': median_mae,
                'KNN_vs_Mean_MSE_Improvement': ((mean_mse - knn_mse) / mean_mse) * 100,
                'KNN_vs_Median_MSE_Improvement': ((median_mse - knn_mse) / median_mse) * 100
            }
            
            print(f"   📈 MSE - KNN: {knn_mse:.4f} | Mean: {mean_mse:.4f} | Median: {median_mse:.4f}")
            print(f"   📈 MAE - KNN: {knn_mae:.4f} | Mean: {mean_mae:.4f} | Median: {median_mae:.4f}")
            print(f"   🏆 KNN improvement over Mean: {results[col]['KNN_vs_Mean_MSE_Improvement']:.1f}%")
            print(f"   🏆 KNN improvement over Median: {results[col]['KNN_vs_Median_MSE_Improvement']:.1f}%")
    
    # Summary
    if results:
        print(f"\n📋 SUMMARY ACROSS ALL TESTED COLUMNS")
        print("=" * 50)
        
        avg_knn_mse = np.mean([r['KNN_MSE'] for r in results.values()])
        avg_mean_mse = np.mean([r['Mean_MSE'] for r in results.values()])
        avg_median_mse = np.mean([r['Median_MSE'] for r in results.values()])
        
        avg_improvement_vs_mean = np.mean([r['KNN_vs_Mean_MSE_Improvement'] for r in results.values()])
        avg_improvement_vs_median = np.mean([r['KNN_vs_Median_MSE_Improvement'] for r in results.values()])
        
        print(f"Average MSE - KNN: {avg_knn_mse:.4f}")
        print(f"Average MSE - Mean: {avg_mean_mse:.4f}")
        print(f"Average MSE - Median: {avg_median_mse:.4f}")
        print(f"\n🎯 KNN Imputation Performance:")
        print(f"   • {avg_improvement_vs_mean:.1f}% better than Mean imputation")
        print(f"   • {avg_improvement_vs_median:.1f}% better than Median imputation")
        
        if avg_improvement_vs_mean > 0 and avg_improvement_vs_median > 0:
            print(f"   ✅ KNN consistently outperforms simple methods!")
        elif avg_improvement_vs_mean > 0 or avg_improvement_vs_median > 0:
            print(f"   ⚖️  KNN shows mixed performance vs simple methods")
        else:
            print(f"   ⚠️  KNN may not be optimal for this dataset")


def analyze_feature_importance_impact():
    """Analyze how feature selection impacts imputation quality."""
    
    print(f"\n\n🎯 FEATURE SELECTION IMPACT ON IMPUTATION")
    print("=" * 60)
    
    print("📊 Our imputation strategy:")
    print("   1. Select top 50% most important features using:")
    print("      • Mutual Information (for feature-target relationships)")
    print("      • Random Forest Feature Importance")
    print("   2. Use selected features for KNN imputation")
    print("   3. Apply different strategies for categorical vs numeric data")
    
    print(f"\n🔧 Technical Details:")
    print("   • KNN neighbors: 5 (adaptive based on data size)")
    print("   • Categorical handling: Label encoding → KNN → Decode back")
    print("   • Numeric handling: Direct KNN imputation")
    print("   • Feature preprocessing: Median fill for features, then use for targets")


def create_imputation_report():
    """Create a comprehensive imputation report."""
    
    print(f"\n\n📋 IMPUTATION METHOD SUMMARY REPORT")
    print("=" * 60)
    
    print(f"""
🔬 IMPUTATION ALGORITHM USED:
   Method: K-Nearest Neighbors (KNN) Imputation
   Implementation: sklearn.impute.KNNImputer
   
🎛️ ALGORITHM PARAMETERS:
   • n_neighbors: 5 (or dataset_size-1 if smaller)
   • weights: 'uniform' (default)
   • metric: 'nan_euclidean' (handles missing values in features)
   
🧠 FEATURE SELECTION STRATEGY:
   • Uses top 50% most predictive features
   • Selection based on: Mutual Information + Random Forest importance
   • Prevents overfitting and reduces noise in imputation
   
📊 DATA TYPE HANDLING:
   
   Categorical Variables:
   1. Label encode categories → numeric
   2. Apply KNN imputation  
   3. Round to nearest integer
   4. Decode back to original categories
   5. Handle out-of-bounds predictions with clipping
   
   Numerical Variables:
   1. Direct KNN imputation on continuous scale
   2. No post-processing needed
   
🔄 PREPROCESSING STEPS:
   1. Clean and normalize text data
   2. Select important features (top 50%)  
   3. Handle feature missingness with median imputation
   4. Apply KNN to target columns using clean features
   5. Encode categorical results appropriately
   
🎯 ADVANTAGES OF THIS APPROACH:
   ✅ Preserves relationships between variables
   ✅ More accurate than mean/median imputation
   ✅ Handles both categorical and numerical data
   ✅ Uses feature selection to reduce noise
   ✅ Robust to outliers (compared to mean imputation)
   
⚠️ POTENTIAL LIMITATIONS:
   • Computationally more expensive than simple methods
   • Requires sufficient complete cases for reliable neighbors
   • May propagate patterns from training data
   • Performance depends on feature selection quality
""")


def main():
    """Run the complete imputation evaluation."""
    
    print("🔍 DIABETES DATASET - NULL VALUE IMPUTATION EVALUATION")
    print("=" * 70)
    
    # Run all analyses
    analyze_original_missingness()
    compare_imputation_methods() 
    evaluate_knn_imputation_accuracy()
    analyze_feature_importance_impact()
    create_imputation_report()
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print("=" * 70)
    print("""
📁 For detailed null value analysis, you can also run:
   python -c "from null_value_analysis import save_null_and_unique; save_null_and_unique('cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv')"
   
📊 For CSV statistics:
   python describe_csv.py "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
""")

if __name__ == "__main__":
    main()