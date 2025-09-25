"""
Root Mean Square Error (RMSE) Analysis for Imputation
====================================================

RMSE is valuable for imputation evaluation because:
1. It's in the same units as the original data (interpretable)
2. Penalizes larger errors more heavily than MAE
3. Standard metric for regression/imputation tasks
4. Easier to compare across different scales
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer, SimpleImputer
import os

def calculate_rmse_for_imputation():
    """Calculate RMSE for our KNN imputation vs baseline methods."""
    
    print("🔍 ROOT MEAN SQUARE ERROR (RMSE) ANALYSIS")
    print("=" * 60)
    
    # Load processed data
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found. Please run the pipeline first.")
        return
    
    df = pd.read_csv(test_file)
    
    # Select key columns for detailed RMSE analysis
    test_columns = {
        'PostBLHBA1C': 'HbA1c Level (%)',
        'PreBLAge': 'Age (years)', 
        'PreRgender': 'Gender (encoded)',
        'PreRarea': 'Area (encoded)',
        'PostRgroupname': 'Treatment Group'
    }
    
    print("🎯 RMSE Analysis for Key Variables:")
    print("=" * 40)
    
    rmse_results = {}
    
    for col, description in test_columns.items():
        if col not in df.columns:
            continue
            
        print(f"\n📊 {col} ({description})")
        print("-" * 50)
        
        # Get complete data
        complete_data = df[df[col].notna()].copy()
        
        if len(complete_data) < 50:
            print(f"   ⚠️  Insufficient data ({len(complete_data)} rows)")
            continue
        
        # Create artificial missingness (20%)
        test_data = complete_data.copy()
        n_missing = int(len(test_data) * 0.2)
        
        np.random.seed(42)
        missing_idx = np.random.choice(test_data.index, n_missing, replace=False)
        true_values = test_data.loc[missing_idx, col].copy()
        test_data.loc[missing_idx, col] = np.nan
        
        # Prepare features for KNN
        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_features if c != col and c in test_data.columns][:8]
        
        if len(feature_cols) == 0:
            print(f"   ⚠️  No suitable features found")
            continue
        
        # Method comparisons
        methods = {}
        
        # 1. KNN Imputation (our method)
        try:
            X = test_data[feature_cols + [col]].copy()
            knn_imputer = KNNImputer(n_neighbors=5)
            X_imputed = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns, index=X.index)
            methods['KNN'] = X_imputed.loc[missing_idx, col]
        except:
            methods['KNN'] = None
        
        # 2. Mean Imputation  
        mean_val = test_data[col].mean()
        methods['Mean'] = pd.Series([mean_val] * len(missing_idx), index=missing_idx)
        
        # 3. Median Imputation
        median_val = test_data[col].median()
        methods['Median'] = pd.Series([median_val] * len(missing_idx), index=missing_idx)
        
        # 4. Mode (for categorical-like data)
        if col in ['PreRgender', 'PreRarea', 'PostRgroupname']:
            mode_val = test_data[col].mode().iloc[0] if not test_data[col].mode().empty else test_data[col].median()
            methods['Mode'] = pd.Series([mode_val] * len(missing_idx), index=missing_idx)
        
        # Calculate metrics for each method
        results = {}
        
        for method_name, predictions in methods.items():
            if predictions is not None:
                try:
                    mse = mean_squared_error(true_values, predictions)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(true_values, predictions)
                    
                    results[method_name] = {
                        'RMSE': rmse,
                        'MSE': mse, 
                        'MAE': mae
                    }
                    
                    print(f"   {method_name:>8}: RMSE={rmse:.4f} | MSE={mse:.4f} | MAE={mae:.4f}")
                except Exception as e:
                    print(f"   {method_name:>8}: Error calculating metrics")
        
        # Find best method and improvements
        if results:
            best_rmse = min([r['RMSE'] for r in results.values()])
            best_method = [name for name, r in results.items() if r['RMSE'] == best_rmse][0]
            
            print(f"\n   🏆 Best Method: {best_method} (RMSE: {best_rmse:.4f})")
            
            if 'KNN' in results:
                knn_rmse = results['KNN']['RMSE']
                
                # Compare KNN vs others
                for method, metrics in results.items():
                    if method != 'KNN':
                        improvement = ((metrics['RMSE'] - knn_rmse) / metrics['RMSE']) * 100
                        status = "📈 Better" if improvement > 0 else "📉 Worse"
                        print(f"   {status}: KNN vs {method} = {improvement:+.1f}% RMSE improvement")
            
            rmse_results[col] = results
        
        # Data context
        if col in df.columns:
            data_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            print(f"   📊 Data Range: {data_stats['min']:.2f} to {data_stats['max']:.2f}")
            print(f"   📊 Mean±Std: {data_stats['mean']:.2f}±{data_stats['std']:.2f}")
    
    return rmse_results

def interpret_rmse_results(rmse_results):
    """Interpret RMSE results in context."""
    
    print(f"\n\n🎯 RMSE INTERPRETATION & CONTEXT")
    print("=" * 50)
    
    print("""
💡 Why RMSE Matters for Imputation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Same Units as Original Data (interpretable)
✅ Penalizes Large Errors More Than Small Ones  
✅ Standard Metric for Regression Tasks
✅ Easy to Compare Across Different Scales
✅ Sensitive to Outliers (shows imputation robustness)
""")
    
    if not rmse_results:
        print("❌ No RMSE results to interpret")
        return
    
    # Overall performance summary
    knn_wins = 0
    total_comparisons = 0
    
    for col, methods in rmse_results.items():
        if 'KNN' in methods:
            knn_rmse = methods['KNN']['RMSE']
            for method_name, metrics in methods.items():
                if method_name != 'KNN':
                    if knn_rmse < metrics['RMSE']:
                        knn_wins += 1
                    total_comparisons += 1
    
    if total_comparisons > 0:
        win_rate = (knn_wins / total_comparisons) * 100
        print(f"📈 KNN Win Rate: {knn_wins}/{total_comparisons} ({win_rate:.1f}%)")
        
        if win_rate >= 70:
            print("🏆 KNN shows STRONG performance!")
        elif win_rate >= 50:
            print("⚖️  KNN shows MIXED performance")  
        else:
            print("⚠️  KNN needs improvement vs baselines")
    
    # Variable-specific insights
    print(f"\n🔬 Variable-Specific RMSE Insights:")
    print("=" * 40)
    
    interpretations = {
        'PostBLHBA1C': {
            'name': 'HbA1c Level',
            'units': '%',
            'good_rmse': 1.0,
            'context': 'Clinical significance: ±0.5% is meaningful'
        },
        'PreBLAge': {
            'name': 'Patient Age', 
            'units': 'years',
            'good_rmse': 5.0,
            'context': '±3-5 years is reasonable for age prediction'
        },
        'PreRgender': {
            'name': 'Gender',
            'units': 'category',
            'good_rmse': 0.3,
            'context': 'Should be nearly perfect for binary categories'
        },
        'PreRarea': {
            'name': 'Urban/Rural',
            'units': 'category', 
            'good_rmse': 0.3,
            'context': 'Geographic categories should be predictable'
        }
    }
    
    for col, methods in rmse_results.items():
        if col in interpretations and 'KNN' in methods:
            info = interpretations[col]
            knn_rmse = methods['KNN']['RMSE']
            
            print(f"\n📊 {info['name']}:")
            print(f"   KNN RMSE: {knn_rmse:.4f} {info['units']}")
            print(f"   Context: {info['context']}")
            
            if knn_rmse <= info['good_rmse']:
                print(f"   ✅ Excellent imputation quality!")
            else:
                print(f"   ⚠️  Consider improving (target: <{info['good_rmse']} {info['units']})")

def main():
    """Run complete RMSE analysis."""
    
    print("🎯 COMPREHENSIVE RMSE ANALYSIS FOR IMPUTATION")
    print("=" * 70)
    
    rmse_results = calculate_rmse_for_imputation()
    interpret_rmse_results(rmse_results)
    
    print(f"\n💡 CONCLUSION:")
    print("=" * 20)
    print("""
RMSE is ESSENTIAL for imputation evaluation because:
• It provides intuitive, interpretable error measurements
• Shows how much our predictions deviate from true values
• Helps identify which variables need imputation improvement
• Enables comparison with domain-specific accuracy thresholds

🎯 Next Steps:
1. If RMSE is high → Consider different imputation methods
2. If RMSE varies by variable → Use column-specific strategies  
3. Compare RMSE to domain requirements (clinical significance)
""")

if __name__ == "__main__":
    main()