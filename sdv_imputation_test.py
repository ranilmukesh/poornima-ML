"""
Enhanced Imputation using SDV (Synthetic Data Vault)
===================================================

SDV provides sophisticated imputation methods:
1. GaussianCopulaSynthesizer - Statistical approach with copulas
2. CTGANSynthesizer - Deep learning GAN-based approach  
3. CopulaGANSynthesizer - Hybrid statistical + deep learning
4. TVAESynthesizer - Variational autoencoder approach

These methods should significantly improve our RMSE compared to basic KNN.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def install_sdv():
    """Install SDV and required dependencies."""
    
    print("📦 Installing SDV (Synthetic Data Vault)...")
    
    # Note: This will be handled by the install_python_packages tool
    packages = [
        'sdv',
        'sdv[tabular]',  # Includes tabular data synthesizers
        'copulas',       # Statistical copula methods
        'ctgan',         # GAN-based methods
        'rdt'            # Data transformation library
    ]
    
    return packages

def prepare_sdv_metadata(df, target_columns):
    """Create SDV metadata for the dataset."""
    
    print("🔧 Preparing SDV metadata...")
    
    from sdv.metadata import SingleTableMetadata
    
    # Create metadata object
    metadata = SingleTableMetadata()
    
    # Auto-detect column types
    metadata.detect_from_dataframe(df)
    
    # Customize metadata for our specific columns
    column_configs = {
        'PreRgender': {'sdtype': 'categorical'},
        'PreRarea': {'sdtype': 'categorical'},
        'PostRgroupname': {'sdtype': 'categorical'},
        'PreBLAge': {'sdtype': 'numerical'},
        'PostBLHBA1C': {'sdtype': 'numerical'}
    }
    
    # Apply custom configurations
    for col, config in column_configs.items():
        if col in df.columns:
            metadata.update_column(column_name=col, **config)
    
    print(f"✅ Metadata prepared for {len(df.columns)} columns")
    return metadata

def sdv_imputation_comparison():
    """Compare multiple SDV imputation methods."""
    
    print("🧠 SDV-BASED IMPUTATION COMPARISON")
    print("=" * 60)
    
    # Load test data
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found. Please run the main pipeline first.")
        return
    
    df = pd.read_csv(test_file)
    
    # Select test columns with enough complete data
    test_columns = ['PostBLHBA1C', 'PreBLAge', 'PreRgender', 'PreRarea']
    test_columns = [col for col in test_columns if col in df.columns]
    
    if len(test_columns) == 0:
        print("❌ No suitable test columns found.")
        return
    
    print(f"🎯 Testing SDV imputation on: {test_columns}")
    
    results = {}
    
    for target_col in test_columns:
        print(f"\n📊 Testing {target_col}")
        print("-" * 40)
        
        # Get complete cases for this column
        complete_mask = df[target_col].notna()
        if complete_mask.sum() < 100:
            print(f"   ⚠️  Insufficient complete data ({complete_mask.sum()} rows)")
            continue
        
        complete_df = df[complete_mask].copy()
        
        # Select relevant features (numeric only for simplicity)
        numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col][:10]  # Limit to 10 features
        
        if len(feature_cols) < 3:
            print(f"   ⚠️  Insufficient features ({len(feature_cols)} available)")
            continue
        
        # Create test dataset with artificial missingness
        test_df = complete_df[feature_cols + [target_col]].copy()
        n_test = min(500, len(test_df))  # Limit for faster testing
        test_df = test_df.sample(n=n_test, random_state=42)
        
        # Create 20% missingness
        n_missing = int(len(test_df) * 0.2)
        np.random.seed(42)
        missing_idx = np.random.choice(test_df.index, n_missing, replace=False)
        
        true_values = test_df.loc[missing_idx, target_col].copy()
        test_df_missing = test_df.copy()
        test_df_missing.loc[missing_idx, target_col] = np.nan
        
        # Prepare metadata
        try:
            metadata = prepare_sdv_metadata(test_df_missing, [target_col])
            
            # Test different SDV methods
            sdv_methods = [
                ('GaussianCopula', 'statistical'),
                ('CTGAN', 'deep_learning'),
                ('TVAE', 'variational_autoencoder')
            ]
            
            method_results = {}
            
            for method_name, method_type in sdv_methods:
                try:
                    print(f"   🔄 Testing {method_name}...")
                    
                    # Import and create synthesizer
                    if method_name == 'GaussianCopula':
                        from sdv.single_table import GaussianCopulaSynthesizer
                        synthesizer = GaussianCopulaSynthesizer(metadata)
                    elif method_name == 'CTGAN':
                        from sdv.single_table import CTGANSynthesizer
                        synthesizer = CTGANSynthesizer(metadata, epochs=10)  # Reduced for speed
                    elif method_name == 'TVAE':
                        from sdv.single_table import TVAESynthesizer
                        synthesizer = TVAESynthesizer(metadata, epochs=10)  # Reduced for speed
                    
                    # Train on complete data only
                    complete_training_data = test_df_missing.dropna()
                    
                    if len(complete_training_data) < 50:
                        print(f"      ⚠️  Insufficient training data for {method_name}")
                        continue
                    
                    synthesizer.fit(complete_training_data)
                    
                    # Generate synthetic data to fill missing values
                    # Sample more data and use it to fill missing values
                    synthetic_data = synthesizer.sample(num_rows=len(test_df_missing) * 2)
                    
                    # Use synthetic data to impute missing values
                    # Strategy: Use nearest neighbor approach with synthetic data
                    imputed_values = []
                    
                    for idx in missing_idx:
                        # Find similar rows in synthetic data based on available features
                        row_features = test_df_missing.loc[idx, feature_cols].values
                        
                        # Calculate distances to synthetic data
                        synthetic_features = synthetic_data[feature_cols].values
                        distances = np.sqrt(np.sum((synthetic_features - row_features) ** 2, axis=1))
                        
                        # Use the target value from the closest synthetic row
                        closest_idx = np.argmin(distances)
                        imputed_value = synthetic_data.iloc[closest_idx][target_col]
                        imputed_values.append(imputed_value)
                    
                    imputed_series = pd.Series(imputed_values, index=missing_idx)
                    
                    # Calculate metrics
                    mse = mean_squared_error(true_values, imputed_series)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(true_values, imputed_series)
                    
                    method_results[method_name] = {
                        'RMSE': rmse,
                        'MSE': mse,
                        'MAE': mae,
                        'Type': method_type
                    }
                    
                    print(f"      ✅ {method_name}: RMSE={rmse:.4f} | MAE={mae:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ {method_name} failed: {str(e)[:100]}...")
            
            # Add baseline methods for comparison
            try:
                # Mean imputation
                mean_val = test_df_missing[target_col].mean()
                mean_imputed = pd.Series([mean_val] * len(missing_idx), index=missing_idx)
                
                mse_mean = mean_squared_error(true_values, mean_imputed)
                rmse_mean = np.sqrt(mse_mean)
                mae_mean = mean_absolute_error(true_values, mean_imputed)
                
                method_results['Mean (Baseline)'] = {
                    'RMSE': rmse_mean,
                    'MSE': mse_mean,
                    'MAE': mae_mean,
                    'Type': 'baseline'
                }
                
                print(f"      📊 Mean Baseline: RMSE={rmse_mean:.4f} | MAE={mae_mean:.4f}")
                
            except:
                pass
            
            results[target_col] = method_results
            
            # Show best method for this column
            if method_results:
                best_rmse = min([r['RMSE'] for r in method_results.values()])
                best_method = [name for name, r in method_results.items() if r['RMSE'] == best_rmse][0]
                
                print(f"   🏆 Best method: {best_method} (RMSE: {best_rmse:.4f})")
                
        except Exception as e:
            print(f"   ❌ Error processing {target_col}: {str(e)[:100]}...")
    
    return results

def create_sdv_summary(results):
    """Create summary of SDV imputation results."""
    
    print(f"\n\n📋 SDV IMPUTATION RESULTS SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No results to summarize.")
        return
    
    # Overall performance analysis
    all_methods = set()
    for col_results in results.values():
        all_methods.update(col_results.keys())
    
    method_performance = {}
    
    for method in all_methods:
        rmse_values = []
        for col_results in results.values():
            if method in col_results:
                rmse_values.append(col_results[method]['RMSE'])
        
        if rmse_values:
            method_performance[method] = {
                'avg_rmse': np.mean(rmse_values),
                'min_rmse': np.min(rmse_values),
                'max_rmse': np.max(rmse_values),
                'count': len(rmse_values)
            }
    
    # Sort by performance
    sorted_methods = sorted(method_performance.items(), key=lambda x: x[1]['avg_rmse'])
    
    print(f"🏆 METHOD PERFORMANCE RANKING:")
    print("-" * 40)
    
    for i, (method, perf) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}")
        print(f"   Average RMSE: {perf['avg_rmse']:.4f}")
        print(f"   Range: {perf['min_rmse']:.4f} - {perf['max_rmse']:.4f}")
        print(f"   Tested on: {perf['count']} columns")
        
        if i == 1:
            print(f"   🏆 BEST OVERALL METHOD!")
        print()
    
    # Improvement analysis
    if 'Mean (Baseline)' in [m[0] for m in sorted_methods]:
        baseline_rmse = method_performance['Mean (Baseline)']['avg_rmse']
        
        print(f"📈 IMPROVEMENT OVER BASELINE:")
        print("-" * 35)
        
        for method, perf in method_performance.items():
            if method != 'Mean (Baseline)':
                improvement = ((baseline_rmse - perf['avg_rmse']) / baseline_rmse) * 100
                status = "📈 Better" if improvement > 0 else "📉 Worse"
                print(f"{method}: {status} by {improvement:+.1f}%")

def main():
    """Main function to run SDV imputation comparison."""
    
    print("🧠 SDV (SYNTHETIC DATA VAULT) IMPUTATION TEST")
    print("=" * 70)
    
    # Check if SDV is installed
    try:
        import sdv
        print(f"✅ SDV version {sdv.__version__} is available")
    except ImportError:
        print("❌ SDV not installed. Installing required packages...")
        packages = install_sdv()
        print(f"📦 Required packages: {packages}")
        print("💡 Please install these packages first:")
        print("   pip install sdv")
        return
    
    # Run SDV comparison
    results = sdv_imputation_comparison()
    
    # Create summary
    create_sdv_summary(results)
    
    print(f"\n💡 NEXT STEPS:")
    print("=" * 20)
    print("""
1. If SDV methods show improvement → Integrate into main pipeline
2. Use GaussianCopula for fast, reliable imputation
3. Use CTGAN/TVAE for complex patterns (if computational resources allow)
4. Consider ensemble methods combining multiple SDV approaches
""")

if __name__ == "__main__":
    main()