#!/usr/bin/env python3
"""
PyCaret AutoML for Diabetes HbA1c Prediction
============================================
Target: PostBLHBA1C prediction with MAE < 0.5
Strategy: Four-model approach (3 individual datasets + 1 combined)
Approach: PyCaret AutoML with Stack Ridge ensemble
Environment: Google Colab with GPU acceleration
"""

# Installation command for Google Colab (uncomment if needed)
# !pip install pycaret[full] pandas scikit-learn

import pandas as pd
import numpy as np
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*70)
print("🚀 PyCaret AutoML for Diabetes HbA1c Prediction")
print("="*70)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Target: PostBLHBA1C prediction with MAE < 0.5")
print("📊 Strategy: Four-model approach (3 individual + 1 combined)")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_column_names(df):
    """Cleans column names for consistency."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def load_and_check_data(file_paths):
    """Loads, concatenates, and checks data files."""
    data_frames = []
    print(f"\n--- Loading Data ---")
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found at {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
            df = clean_column_names(df)
            
            if 'PostBLHBA1C' in df.columns:
                df['PostBLHBA1C'] = pd.to_numeric(df['PostBLHBA1C'], errors='coerce')
            else:
                print(f"WARNING: Target 'PostBLHBA1C' not in {file_path}. Skipping file.")
                continue
                 
            data_frames.append(df)
            print(f"✓ Successfully loaded {os.path.basename(file_path)} ({len(df)} rows)")
        except Exception as e:
            print(f"ERROR: Error loading {file_path}: {e}")
            
    if not data_frames:
        print("ERROR: No data files were successfully loaded. Exiting.")
        return None
        
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=['PostBLHBA1C'])
    print(f"✓ Loaded and combined {len(data_frames)} files. Total rows: {len(combined_df)}")
    
    # Data validation
    print(f"\n[VALIDATION] Checking data quality...")
    print(f"  • Rows: {len(combined_df)}")
    print(f"  • Columns: {len(combined_df.columns)}")
    print(f"  • Missing values: {combined_df.isnull().sum().sum()}")
    print(f"  • Target mean: {combined_df['PostBLHBA1C'].mean():.4f}")
    print(f"  • Target std: {combined_df['PostBLHBA1C'].std():.4f}")
    
    # Remove rows with all NaN (except target)
    feature_cols = [col for col in combined_df.columns if col != 'PostBLHBA1C']
    combined_df = combined_df[~combined_df[feature_cols].isna().all(axis=1)]
    
    print(f"✓ Data validation complete. Final rows: {len(combined_df)}")
    return combined_df

def train_pycaret_model(data, dataset_name, model_dir):
    """Train PyCaret AutoML model with Stack Ridge ensemble."""
    from pycaret.regression import setup, compare_models, stack_models, finalize_model, save_model, pull, predict_model
    
    print(f"\n{'='*70}")
    print(f"--- Training PyCaret on {dataset_name} Dataset ---")
    print(f"{'='*70}")
    
    try:
        # Setup PyCaret environment
        print("\n[SETUP] Initializing PyCaret environment...")
        try:
            exp = setup(
                data=data,
                target='PostBLHBA1C',
                session_id=42,
                use_gpu=True,
                imputation_type=None,  # Skip imputation (data is pre-imputed)
                normalize=True,
                transformation=False,  # Disabled due to optimization issues
                remove_multicollinearity=True,
                fold=10,
                train_size=0.8,
                verbose=False,
                html=False,
                log_experiment=False,
                system_log=False
            )
            print("✓ PyCaret environment initialized successfully")
        except Exception as setup_error:
            print(f"⚠️  Setup error with standard settings: {setup_error}")
            print("Retrying with minimal preprocessing...")
            exp = setup(
                data=data,
                target='PostBLHBA1C',
                session_id=42,
                use_gpu=False,  # Disable GPU to avoid compatibility issues
                imputation_type=None,
                normalize=False,
                transformation=False,
                remove_multicollinearity=False,
                fold=10,
                train_size=0.8,
                verbose=False,
                html=False,
                log_experiment=False,
                system_log=False
            )
            print("✓ PyCaret environment initialized with minimal preprocessing")
        
        # Compare models
        print("\n[MODELING] Comparing models (sorting by MAE)...")
        include_models = ['et', 'rf', 'gbr', 'xgboost',  # Removed 'lightgbm' due to GPU/data issues
                         'catboost', 'ridge', 'enet', 'huber', 'dt', 'ada', 'knn']
        
        best_models = compare_models(
            include=include_models,
            sort='MAE',
            n_select=3,
            fold=10,
            verbose=False
        )
        print(f"✓ Top 3 models selected based on MAE")
        
        # Stack models with Ridge meta-learner
        print("\n[STACKING] Creating stacked ensemble with Ridge meta-learner...")
        stacked_model = stack_models(
            estimator_list=best_models,
            meta_model='ridge',
            fold=5,
            verbose=False
        )
        print("✓ Stacked model created successfully")
        
        # Finalize model
        print("\n[FINALIZING] Training final model on full training set...")
        final_model = finalize_model(stacked_model)
        print("✓ Final model trained successfully")
        
        # Evaluation - Cross-Validation Metrics
        print(f"\n{'='*70}")
        print("--- Cross-Validation Metrics (10-Fold, Stacked Model) ---")
        print(f"{'='*70}")
        cv_metrics_df = pull()
        
        # Find stacking model metrics
        stacking_row = cv_metrics_df[cv_metrics_df.index.str.contains('Stacking', case=False, na=False)]
        if not stacking_row.empty:
            cv_mae = stacking_row['MAE'].values[0]
            cv_rmse = stacking_row['RMSE'].values[0]
            cv_r2 = stacking_row['R2'].values[0]
        else:
            cv_mae = cv_metrics_df['MAE'].iloc[0]
            cv_rmse = cv_metrics_df['RMSE'].iloc[0]
            cv_r2 = cv_metrics_df['R2'].iloc[0]
        
        print(f"CV MAE:  {cv_mae:.4f}")
        print(f"CV RMSE: {cv_rmse:.4f}")
        print(f"CV R²:   {cv_r2:.4f}")
        
        # Evaluation - Test Set Metrics
        print(f"\n{'='*70}")
        print("--- Test Set Metrics (20% Hold-out, Final Model) ---")
        print(f"{'='*70}")
        holdout_pred = predict_model(final_model)
        
        # Calculate test metrics manually from predictions
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_true = holdout_pred['PostBLHBA1C']
        y_pred = holdout_pred['prediction_label']
        
        test_mae = mean_absolute_error(y_true, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        test_r2 = r2_score(y_true, y_pred)
        
        print(f"Test MAE:  {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²:   {test_r2:.4f}")
        
        # Pass/Fail evaluation
        print(f"\n{'='*70}")
        if test_mae < 0.5:
            print("✅ CLINICAL GOAL ACHIEVED (MAE < 0.5)")
        else:
            print("❌ CLINICAL GOAL NOT MET (MAE >= 0.5)")
        print(f"{'='*70}")
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"pycaret_model_{dataset_name.replace(' ', '_')}"
        model_path = os.path.join(model_dir, model_name)
        save_model(final_model, model_path)
        print(f"\n💾 Model saved to: {model_path}.pkl")
        
        return {
            'dataset': dataset_name,
            'cv_mae': cv_mae,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'goal_achieved': test_mae < 0.5
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: Training failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute four-model training strategy."""
    
    # Detect environment and set paths accordingly
    if os.path.exists("/content"):
        # Google Colab environment
        data_dir = "/content/final_imputed_data"
        model_dir = "/content/models"
        print("\n🌐 Detected: Google Colab environment")
    else:
        # Local environment
        data_dir = "./final_imputed_data"
        model_dir = "./models"
        print("\n💻 Detected: Local environment")
    
    # Define data file paths
    file1 = os.path.join(data_dir, "nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv")
    file2 = os.path.join(data_dir, "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv")
    file3 = os.path.join(data_dir, "PrePostFinal (3)_selected_columns_cleaned_processed.csv")
    
    results = []
    
    # Model 1: nmbfinalDiabetes
    print("\n" + "="*70)
    print("MODEL 1: nmbfinalDiabetes Dataset")
    print("="*70)
    data1 = load_and_check_data([file1])
    if data1 is not None:
        result1 = train_pycaret_model(data1, "nmbfinalDiabetes", model_dir)
        if result1:
            results.append(result1)
    
    # Model 2: nmbfinalnewDiabetes
    print("\n" + "="*70)
    print("MODEL 2: nmbfinalnewDiabetes Dataset")
    print("="*70)
    data2 = load_and_check_data([file2])
    if data2 is not None:
        result2 = train_pycaret_model(data2, "nmbfinalnewDiabetes", model_dir)
        if result2:
            results.append(result2)
    
    # Model 3: PrePostFinal
    print("\n" + "="*70)
    print("MODEL 3: PrePostFinal Dataset")
    print("="*70)
    data3 = load_and_check_data([file3])
    if data3 is not None:
        result3 = train_pycaret_model(data3, "PrePostFinal", model_dir)
        if result3:
            results.append(result3)
    
    # Model 4: Combined
    print("\n" + "="*70)
    print("MODEL 4: Combined Dataset")
    print("="*70)
    data_combined = load_and_check_data([file1, file2, file3])
    if data_combined is not None:
        result4 = train_pycaret_model(data_combined, "Combined", model_dir)
        if result4:
            results.append(result4)
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - PyCaret AutoML Results")
    print("="*70)
    
    if results:
        summary_df = pd.DataFrame(results)
        print("\nCross-Validation Results (10-Fold):")
        print(summary_df[['dataset', 'cv_mae', 'cv_rmse', 'cv_r2']].to_string(index=False))
        
        print("\nTest Set Results (20% Hold-out):")
        print(summary_df[['dataset', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
        
        print("\nClinical Goal Achievement:")
        for result in results:
            status = "✅ ACHIEVED" if result['goal_achieved'] else "❌ NOT MET"
            print(f"  {result['dataset']:25s} MAE={result['test_mae']:.4f} {status}")
        
        # Overall statistics
        avg_test_mae = summary_df['test_mae'].mean()
        print(f"\n📈 Average Test MAE across all models: {avg_test_mae:.4f}")
        
        achieved_count = summary_df['goal_achieved'].sum()
        total_count = len(summary_df)
        print(f"🎯 Models achieving clinical goal: {achieved_count}/{total_count}")
    else:
        print("\n⚠️  No models were successfully trained.")
    
    print("\n" + "="*70)
    print(f"✅ Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
