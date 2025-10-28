#!/usr/bin/env python3
"""
AutoGluon AutoML for Diabetes HbA1c Prediction
==============================================
Target: PostBLHBA1C prediction with MAE < 0.5
Strategy: Four-model approach (3 individual datasets + 1 combined)
Approach: AutoGluon TabularPredictor with best_quality presets
Environment: Google Colab with GPU acceleration
"""

# Installation command for Google Colab (uncomment if needed)
# !pip install autogluon pandas scikit-learn

import pandas as pd
import numpy as np
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*70)
print("🚀 AutoGluon AutoML for Diabetes HbA1c Prediction")
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
    return combined_df

def get_autogluon_hyperparameters():
    """Configure GPU-optimized hyperparameters for AutoGluon."""
    return {
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini'}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr'}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini'}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr'}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
        'NN_TORCH': {},
        'LR': {},
    }

def train_autogluon_model(data, dataset_name, model_dir):
    """Train AutoGluon TabularPredictor model."""
    from autogluon.tabular import TabularPredictor
    
    print(f"\n{'='*70}")
    print(f"--- Training AutoGluon on {dataset_name} Dataset ---")
    print(f"{'='*70}")
    
    try:
        # Define target
        target = 'PostBLHBA1C'
        print(f"✓ Target variable: {target}")
        print(f"✓ Feature count: {len(data.columns) - 1}")
        
        # Split data (80/20 train/test with seed 42)
        print("\n[SPLIT] Creating train/test split (80/20)...")
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=42
        )
        print(f"✓ Train set: {len(train_data)} rows")
        print(f"✓ Test set:  {len(test_data)} rows")
        
        # Configure model save path
        model_save_path = os.path.join(model_dir, f"autogluon_model_{dataset_name.replace(' ', '_')}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize TabularPredictor
        print("\n[SETUP] Initializing AutoGluon TabularPredictor...")
        predictor = TabularPredictor(
            label=target,
            eval_metric='mean_absolute_error',
            path=model_save_path,
            verbosity=2
        )
        print("✓ TabularPredictor initialized")
        
        # Get GPU-optimized hyperparameters
        hyperparameters = get_autogluon_hyperparameters()
        
        # Train models
        print("\n[MODELING] Training AutoGluon models (max 600 seconds)...")
        predictor.fit(
            train_data=train_data,
            presets='best_quality',
            time_limit=600,
            num_bag_folds=10,
            num_bag_sets=1,
            num_stack_levels=1,
            hyperparameters=hyperparameters,
            verbosity=2
        )
        print("✓ AutoGluon training completed")
        
        # Get leaderboard
        print(f"\n{'='*70}")
        print("--- AutoGluon Leaderboard (Top 15 Models) ---")
        print(f"{'='*70}")
        leaderboard = predictor.leaderboard(test_data, silent=True)
        print(leaderboard.head(15).to_string())
        
        # Get best model
        best_model = leaderboard.iloc[0]['model']
        print(f"\n🏆 Best Model: {best_model}")
        
        # Cross-Validation Metrics (from leaderboard - OOF predictions)
        print(f"\n{'='*70}")
        print("--- Cross-Validation Metrics (10-Fold Bagging, OOF) ---")
        print(f"{'='*70}")
        
        # Get OOF (out-of-fold) scores from leaderboard
        best_model_row = leaderboard[leaderboard['model'] == best_model].iloc[0]
        cv_score = best_model_row['score_val']  # Validation score (MAE)
        
        print(f"CV MAE (OOF): {cv_score:.4f}")
        print("Note: AutoGluon uses bagging with OOF predictions for CV evaluation")
        
        # Test Set Metrics (20% Hold-out)
        print(f"\n{'='*70}")
        print("--- Test Set Metrics (20% Hold-out, Best Model) ---")
        print(f"{'='*70}")
        
        # Evaluate on test set
        test_performance = predictor.evaluate(test_data, silent=True)
        test_mae = test_performance['mean_absolute_error']
        
        # Get predictions for additional metrics
        predictions = predictor.predict(test_data)
        y_test = test_data[target]
        
        from sklearn.metrics import mean_squared_error, r2_score
        test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        test_r2 = r2_score(y_test, predictions)
        
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
        
        print(f"\n💾 Model automatically saved to: {model_save_path}")
        
        # Feature importance
        print(f"\n{'='*70}")
        print("--- Top 15 Feature Importances ---")
        print(f"{'='*70}")
        try:
            feature_importance = predictor.feature_importance(test_data)
            top_features = feature_importance.head(15)
            print(top_features.to_string())
        except Exception as e:
            print(f"⚠️  Feature importance not available: {e}")
        
        return {
            'dataset': dataset_name,
            'best_model': best_model,
            'cv_mae': cv_score,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'goal_achieved': test_mae < 0.5,
            'model_path': model_save_path
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
        result1 = train_autogluon_model(data1, "nmbfinalDiabetes", model_dir)
        if result1:
            results.append(result1)
    
    # Model 2: nmbfinalnewDiabetes
    print("\n" + "="*70)
    print("MODEL 2: nmbfinalnewDiabetes Dataset")
    print("="*70)
    data2 = load_and_check_data([file2])
    if data2 is not None:
        result2 = train_autogluon_model(data2, "nmbfinalnewDiabetes", model_dir)
        if result2:
            results.append(result2)
    
    # Model 3: PrePostFinal
    print("\n" + "="*70)
    print("MODEL 3: PrePostFinal Dataset")
    print("="*70)
    data3 = load_and_check_data([file3])
    if data3 is not None:
        result3 = train_autogluon_model(data3, "PrePostFinal", model_dir)
        if result3:
            results.append(result3)
    
    # Model 4: Combined
    print("\n" + "="*70)
    print("MODEL 4: Combined Dataset")
    print("="*70)
    data_combined = load_and_check_data([file1, file2, file3])
    if data_combined is not None:
        result4 = train_autogluon_model(data_combined, "Combined", model_dir)
        if result4:
            results.append(result4)
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - AutoGluon Results")
    print("="*70)
    
    if results:
        summary_df = pd.DataFrame(results)
        print("\nCross-Validation Results (10-Fold Bagging, OOF):")
        print(summary_df[['dataset', 'cv_mae']].to_string(index=False))
        
        print("\nTest Set Results (20% Hold-out):")
        print(summary_df[['dataset', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
        
        print("\nBest Models:")
        print(summary_df[['dataset', 'best_model']].to_string(index=False))
        
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
        
        print("\nModel Locations:")
        for result in results:
            print(f"  {result['dataset']:25s} → {result['model_path']}")
    else:
        print("\n⚠️  No models were successfully trained.")
    
    print("\n" + "="*70)
    print(f"✅ Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
