#!/usr/bin/env python3
"""
H2O AutoML for Diabetes HbA1c Prediction - ADVANCED VERSION
============================================================
Target: PostBLHBA1C prediction with MAE < 0.5
Strategy: Multi-dataset approach with advanced analysis and validation

Features:
- Focused AutoML (XGBoost, GBM, StackedEnsemble)
- SHAP analysis and feature importance
- External validation across datasets
- Two-dataset combinations training
- Optional preprocessing and manual stacking
- Modular design with control flags

Author: AI Assistant
Date: 2025-10-27
Version: 2.0 - Advanced Analysis & Validation
"""

# Installation command for Google Colab (uncomment if needed)
# !pip install h2o pandas scikit-learn

import pandas as pd
import numpy as np
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# ============================================================================
# EXECUTION CONTROL FLAGS - Customize your run here
# ============================================================================

# Core runs (always enabled)
RUN_INDIVIDUAL_DATASETS = True  # Models 1-3: Individual datasets
RUN_THREE_DATASET_COMBO = True  # Model 4: All three datasets combined

# Extended training options
RUN_TWO_DATASET_COMBOS = True   # Models 5-7: Pairwise dataset combinations
RUN_EXTERNAL_VALIDATION = True  # Cross-validate models on different datasets

# Optional advanced strategies
RUN_PREPROCESSING_STRATEGY = False  # Strategy 2: Yeo-Johnson transformation
RUN_MANUAL_STACKING = False         # Strategy 3: Custom metalearner tuning

# Runtime adjustments
USE_EXTENDED_RUNTIME_FOR_COMBOS = True  # 3600s for combined datasets, 1800s for individual
STANDARD_RUNTIME = 1800  # 30 minutes
EXTENDED_RUNTIME = 3600  # 60 minutes

print("="*80)
print("🚀 H2O AutoML for Diabetes HbA1c Prediction - ADVANCED VERSION")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Target: PostBLHBA1C prediction with MAE < 0.5")
print("\n� Execution Plan:")
print(f"  ✓ Individual Datasets: {RUN_INDIVIDUAL_DATASETS}")
print(f"  ✓ 3-Dataset Combination: {RUN_THREE_DATASET_COMBO}")
print(f"  ✓ 2-Dataset Combinations: {RUN_TWO_DATASET_COMBOS}")
print(f"  ✓ External Validation: {RUN_EXTERNAL_VALIDATION}")
print(f"  • Preprocessing Strategy: {RUN_PREPROCESSING_STRATEGY}")
print(f"  • Manual Stacking: {RUN_MANUAL_STACKING}")
print("="*80)
# ============================================================================
# HELPER FUNCTIONS - Data Loading & Preprocessing
# ============================================================================

def clean_column_names(df):
    """Cleans column names for H2O compatibility."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df


def load_and_check_data(file_paths, dataset_name="Dataset"):
    """Loads, concatenates, and validates data files."""
    data_frames = []
    print(f"\n{'='*80}")
    print(f"📂 Loading Data for: {dataset_name}")
    print(f"{'='*80}")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️  WARNING: File not found at {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
            df = clean_column_names(df)
            
            # Validate target variable
            if 'PostBLHBA1C' not in df.columns:
                print(f"⚠️  WARNING: Target 'PostBLHBA1C' not in {os.path.basename(file_path)}. Skipping.")
                continue
                
            df['PostBLHBA1C'] = pd.to_numeric(df['PostBLHBA1C'], errors='coerce')
            data_frames.append(df)
            print(f"  ✓ {os.path.basename(file_path)}: {len(df):,} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"  ❌ ERROR loading {os.path.basename(file_path)}: {e}")

    if not data_frames:
        print("\n❌ ERROR: No data files were successfully loaded.")
        return None

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=['PostBLHBA1C'])
    
    print(f"\n📊 Combined Dataset Summary:")
    print(f"  • Total rows: {len(combined_df):,}")
    print(f"  • Total columns: {len(combined_df.columns)}")
    print(f"  • Target range: [{combined_df['PostBLHBA1C'].min():.2f}, {combined_df['PostBLHBA1C'].max():.2f}]")
    print(f"  • Target mean: {combined_df['PostBLHBA1C'].mean():.4f}")
    print(f"  • Missing values: {combined_df.isnull().sum().sum():,}")
    
    return combined_df


def apply_yeo_johnson_preprocessing(df, target_col='PostBLHBA1C'):
    """Apply Yeo-Johnson power transformation to numeric features (Strategy 2)."""
    from sklearn.preprocessing import PowerTransformer
    
    print(f"\n🔧 Applying Yeo-Johnson Preprocessing...")
    df_processed = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) == 0:
        print("  ⚠️  No numeric columns to transform")
        return df_processed
    
    # Apply transformation
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df_processed[numeric_cols] = pt.fit_transform(df_processed[numeric_cols])
    
    print(f"  ✓ Transformed {len(numeric_cols)} numeric columns")
    return df_processed


# ============================================================================
# ADVANCED ANALYSIS HELPER FUNCTIONS
# ============================================================================

def get_feature_importance(model, top_n=20):
    """Extract and display feature importance from H2O model."""
    try:
        varimp = model.varimp(use_pandas=True)
        if varimp is not None and len(varimp) > 0:
            print(f"\n📊 Feature Importance (Top {min(top_n, len(varimp))}):")
            print("="*80)
            top_features = varimp.head(top_n)
            for idx, row in top_features.iterrows():
                print(f"  {idx+1:2d}. {row['variable']:<40s} Importance: {row['relative_importance']:.4f}")
            return top_features
        else:
            print("\n⚠️  Feature importance not available for this model type")
            return None
    except Exception as e:
        print(f"\n⚠️  Could not extract feature importance: {e}")
        return None


def perform_shap_analysis(model, test_data, dataset_name, max_rows=500):
    """Perform SHAP analysis with error handling and timeout warnings."""
    print(f"\n🔍 SHAP Analysis for {dataset_name}")
    print("="*80)
    
    try:
        # Limit rows for SHAP to avoid excessive computation
        test_sample = test_data if test_data.nrows <= max_rows else test_data[:max_rows, :]
        
        print(f"  • Computing SHAP contributions (using {test_sample.nrows} rows)...")
        start_time = time.time()
        
        # Calculate contributions
        contributions = model.predict_contributions(test_sample)
        elapsed_time = time.time() - start_time
        
        print(f"  ✓ SHAP contributions computed in {elapsed_time:.2f} seconds")
        
        # Display head of contributions
        print(f"\n  SHAP Contributions (first 5 rows):")
        print(contributions.head(5))
        
        # Try to generate SHAP summary plot
        try:
            print(f"\n  • Generating SHAP summary plot...")
            model.shap_summary_plot(test_sample)
            print(f"  ✓ SHAP summary plot generated")
        except Exception as plot_error:
            print(f"  ⚠️  Could not generate SHAP plot: {plot_error}")
        
        # Warning for StackedEnsemble
        if 'StackedEnsemble' in model.model_id:
            print(f"\n  ⚠️  NOTE: SHAP for StackedEnsemble primarily reflects the metalearner,")
            print(f"          not the base models. Consider analyzing base models individually.")
        
        return contributions
        
    except Exception as e:
        print(f"\n  ❌ SHAP analysis failed: {e}")
        print(f"      This may occur with certain model types or large datasets.")
        return None


def print_detailed_metrics(performance, metric_type="Test"):
    """Print comprehensive model performance metrics."""
    print(f"\n📈 {metric_type} Set Performance Metrics:")
    print("="*80)
    
    try:
        mae = performance.mae()
        rmse = performance.rmse()
        r2 = performance.r2()
        mse = performance.mse()
        
        print(f"  MAE:   {mae:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  R²:    {r2:.6f}")
        
        # Additional metrics if available
        try:
            mean_residual_deviance = performance.mean_residual_deviance()
            print(f"  Mean Residual Deviance: {mean_residual_deviance:.6f}")
        except:
            pass
            
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mse': mse}
        
    except Exception as e:
        print(f"  ⚠️  Could not extract all metrics: {e}")
        return None

        return None


# ============================================================================
# CORE TRAINING FUNCTION - Strategy 1 (Focused AutoML)
# ============================================================================

def train_h2o_model(data, dataset_name, model_dir, max_runtime_secs=1800, 
                    apply_preprocessing=False, use_manual_stacking=False):
    """
    Train H2O AutoML model with comprehensive analysis.
    
    Args:
        data: pandas DataFrame with cleaned data
        dataset_name: str, identifier for this dataset
        model_dir: str, directory to save models
        max_runtime_secs: int, max training time in seconds
        apply_preprocessing: bool, apply Yeo-Johnson transformation
        use_manual_stacking: bool, create custom stacked ensemble
    
    Returns:
        dict with training results and analysis
    """
    import h2o
    from h2o.automl import H2OAutoML
    
    print(f"\n{'='*80}")
    print(f"🎓 Training H2O AutoML: {dataset_name}")
    print(f"{'='*80}")
    print(f"  Strategy: Focused AutoML (XGBoost, GBM, StackedEnsemble)")
    print(f"  Max Runtime: {max_runtime_secs}s ({max_runtime_secs/60:.1f} minutes)")
    print(f"  Preprocessing: {'Enabled' if apply_preprocessing else 'Disabled'}")
    print(f"  Manual Stacking: {'Enabled' if use_manual_stacking else 'Disabled'}")

    try:
        # Optional preprocessing
        if apply_preprocessing:
            data = apply_yeo_johnson_preprocessing(data)
        
        # Convert to H2OFrame
        print(f"\n[SETUP] Converting to H2O format...")
        h2o_df = h2o.H2OFrame(data)
        
        # Define target and features
        y = 'PostBLHBA1C'
        x = h2o_df.columns
        x.remove(y)
        print(f"  ✓ Target: {y}")
        print(f"  ✓ Features: {len(x)} columns")

        # Train/Test Split (80/20 with seed 42)
        print(f"\n[SPLIT] Creating 80/20 train/test split (seed=42)...")
        splits = h2o_df.split_frame(ratios=[0.8], seed=42)
        train_h2o = splits[0]
        test_h2o = splits[1]
        print(f"  ✓ Train: {train_h2o.nrows:,} rows")
        print(f"  ✓ Test:  {test_h2o.nrows:,} rows")

        # Configure H2O AutoML - STRATEGY 1 (Focused)
        print(f"\n[AUTOML] Configuring focused AutoML...")
        print(f"  • Algorithms: XGBoost, GBM, StackedEnsemble")
        print(f"  • Max Models: 50")
        print(f"  • Cross-Validation: 10 folds")
        
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=50,
            nfolds=10,
            sort_metric='mae',
            include_algos=["XGBoost", "GBM", "StackedEnsemble"],
            stopping_metric='mae',
            stopping_tolerance=0.001,
            stopping_rounds=5,
            project_name=f"hba1c_{dataset_name.replace(' ', '_').replace('+', '_')}",
            keep_cross_validation_predictions=True,
            keep_cross_validation_models=True,
            seed=42,
            verbosity="info"
        )

        # Train
        print(f"\n[TRAINING] Starting AutoML training...")
        training_start = time.time()
        aml.train(x=x, y=y, training_frame=train_h2o, leaderboard_frame=test_h2o)
        training_time = time.time() - training_start
        print(f"  ✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        # Get leader model
        leader = aml.leader
        print(f"\n🏆 Leader Model: {leader.model_id}")
        print(f"   Model Type: {leader.algo}")

        # Leaderboard
        lb = aml.leaderboard
        print(f"\n{'='*80}")
        print("📊 H2O Leaderboard (Top 15 Models)")
        print(f"{'='*80}")
        print(lb.head(rows=min(15, lb.nrows)))

        # Cross-Validation Metrics (10-Fold)
        print(f"\n{'='*80}")
        print("🔄 Cross-Validation Metrics (10-Fold, Leader Model)")
        print(f"{'='*80}")
        cv_mae = leader.mae(xval=True)
        cv_rmse = leader.rmse(xval=True)
        cv_r2 = leader.r2(xval=True)
        print(f"  CV MAE:  {cv_mae:.6f}")
        print(f"  CV RMSE: {cv_rmse:.6f}")
        print(f"  CV R²:   {cv_r2:.6f}")

        # Test Set Performance - Detailed
        test_performance = leader.model_performance(test_h2o)
        test_metrics = print_detailed_metrics(test_performance, "Test")
        
        # Pass/Fail Clinical Goal
        print(f"\n{'='*80}")
        test_mae = test_metrics['mae']
        if test_mae < 0.5:
            print("✅ CLINICAL GOAL ACHIEVED (MAE < 0.5)")
        else:
            print(f"❌ CLINICAL GOAL NOT MET (MAE = {test_mae:.4f} >= 0.5)")
        print(f"{'='*80}")

        # Feature Importance Analysis
        top_features = get_feature_importance(leader, top_n=20)

        # SHAP Analysis
        shap_contributions = perform_shap_analysis(leader, test_h2o, dataset_name)

        # Optional: Manual Stacked Ensemble (Strategy 3)
        manual_stack_result = None
        if use_manual_stacking and len(aml.leaderboard) > 5:
            print(f"\n{'='*80}")
            print("🔧 Creating Manual Stacked Ensemble (Strategy 3)")
            print(f"{'='*80}")
            try:
                # Get base models (exclude existing stacks)
                base_models = [model for model in aml.leaderboard['model_id'].as_data_frame()['model_id'] 
                              if 'StackedEnsemble' not in model][:10]  # Top 10 base models
                
                print(f"  • Using {len(base_models)} base models")
                print(f"  • Metalearner: GBM")
                
                manual_stack = h2o.estimators.H2OStackedEnsembleEstimator(
                    base_models=base_models,
                    metalearner_algorithm="gbm",
                    metalearner_nfolds=5,
                    seed=42
                )
                manual_stack.train(x=x, y=y, training_frame=train_h2o)
                
                manual_perf = manual_stack.model_performance(test_h2o)
                manual_mae = manual_perf.mae()
                print(f"\n  📊 Manual Stack Test MAE: {manual_mae:.6f}")
                
                if manual_mae < test_mae:
                    print(f"  ✅ Manual stack improved over AutoML leader by {test_mae - manual_mae:.6f}")
                    manual_stack_result = {
                        'mae': manual_mae,
                        'improved': True,
                        'improvement': test_mae - manual_mae
                    }
                else:
                    print(f"  ⚠️  Manual stack did not improve (difference: {manual_mae - test_mae:.6f})")
                    manual_stack_result = {'mae': manual_mae, 'improved': False}
                    
            except Exception as e:
                print(f"  ❌ Manual stacking failed: {e}")

        # Save Model
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"h2o_model_{dataset_name.replace(' ', '_').replace('+', '_')}"
        model_path = os.path.join(model_dir, model_name)
        saved_path = h2o.save_model(leader, path=model_path, force=True)
        print(f"\n💾 Model saved to: {saved_path}")

        # Compile results
        result = {
            'dataset': dataset_name,
            'model_id': leader.model_id,
            'model_type': leader.algo,
            'cv_mae': cv_mae,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_mse': test_metrics['mse'],
            'goal_achieved': test_metrics['mae'] < 0.5,
            'training_time_sec': training_time,
            'model_path': saved_path,
            'top_features': top_features['variable'].tolist()[:5] if top_features is not None else [],
            'preprocessing_applied': apply_preprocessing,
            'manual_stack_result': manual_stack_result,
            'train_h2o': train_h2o,  # Keep for external validation
            'test_h2o': test_h2o
        }

        return result

    except Exception as e:
        print(f"\n❌ ERROR: Training failed for {dataset_name}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# EXTERNAL VALIDATION FUNCTION
# ============================================================================

def perform_external_validation(model_path, external_data, validation_name, model_name):
    """
    Load a saved model and validate on external dataset.
    
    Args:
        model_path: str, path to saved H2O model
        external_data: pandas DataFrame with external validation data
        validation_name: str, name of external dataset
        model_name: str, name of source model
    
    Returns:
        dict with validation metrics
    """
    import h2o
    
    print(f"\n{'='*80}")
    print(f"🔬 External Validation: {model_name} → {validation_name}")
    print(f"{'='*80}")
    
    try:
        # Load saved model
        print(f"  • Loading model from: {model_path}")
        model = h2o.load_model(model_path)
        print(f"  ✓ Model loaded: {model.model_id}")
        
        # Convert external data to H2OFrame
        print(f"  • Converting external data to H2O format...")
        h2o_external = h2o.H2OFrame(external_data)
        
        # Recreate same 80/20 split with seed=42 to get test portion
        print(f"  • Creating validation split (80/20, seed=42)...")
        splits = h2o_external.split_frame(ratios=[0.8], seed=42)
        external_test = splits[1]  # Use test portion only
        print(f"  ✓ Validation set: {external_test.nrows:,} rows")
        
        # Evaluate model on external test set
        print(f"  • Evaluating model performance...")
        external_perf = model.model_performance(external_test)
        
        ext_mae = external_perf.mae()
        ext_rmse = external_perf.rmse()
        ext_r2 = external_perf.r2()
        
        print(f"\n  📊 External Validation Results:")
        print(f"     MAE:   {ext_mae:.6f}")
        print(f"     RMSE:  {ext_rmse:.6f}")
        print(f"     R²:    {ext_r2:.6f}")
        
        if ext_mae < 0.5:
            print(f"     ✅ Clinical goal achieved on external data")
        else:
            print(f"     ⚠️  Clinical goal not met on external data")
        
        return {
            'model_name': model_name,
            'validation_dataset': validation_name,
            'ext_mae': ext_mae,
            'ext_rmse': ext_rmse,
            'ext_r2': ext_r2,
            'goal_achieved': ext_mae < 0.5
        }
        
    except Exception as e:
        print(f"\n  ❌ External validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION - Orchestrate all training runs and validation
# ============================================================================

def main():
    """Execute comprehensive multi-dataset training and validation strategy."""
    import h2o

    # Initialize H2O
    print(f"\n{'='*80}")
    print("🚀 Initializing H2O Cluster")
    print(f"{'='*80}")
    try:
        h2o.init(max_mem_size='16G', nthreads=-1)
        print("✓ H2O cluster initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: H2O initialization issue: {e}")
        print("Attempting to connect to existing cluster...")
        try:
            h2o.connect()
            print("✓ Connected to existing H2O cluster")
        except:
            print("❌ ERROR: Could not initialize or connect to H2O cluster")
            return

    # Detect environment
    if os.path.exists("/content"):
        data_dir = "/content/final_imputed_data"
        model_dir = "/content/models"
        print(f"\n🌐 Environment: Google Colab")
    else:
        data_dir = "./final_imputed_data"
        model_dir = "./models"
        print(f"\n💻 Environment: Local")

    print(f"   Data Directory: {data_dir}")
    print(f"   Model Directory: {model_dir}")

    # Define file paths - try both _final_imputed and _cleaned_processed versions
    # Check which files actually exist
    file1_imputed = os.path.join(data_dir, "nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv")
    file1_cleaned = os.path.join(data_dir, "nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv")
    file1 = file1_imputed if os.path.exists(file1_imputed) else file1_cleaned
    
    file2_imputed = os.path.join(data_dir, "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv")
    file2_cleaned = os.path.join(data_dir, "nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv")
    file2 = file2_imputed if os.path.exists(file2_imputed) else file2_cleaned
    
    file3_imputed = os.path.join(data_dir, "PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv")
    file3_cleaned = os.path.join(data_dir, "PrePostFinal (3)_selected_columns_cleaned_processed.csv")
    file3 = file3_imputed if os.path.exists(file3_imputed) else file3_cleaned
    
    print(f"\n📁 File Detection:")
    print(f"  • Dataset 1: {'_final_imputed' if os.path.exists(file1_imputed) else '_cleaned_processed'}")
    print(f"  • Dataset 2: {'_final_imputed' if os.path.exists(file2_imputed) else '_cleaned_processed'}")
    print(f"  • Dataset 3: {'_final_imputed' if os.path.exists(file3_imputed) else '_cleaned_processed'}")

    # Storage for results
    all_results = []
    external_validation_results = []
    
    # Data cache for reuse
    data_cache = {}

    # ========================================================================
    # PHASE 1: CORE INDIVIDUAL DATASETS
    # ========================================================================
    
    if RUN_INDIVIDUAL_DATASETS:
        print(f"\n\n{'#'*80}")
        print("# PHASE 1: TRAINING ON INDIVIDUAL DATASETS (Models 1-3)")
        print(f"{'#'*80}")
        
        datasets = [
            ("nmbfinalDiabetes", [file1]),
            ("nmbfinalnewDiabetes", [file2]),
            ("PrePostFinal", [file3])
        ]
        
        for dataset_name, file_paths in datasets:
            data = load_and_check_data(file_paths, dataset_name)
            if data is not None:
                data_cache[dataset_name] = data
                result = train_h2o_model(
                    data, 
                    dataset_name, 
                    model_dir, 
                    max_runtime_secs=STANDARD_RUNTIME,
                    apply_preprocessing=False,
                    use_manual_stacking=False
                )
                if result:
                    all_results.append(result)

    # ========================================================================
    # PHASE 2: THREE-DATASET COMBINATION
    # ========================================================================
    
    if RUN_THREE_DATASET_COMBO:
        print(f"\n\n{'#'*80}")
        print("# PHASE 2: TRAINING ON 3-DATASET COMBINATION (Model 4)")
        print(f"{'#'*80}")
        
        data_combined = load_and_check_data([file1, file2, file3], "Combined_3_Datasets")
        if data_combined is not None:
            data_cache["Combined_3_Datasets"] = data_combined
            runtime = EXTENDED_RUNTIME if USE_EXTENDED_RUNTIME_FOR_COMBOS else STANDARD_RUNTIME
            result = train_h2o_model(
                data_combined,
                "Combined_3_Datasets",
                model_dir,
                max_runtime_secs=runtime,
                apply_preprocessing=False,
                use_manual_stacking=False
            )
            if result:
                all_results.append(result)

    # ========================================================================
    # PHASE 3: TWO-DATASET COMBINATIONS
    # ========================================================================
    
    if RUN_TWO_DATASET_COMBOS:
        print(f"\n\n{'#'*80}")
        print("# PHASE 3: TRAINING ON 2-DATASET COMBINATIONS (Models 5-7)")
        print(f"{'#'*80}")
        
        two_dataset_combos = [
            ("nmbfinalDiabetes + nmbfinalnewDiabetes", [file1, file2]),
            ("nmbfinalDiabetes + PrePostFinal", [file1, file3]),
            ("nmbfinalnewDiabetes + PrePostFinal", [file2, file3])
        ]
        
        for combo_name, file_paths in two_dataset_combos:
            data_combo = load_and_check_data(file_paths, combo_name)
            if data_combo is not None:
                data_cache[combo_name] = data_combo
                runtime = EXTENDED_RUNTIME if USE_EXTENDED_RUNTIME_FOR_COMBOS else STANDARD_RUNTIME
                result = train_h2o_model(
                    data_combo,
                    combo_name,
                    model_dir,
                    max_runtime_secs=runtime,
                    apply_preprocessing=False,
                    use_manual_stacking=False
                )
                if result:
                    all_results.append(result)

    # ========================================================================
    # PHASE 4: EXTERNAL VALIDATION
    # ========================================================================
    
    if RUN_EXTERNAL_VALIDATION and len(all_results) >= 2:
        print(f"\n\n{'#'*80}")
        print("# PHASE 4: EXTERNAL VALIDATION (Cross-Dataset Testing)")
        print(f"{'#'*80}")
        
        # Validation pairs: Model trained on Dataset A, tested on Dataset B
        validation_pairs = [
            ("nmbfinalDiabetes", "nmbfinalnewDiabetes"),
            ("nmbfinalnewDiabetes", "nmbfinalDiabetes"),
            ("PrePostFinal", "nmbfinalDiabetes"),
        ]
        
        for model_dataset, validation_dataset in validation_pairs:
            # Find model result
            model_result = next((r for r in all_results if r['dataset'] == model_dataset), None)
            if model_result is None or 'model_path' not in model_result:
                print(f"\n⚠️  Skipping validation: {model_dataset} → {validation_dataset} (model not found)")
                continue
            
            # Get validation data
            if validation_dataset not in data_cache:
                print(f"\n⚠️  Skipping validation: {model_dataset} → {validation_dataset} (data not loaded)")
                continue
            
            validation_data = data_cache[validation_dataset]
            ext_result = perform_external_validation(
                model_result['model_path'],
                validation_data,
                validation_dataset,
                model_dataset
            )
            
            if ext_result:
                external_validation_results.append(ext_result)

    # ========================================================================
    # PHASE 5: OPTIONAL STRATEGIES (if goals not met)
    # ========================================================================
    
    # Check if any model achieved the goal
    goals_achieved = sum(r.get('goal_achieved', False) for r in all_results)
    
    if goals_achieved == 0 and (RUN_PREPROCESSING_STRATEGY or RUN_MANUAL_STACKING):
        print(f"\n\n{'#'*80}")
        print("# PHASE 5: OPTIONAL STRATEGIES (No models met MAE < 0.5 goal)")
        print(f"{'#'*80}")
        
        # Apply to best performing dataset so far
        best_result = min(all_results, key=lambda x: x.get('test_mae', float('inf')))
        best_dataset = best_result['dataset']
        
        print(f"\n  Applying optional strategies to best dataset: {best_dataset}")
        print(f"  Current best MAE: {best_result['test_mae']:.6f}")
        
        if best_dataset in data_cache:
            best_data = data_cache[best_dataset]
            
            # Strategy 2: Preprocessing
            if RUN_PREPROCESSING_STRATEGY:
                print(f"\n  🔧 Running Strategy 2: Yeo-Johnson Preprocessing")
                result_prep = train_h2o_model(
                    best_data,
                    f"{best_dataset}_Preprocessed",
                    model_dir,
                    max_runtime_secs=EXTENDED_RUNTIME,
                    apply_preprocessing=True,
                    use_manual_stacking=False
                )
                if result_prep:
                    all_results.append(result_prep)
            
            # Strategy 3: Manual Stacking
            if RUN_MANUAL_STACKING:
                print(f"\n  🔧 Running Strategy 3: Manual Stacked Ensemble")
                result_stack = train_h2o_model(
                    best_data,
                    f"{best_dataset}_ManualStack",
                    model_dir,
                    max_runtime_secs=EXTENDED_RUNTIME,
                    apply_preprocessing=False,
                    use_manual_stacking=True
                )
                if result_stack:
                    all_results.append(result_stack)

    # ========================================================================
    # FINAL SUMMARY AND REPORTING
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print("📊 COMPREHENSIVE FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        # Create summary DataFrame
        summary_df = pd.DataFrame([
            {
                'Dataset': r['dataset'],
                'Model_Type': r.get('model_type', 'N/A'),
                'CV_MAE': r.get('cv_mae', 0),
                'Test_MAE': r.get('test_mae', 0),
                'Test_RMSE': r.get('test_rmse', 0),
                'Test_R2': r.get('test_r2', 0),
                'Goal_Achieved': '✅' if r.get('goal_achieved', False) else '❌',
                'Training_Time_Min': r.get('training_time_sec', 0) / 60
            }
            for r in all_results
        ])
        
        # Sort by Test MAE
        summary_df = summary_df.sort_values('Test_MAE')
        
        print(f"\n{'='*80}")
        print("📈 ALL MODELS COMPARISON (Sorted by Test MAE)")
        print(f"{'='*80}\n")
        print(summary_df.to_string(index=False))
        
        # Clinical Goal Achievement
        achieved_count = summary_df['Goal_Achieved'].str.contains('✅').sum()
        total_count = len(summary_df)
        
        print(f"\n{'='*80}")
        print(f"🎯 Clinical Goal Achievement: {achieved_count}/{total_count} models")
        print(f"{'='*80}")
        
        # Best model
        best_model = all_results[0] if len(all_results) > 0 else None
        best_mae = min(r.get('test_mae', float('inf')) for r in all_results)
        best_dataset = next(r['dataset'] for r in all_results if r.get('test_mae') == best_mae)
        
        print(f"\n🏆 Best Model: {best_dataset}")
        print(f"   Test MAE: {best_mae:.6f}")
        
        # Feature importance comparison
        print(f"\n{'='*80}")
        print("🔍 TOP 5 FEATURES COMPARISON (Across Models)")
        print(f"{'='*80}")
        
        for result in all_results[:5]:  # Show top 5 models
            if 'top_features' in result and result['top_features']:
                print(f"\n{result['dataset']}:")
                for i, feat in enumerate(result['top_features'], 1):
                    print(f"  {i}. {feat}")
        
        # External validation summary
        if external_validation_results:
            print(f"\n{'='*80}")
            print("🔬 EXTERNAL VALIDATION SUMMARY")
            print(f"{'='*80}\n")
            
            ext_df = pd.DataFrame(external_validation_results)
            print(ext_df.to_string(index=False))
            
            ext_goals = sum(r.get('goal_achieved', False) for r in external_validation_results)
            print(f"\n  External validations achieving goal: {ext_goals}/{len(external_validation_results)}")
        
        # Overall statistics
        avg_test_mae = summary_df['Test_MAE'].mean()
        median_test_mae = summary_df['Test_MAE'].median()
        
        print(f"\n{'='*80}")
        print(f"� AGGREGATE STATISTICS")
        print(f"{'='*80}")
        print(f"  Average Test MAE:    {avg_test_mae:.6f}")
        print(f"  Median Test MAE:     {median_test_mae:.6f}")
        print(f"  Best Test MAE:       {best_mae:.6f}")
        print(f"  Total Training Time: {summary_df['Training_Time_Min'].sum():.2f} minutes")
        
    else:
        print("\n⚠️  No models were successfully trained.")

    print(f"\n{'='*80}")
    print(f"✅ Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # Shutdown H2O
    print(f"\n🔒 Shutting down H2O cluster...")
    h2o.cluster().shutdown()
    print("✓ H2O cluster shut down successfully")


if __name__ == "__main__":
    main()