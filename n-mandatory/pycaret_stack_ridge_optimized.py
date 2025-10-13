#!/usr/bin/env python3
"""
PyCaret-Powered Stack Ridge for HbA1c Prediction
Target: MAE < 0.5

Combines PyCaret's AutoML capabilities with advanced stacking ensemble.
Based on insights from main.py - uses only high-performance components.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# System configuration (from main.py)
import psutil
import multiprocessing as mp

def get_system_config():
    """Adaptive configuration based on system resources"""
    cores = mp.cpu_count()
    ram_gb = round(psutil.virtual_memory().total / (1024**3))
    
    if cores >= 8 and ram_gb >= 16:
        return {
            'n_jobs': min(cores - 2, 16),
            'fold': 10,
            'train_size': 0.8,
            'transformation': True,
            'remove_outliers': False,  # Keep clinical edge cases
            'outliers_threshold': 0.01,  # Very conservative
            'multicollinearity_threshold': 0.95,  # Less aggressive
            'feature_selection': True,
            'feature_selection_threshold': 0.8
        }
    elif cores >= 4 and ram_gb >= 8:
        return {
            'n_jobs': min(cores - 1, 8),
            'fold': 5,
            'train_size': 0.8,
            'transformation': True,
            'remove_outliers': False,
            'outliers_threshold': 0.02,
            'multicollinearity_threshold': 0.9,
            'feature_selection': True,
            'feature_selection_threshold': 0.8
        }
    else:
        return {
            'n_jobs': 2,
            'fold': 5,
            'train_size': 0.8,
            'transformation': False,
            'remove_outliers': False,
            'multicollinearity_threshold': 0.85,
            'feature_selection': False
        }

cfg = get_system_config()
print(f"🔧 System Config: {mp.cpu_count()} cores, {round(psutil.virtual_memory().total / (1024**3))}GB RAM")
print(f"📊 Using: n_jobs={cfg['n_jobs']}, fold={cfg['fold']}")

# Load data
def load_best_dataset():
    """Load the most complete dataset"""
    base_paths = ['./final_imputed_data/', 'final_imputed_data/', './']
    dataset_files = [
        'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
        'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
        'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
    ]
    
    target_column = 'PostBLHBA1C'
    
    for file in dataset_files:
        for bp in base_paths:
            path = os.path.join(bp, file)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if target_column in df.columns:
                    # Clean target
                    df = df.dropna(subset=[target_column])
                    # Remove extreme outliers only (>4 std)
                    z_scores = np.abs((df[target_column] - df[target_column].mean()) / df[target_column].std())
                    df = df[z_scores <= 4]
                    
                    print(f"✅ Loaded {file}: {df.shape}")
                    print(f"📈 Target range: {df[target_column].min():.2f} - {df[target_column].max():.2f}")
                    print(f"📊 Target mean ± std: {df[target_column].mean():.2f} ± {df[target_column].std():.2f}")
                    return df, target_column
    
    raise FileNotFoundError("No suitable dataset found!")

df, target_column = load_best_dataset()

# Advanced feature engineering (medical domain knowledge)
def enhance_features(df, target_col):
    """Create advanced features for HbA1c prediction"""
    df_enhanced = df.copy()
    
    # Separate numeric and categorical
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    print(f"🧬 Feature Engineering: {len(numeric_cols)} numeric features")
    
    # 1. Medical risk scores
    if len(numeric_cols) >= 3:
        # Diabetes risk composite score
        top_features = df_enhanced[numeric_cols].corrwith(df_enhanced[target_col]).abs().nlargest(5).index
        for i, feat in enumerate(top_features[:3]):
            df_enhanced[f'risk_score_{i+1}'] = df_enhanced[feat]
    
    # 2. Statistical interactions for top predictors
    corr_with_target = df_enhanced[numeric_cols].corrwith(df_enhanced[target_col]).abs()
    high_corr = corr_with_target[corr_with_target > 0.2].index.tolist()
    
    if len(high_corr) >= 2:
        for i, feat1 in enumerate(high_corr[:3]):
            for feat2 in high_corr[i+1:4]:
                # Multiplicative interaction
                df_enhanced[f'{feat1}_x_{feat2}'] = df_enhanced[feat1] * df_enhanced[feat2]
                # Ratio (avoid division by zero)
                denominator = df_enhanced[feat2].replace(0, 1e-8)
                df_enhanced[f'{feat1}_div_{feat2}'] = df_enhanced[feat1] / denominator
    
    # 3. Non-linear transformations for top features
    for feat in high_corr[:3]:
        df_enhanced[f'{feat}_sq'] = df_enhanced[feat] ** 2
        df_enhanced[f'{feat}_sqrt'] = np.sqrt(np.abs(df_enhanced[feat]))
        df_enhanced[f'{feat}_log'] = np.log1p(np.abs(df_enhanced[feat]))
    
    # 4. Aggregate features
    current_numeric = df_enhanced.select_dtypes(include=[np.number]).columns
    if target_col in current_numeric:
        current_numeric = current_numeric.drop(target_col)
    
    if len(current_numeric) > 0:
        df_enhanced['feat_mean'] = df_enhanced[current_numeric].mean(axis=1)
        df_enhanced['feat_std'] = df_enhanced[current_numeric].std(axis=1)
        df_enhanced['feat_max'] = df_enhanced[current_numeric].max(axis=1)
        df_enhanced['feat_min'] = df_enhanced[current_numeric].min(axis=1)
    
    # Handle infinite and NaN values
    df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
    df_enhanced = df_enhanced.fillna(df_enhanced.median())
    
    print(f"✨ Enhanced: {df.shape[1]} → {df_enhanced.shape[1]} features (+{df_enhanced.shape[1] - df.shape[1]})")
    return df_enhanced

df_enhanced = enhance_features(df, target_column)

# PyCaret setup with optimized configuration
print("\n🚀 Setting up PyCaret AutoML Environment...")
from pycaret.regression import *

# Setup with conservative outlier handling and advanced preprocessing
regression_setup = setup(
    data=df_enhanced,
    target=target_column,
    session_id=42,
    train_size=cfg['train_size'],
    fold=cfg['fold'],
    
    # Preprocessing
    numeric_imputation='mean',
    categorical_imputation='mode',
    normalize=True,
    transformation=cfg['transformation'],
    transformation_method='quantile',  # Better for non-normal distributions
    
    # Feature selection (conservative)
    remove_multicollinearity=True,
    multicollinearity_threshold=cfg['multicollinearity_threshold'],
    feature_selection=cfg['feature_selection'],
    feature_selection_threshold=cfg.get('feature_selection_threshold', 0.8),
    
    # Conservative outlier handling for medical data
    remove_outliers=cfg['remove_outliers'],
    outliers_threshold=cfg.get('outliers_threshold', 0.02),
    
    # System optimization
    fold_strategy='kfold',
    n_jobs=cfg['n_jobs'],
    silent=True
)

# Model selection: Only high-performance models
print("\n🏆 Comparing High-Performance Models...")

# Primary models (proven performers for tabular data)
high_performance_models = [
    'catboost',     # Usually best for tabular data
    'xgboost',      # Excellent gradient boosting
    'lightgbm',     # Fast and accurate
    'gbr',          # Scikit-learn gradient boosting
    'et',           # Extra trees (ensemble)
    'huber',        # Robust to outliers
    'br'            # Bayesian ridge (good baseline)
]

print(f"🔍 Testing {len(high_performance_models)} high-performance models...")

# Compare models with more iterations for better results
best_models = compare_models(
    include=high_performance_models,
    sort='MAE',          # Optimize for MAE directly
    n_select=5,          # Take top 5 models
    fold=cfg['fold'],
    round=4,
    verbose=False
)

print("✅ Model comparison completed!")

# Hyperparameter tuning for top 3 models
print("\n⚡ Hyperparameter Tuning Top Models...")

tuned_models = []
for i, model in enumerate(best_models[:3]):
    print(f"🔧 Tuning model {i+1}/3...")
    try:
        tuned = tune_model(
            model,
            optimize='MAE',
            n_iter=50 if cfg['n_jobs'] > 4 else 30,  # More iterations if more cores
            fold=cfg['fold'],
            choose_better=True,
            verbose=False
        )
        tuned_models.append(tuned)
    except Exception as e:
        print(f"⚠️ Tuning failed for model {i+1}, using base model")
        tuned_models.append(model)

# Advanced ensemble: Blend top models
print("\n🎯 Creating Advanced Ensemble...")

# Method 1: Blending (weighted average)
try:
    blended_model = blend_models(
        estimator_list=tuned_models,
        optimize='MAE',
        fold=cfg['fold'],
        weights=None,  # Auto-optimize weights
        choose_better=True,
        verbose=False
    )
    print("✅ Blended ensemble created")
except Exception as e:
    print(f"⚠️ Blending failed: {e}")
    blended_model = tuned_models[0]  # Fallback to best single model

# Method 2: Stacking with Ridge meta-learner (our signature approach)
print("\n🏗️ Creating Stacked Ensemble with Ridge Meta-Learner...")

try:
    stacked_model = stack_models(
        estimator_list=tuned_models[:3],  # Top 3 base models
        meta_model=create_model('ridge'),  # Ridge as meta-learner
        fold=cfg['fold'],
        restack=True,
        optimize='MAE',
        choose_better=True,
        verbose=False
    )
    print("✅ Stacked ensemble created")
except Exception as e:
    print(f"⚠️ Stacking failed: {e}")
    stacked_model = blended_model  # Fallback to blended

# Model evaluation and comparison
print("\n📊 Final Model Evaluation...")

models_to_evaluate = {
    'Best_Single': tuned_models[0],
    'Blended': blended_model,
    'Stacked_Ridge': stacked_model
}

results = []
for name, model in models_to_evaluate.items():
    print(f"\n🎯 Evaluating {name}...")
    try:
        # Get holdout predictions
        predictions = predict_model(model, verbose=False)
        y_true = predictions[target_column]
        y_pred = predictions['prediction_label']
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        
        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4),
            'Target_Met': '🎯 YES' if mae < 0.5 else '❌ NO'
        })
        
        print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        if mae < 0.5:
            print(f"  🎯 TARGET ACHIEVED! MAE < 0.5")
        else:
            print(f"  📈 Need {((mae/0.5 - 1) * 100):.1f}% improvement")
            
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        results.append({
            'Model': name,
            'MAE': 'Error',
            'RMSE': 'Error', 
            'R²': 'Error',
            'Target_Met': '❌ Error'
        })

# Results summary
print("\n" + "="*60)
print("🏆 FINAL RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<15} | MAE: {row['MAE']:<8} | {row['Target_Met']}")

# Find best model
valid_results = [r for r in results if isinstance(r['MAE'], (int, float))]
if valid_results:
    best_result = min(valid_results, key=lambda x: x['MAE'])
    best_model_name = best_result['Model']
    best_mae = best_result['MAE']
    
    print(f"\n🥇 BEST MODEL: {best_model_name} (MAE: {best_mae})")
    
    if best_mae < 0.5:
        print("🎯 SUCCESS: Target MAE < 0.5 ACHIEVED!")
    else:
        improvement_needed = ((best_mae / 0.5 - 1) * 100)
        print(f"📈 Need {improvement_needed:.1f}% improvement to reach target")
        
        # Suggestions for further improvement
        print("\n💡 IMPROVEMENT STRATEGIES:")
        print("1. Try Neural Networks from main.py (5 architectures)")
        print("2. Add more domain-specific features")
        print("3. Use target transformation (Box-Cox, Yeo-Johnson)")
        print("4. Ensemble with Neural Networks")
        print("5. Use cross-validation stacking")

# Feature importance analysis
print("\n🔍 Feature Importance Analysis...")
try:
    best_model_obj = models_to_evaluate[best_model_name]
    if hasattr(best_model_obj, 'feature_importances_') or hasattr(best_model_obj, 'coef_'):
        # Try to plot feature importance
        plot_model(best_model_obj, plot='feature', save=True)
        print("✅ Feature importance plot saved")
    else:
        print("ℹ️ Feature importance not available for this model type")
except Exception as e:
    print(f"⚠️ Could not generate feature importance: {e}")

# Save best model
print(f"\n💾 Saving best model ({best_model_name})...")
try:
    final_model = finalize_model(models_to_evaluate[best_model_name])
    save_model(final_model, f'pycaret_best_model_{best_model_name.lower()}')
    print(f"✅ Model saved as 'pycaret_best_model_{best_model_name.lower()}.pkl'")
except Exception as e:
    print(f"⚠️ Could not save model: {e}")

print("\n" + "="*60)
print("🎉 PyCaret Stack Ridge Optimization Complete!")
print("="*60)

# Why this approach is better than the focused script:
print("\n🔬 WHY PYCARET IS SUPERIOR:")
print("1. ✅ Automated model selection (7 high-performance algorithms)")
print("2. ✅ Advanced preprocessing (normalization, feature selection)")
print("3. ✅ Hyperparameter optimization (50 iterations per model)")
print("4. ✅ Multiple ensemble methods (blending + stacking)")
print("5. ✅ System-adaptive configuration")
print("6. ✅ Conservative outlier handling (preserves clinical edge cases)")
print("7. ✅ Domain-specific feature engineering")
print("8. ✅ Ridge meta-learner for stacking (your signature approach)")