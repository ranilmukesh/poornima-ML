#!/usr/bin/env python3
"""
🎯 ULTRA-OPTIMIZED Stack Ridge for HbA1c Prediction
Target: MAE < 0.5 with advanced techniques
"""

import sys, os, warnings
import numpy as np, pandas as pd
import pickle
from datetime import datetime
import subprocess

warnings.filterwarnings('ignore')

# Install packages
def install_if_needed(package):
    try:
        __import__(package.split('==')[0])
        print(f"✅ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

packages = ['scikit-learn>=1.3.0', 'optuna', 'xgboost', 'lightgbm', 'catboost']
for pkg in packages:
    install_if_needed(pkg)

# Imports
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.neural_network import MLPRegressor
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🎯 ULTRA-OPTIMIZED STACK RIDGE - TARGET MAE < 0.5")
print("=" * 60)

# =============================================================================
# ENHANCED DATA LOADING & PREPROCESSING
# =============================================================================
print("\n📊 ENHANCED DATA LOADING")
print("-" * 30)

base_paths = ['./final_imputed_data/', 'final_imputed_data/', './']
dataset_files = [
    'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
    'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
    'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
]
target_column = 'PostBLHBA1C'

# Load dataset
df = None
for file in dataset_files:
    for bp in base_paths:
        path = os.path.join(bp, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded dataset: {df.shape}")
            break
    if df is not None:
        break

if df is None:
    raise FileNotFoundError("No dataset found!")

# Target preprocessing
df = df.dropna(subset=[target_column]).copy()
y_original = df[target_column].copy()

# Target transformation for better distribution
print(f"Original target range: {y_original.min():.2f} - {y_original.max():.2f}")
print(f"Original target mean±std: {y_original.mean():.2f}±{y_original.std():.2f}")

# Apply Box-Cox transformation to target for normality
from scipy import stats
y_transformed, lambda_param = stats.boxcox(y_original + 1)  # +1 to handle any zeros
print(f"Box-Cox lambda: {lambda_param:.3f}")

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n🧬 ADVANCED FEATURE ENGINEERING")
print("-" * 35)

X = df.drop(columns=[target_column])

# Handle categorical variables with target encoding
from sklearn.preprocessing import LabelEncoder
categorical_cols = X.select_dtypes(exclude=[np.number]).columns
print(f"Categorical columns: {len(categorical_cols)}")

# Target encoding for categorical variables
for col in categorical_cols:
    if col in X.columns:
        # Simple target encoding
        target_means = df.groupby(col)[target_column].mean()
        X[f'{col}_target_encoded'] = X[col].map(target_means)
        X[f'{col}_target_encoded'] = X[f'{col}_target_encoded'].fillna(target_means.mean())
        
        # Keep original as label encoded
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))

# Keep only numeric features
X = X.select_dtypes(include=[np.number])
print(f"Total features after categorical encoding: {X.shape[1]}")

# Advanced correlation analysis
corr_with_target = X.corrwith(y_original).abs().sort_values(ascending=False)
high_corr_features = corr_with_target[corr_with_target > 0.15]  # Lowered threshold
print(f"Features with |correlation| > 0.15: {len(high_corr_features)}")

# Medical domain-specific feature engineering
print("Creating medical domain features...")

# HbA1c-related features
hba1c_patterns = ['hba1c', 'HBA1C', 'hemoglobin']
hba1c_cols = []
for pattern in hba1c_patterns:
    hba1c_cols.extend([col for col in X.columns if pattern.lower() in col.lower()])

# Diabetes risk factors
diabetes_indicators = ['glucose', 'sugar', 'bmi', 'weight', 'age', 'bp', 'pressure', 'insulin']
diabetes_cols = []
for indicator in diabetes_indicators:
    diabetes_cols.extend([col for col in X.columns if indicator.lower() in col.lower()])

diabetes_cols = list(set(diabetes_cols))
print(f"Found {len(diabetes_cols)} diabetes-related features")

# Create composite diabetes risk score
if len(diabetes_cols) >= 3:
    # Normalize and create weighted risk score
    diabetes_features = X[diabetes_cols].copy()
    diabetes_normalized = (diabetes_features - diabetes_features.min()) / (diabetes_features.max() - diabetes_features.min() + 1e-8)
    
    # Weight by correlation with target
    weights = []
    for col in diabetes_cols:
        corr = abs(X[col].corr(y_original))
        weights.append(corr if not np.isnan(corr) else 0)
    weights = np.array(weights)
    weights = weights / (weights.sum() + 1e-8)
    
    X['diabetes_risk_weighted'] = np.average(diabetes_normalized, weights=weights, axis=1)
    print("✅ Created weighted diabetes risk score")

# Top feature interactions (only between most predictive features)
top_features = high_corr_features.head(5).index.tolist()
for i, feat1 in enumerate(top_features):
    for feat2 in top_features[i+1:]:
        if feat1 in X.columns and feat2 in X.columns:
            X[f'{feat1}_mult_{feat2}'] = X[feat1] * X[feat2]
            
            # Ratio features
            denom = X[feat2].replace(0, np.finfo(float).eps)
            X[f'{feat1}_ratio_{feat2}'] = X[feat1] / denom

print(f"Features after medical engineering: {X.shape[1]}")

# Clean and impute
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Advanced feature selection using multiple methods
print("\n🎯 MULTI-METHOD FEATURE SELECTION")
print("-" * 40)

# 1. Correlation-based selection
corr_selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
corr_selected = corr_selector.fit_transform(X, y_original)
corr_features = X.columns[corr_selector.get_support()]

# 2. Mutual information selection
mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(50, X.shape[1]))
mi_selected = mi_selector.fit_transform(X, y_original)
mi_features = X.columns[mi_selector.get_support()]

# 3. Combine selections (union approach)
combined_features = list(set(corr_features) | set(mi_features))
X_selected = X[combined_features]

print(f"F-regression selected: {len(corr_features)}")
print(f"Mutual info selected: {len(mi_features)}")
print(f"Combined features: {len(combined_features)}")

# Target-aware scaling
print("\n📊 TARGET-AWARE PREPROCESSING")
print("-" * 35)

# Use PowerTransformer for better normalization
scaler = PowerTransformer(method='yeo-johnson', standardize=True)
X_scaled = scaler.fit_transform(X_selected)

# Stratified split based on target quantiles
y_quantiles = pd.qcut(y_original, q=5, labels=False)
X_train, X_test, y_train, y_test, y_train_transformed, y_test_transformed = train_test_split(
    X_scaled, y_original, y_transformed, test_size=0.2, random_state=42, stratify=y_quantiles
)

print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Target distribution preserved in split")

# =============================================================================
# ULTRA-OPTIMIZED MODELS
# =============================================================================
print("\n🚀 ULTRA-OPTIMIZED MODELS")
print("-" * 30)

def optimize_xgb_ultra(X, y, n_trials=100):
    """Ultra-optimized XGBoost with extensive hyperparameter search"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_lgb_ultra(X, y, n_trials=100):
    """Ultra-optimized LightGBM"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True
        }
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_catboost(X, y, n_trials=100):
    """Optimize CatBoost for mixed data types"""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
            'random_state': 42,
            'verbose': False
        }
        model = CatBoostRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_neural_network(X, y, n_trials=50):
    """Optimize Neural Network"""
    def objective(trial):
        params = {
            'hidden_layer_sizes': (trial.suggest_int('neurons', 50, 200),),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('lr', 0.001, 0.1, log=True),
            'max_iter': 1000,
            'random_state': 42
        }
        model = MLPRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# Optimize all models
print("Optimizing XGBoost (100 trials)...")
best_xgb_params, best_xgb_mae = optimize_xgb_ultra(X_train, y_train, n_trials=100)
xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)
print(f"✅ XGBoost MAE: {best_xgb_mae:.3f}")

print("Optimizing LightGBM (100 trials)...")
best_lgb_params, best_lgb_mae = optimize_lgb_ultra(X_train, y_train, n_trials=100)
lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(X_train, y_train)
print(f"✅ LightGBM MAE: {best_lgb_mae:.3f}")

print("Optimizing CatBoost (100 trials)...")
best_cat_params, best_cat_mae = optimize_catboost(X_train, y_train, n_trials=100)
cat_model = CatBoostRegressor(**best_cat_params)
cat_model.fit(X_train, y_train)
print(f"✅ CatBoost MAE: {best_cat_mae:.3f}")

print("Optimizing Neural Network (50 trials)...")
best_nn_params, best_nn_mae = optimize_neural_network(X_train, y_train, n_trials=50)
nn_model = MLPRegressor(**best_nn_params)
nn_model.fit(X_train, y_train)
print(f"✅ Neural Network MAE: {best_nn_mae:.3f}")

# =============================================================================
# ADVANCED ENSEMBLE METHODS
# =============================================================================
print("\n🎯 ADVANCED ENSEMBLE METHODS")
print("-" * 35)

# Define base models
base_models = {
    'xgb': xgb.XGBRegressor(**best_xgb_params),
    'lgb': lgb.LGBMRegressor(**best_lgb_params),
    'cat': CatBoostRegressor(**best_cat_params),
    'nn': MLPRegressor(**best_nn_params)
}

# 1. Out-of-fold stacking with multiple meta-learners
n_folds = 7  # Increased folds for better generalization
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

print("Generating out-of-fold predictions...")
for model_idx, (model_name, model) in enumerate(base_models.items()):
    oof_pred = np.zeros(X_train.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        
        val_pred = model_clone.predict(X_fold_val)
        oof_pred[val_idx] = val_pred
        
        test_preds[:, model_idx] += model_clone.predict(X_test) / n_folds
    
    oof_preds[:, model_idx] = oof_pred
    oof_mae = mean_absolute_error(y_train, oof_pred)
    print(f"  {model_name} OOF MAE: {oof_mae:.3f}")

# 2. Multiple meta-learners ensemble
print("\nTesting multiple meta-learners...")

meta_learners = {
    'ridge': Ridge(alpha=1.0),
    'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'lgb_meta': lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
}

best_meta_mae = float('inf')
best_meta_model = None
best_meta_name = None

for meta_name, meta_model in meta_learners.items():
    meta_model.fit(oof_preds, y_train)
    meta_pred = meta_model.predict(oof_preds)
    meta_mae = mean_absolute_error(y_train, meta_pred)
    print(f"  {meta_name} meta MAE: {meta_mae:.3f}")
    
    if meta_mae < best_meta_mae:
        best_meta_mae = meta_mae
        best_meta_model = meta_model
        best_meta_name = meta_name

print(f"Best meta-learner: {best_meta_name} (MAE: {best_meta_mae:.3f})")

# 3. Final predictions with best meta-learner
final_test_preds = best_meta_model.predict(test_preds)
final_train_preds = best_meta_model.predict(oof_preds)

# 4. Apply inverse Box-Cox transformation
final_test_preds_original = stats.inv_boxcox(final_test_preds, lambda_param) - 1
final_train_preds_original = stats.inv_boxcox(final_train_preds, lambda_param) - 1

# Calculate final metrics
train_mae = mean_absolute_error(y_train, final_train_preds_original)
test_mae = mean_absolute_error(y_test, final_test_preds_original)
test_rmse = np.sqrt(mean_squared_error(y_test, final_test_preds_original))
test_r2 = np.corrcoef(y_test, final_test_preds_original)[0,1]**2

# Clinical thresholds
errors = np.abs(y_test - final_test_preds_original)
excellent = (errors <= 0.5).mean() * 100
good = (errors <= 1.0).mean() * 100
fair = (errors <= 1.5).mean() * 100

print(f"\n🏆 ULTRA-OPTIMIZED RESULTS:")
print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE: {test_mae:.3f}")
print(f"   Test RMSE: {test_rmse:.3f}")
print(f"   Test R²: {test_r2:.3f}")
print(f"   Excellent (±0.5): {excellent:.1f}%")
print(f"   Good (±1.0): {good:.1f}%")
print(f"   Fair (±1.5): {fair:.1f}%")

if test_mae < 0.5:
    print("\n🎉 TARGET ACHIEVED! MAE < 0.5!")
elif test_mae < 0.6:
    print("\n🚀 EXCELLENT! MAE < 0.6 achieved!")
elif test_mae < 0.7:
    print("\n✅ GOOD! MAE < 0.7 achieved!")
else:
    print(f"\n⚠️ Current MAE: {test_mae:.3f} - Continue optimization")

# =============================================================================
# SAVE ULTRA-OPTIMIZED MODEL
# =============================================================================
print("\n💾 SAVING ULTRA-OPTIMIZED MODEL")
print("-" * 35)

class UltraStackRidge:
    def __init__(self, base_models, meta_model, scaler, features, lambda_param):
        self.base_models = base_models
        self.meta_model = meta_model
        self.scaler = scaler
        self.features = features
        self.lambda_param = lambda_param
    
    def predict(self, X_new):
        X_processed = X_new[self.features]
        X_scaled = self.scaler.transform(X_processed)
        
        base_preds = np.zeros((X_scaled.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            base_preds[:, i] = model.predict(X_scaled)
        
        meta_pred = self.meta_model.predict(base_preds)
        return stats.inv_boxcox(meta_pred, self.lambda_param) - 1

final_ensemble = UltraStackRidge(base_models, best_meta_model, scaler, combined_features, lambda_param)

os.makedirs('models', exist_ok=True)
model_filename = f"models/ultra_stack_ridge_mae_{test_mae:.3f}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(final_ensemble, f)

print(f"✅ Model saved: {model_filename}")
print("=" * 60)