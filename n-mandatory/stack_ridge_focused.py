#!/usr/bin/env python3
"""
🩺 Focused Stack Ridge Ensemble for HbA1c Prediction
Target: MAE < 0.5 with minimal, essential components only
"""

import sys, os, warnings
import numpy as np, pandas as pd
import pickle
from datetime import datetime
import subprocess

warnings.filterwarnings('ignore')

# Install essential packages only
def install_if_needed(package):
    try:
        __import__(package.split('==')[0])
        print(f"✅ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

essential_packages = [
    'scikit-learn>=1.3.0',
    'optuna',
    'xgboost',
    'lightgbm'
]

for pkg in essential_packages:
    install_if_needed(pkg)

# Essential imports only
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🎯 FOCUSED STACK RIDGE ENSEMBLE - TARGET MAE < 0.5")
print("=" * 60)

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n📊 LOADING DATA")
print("-" * 30)

base_paths = ['./final_imputed_data/', 'final_imputed_data/', './']
dataset_files = [
    'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
    'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
    'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
]
dataset_names = ['nmbfinalDiabetes_4', 'nmbfinalnewDiabetes_3', 'PrePostFinal_3']
target_column = 'PostBLHBA1C'

# Load first available dataset
df = None
for name, file in zip(dataset_names, dataset_files):
    for bp in base_paths:
        path = os.path.join(bp, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded {name}: {df.shape}")
            break
    if df is not None:
        break

if df is None:
    raise FileNotFoundError("No dataset found!")

df = df.dropna(subset=[target_column]).copy()
print(f"Dataset after removing missing targets: {df.shape}")
print(f"Target range: {df[target_column].min():.2f} - {df[target_column].max():.2f}")

# =============================================================================
# ESSENTIAL FEATURE ENGINEERING
# =============================================================================
print("\n🛠️ ESSENTIAL FEATURE ENGINEERING")
print("-" * 40)

X = df.drop(columns=[target_column])
y = df[target_column]

# Handle categorical variables
categorical_cols = X.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))

# Keep only numeric features
X = X.select_dtypes(include=[np.number])
print(f"Numeric features: {X.shape[1]}")

# Feature correlation analysis
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
high_corr_features = corr_with_target[corr_with_target > 0.2]
print(f"High correlation features (|r| > 0.2): {len(high_corr_features)}")

# Create essential interaction features for top predictors
if len(high_corr_features) >= 2:
    top_features = high_corr_features.head(3).index.tolist()
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if feat1 in X.columns and feat2 in X.columns:
                X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                # Safe division
                denominator = X[feat2].replace(0, np.finfo(float).eps)
                X[f'{feat1}_div_{feat2}'] = X[feat1] / denominator

# Polynomial features for best predictor
if len(high_corr_features) >= 1:
    best_feature = high_corr_features.index[0]
    X[f'{best_feature}_squared'] = X[best_feature] ** 2
    X[f'{best_feature}_log'] = np.log1p(np.abs(X[best_feature]))

# Domain-specific diabetes feature engineering
print("Creating domain-specific diabetes features...")

# Find potential diabetes-related columns by name patterns
diabetes_patterns = ['hba1c', 'HBA1C', 'glucose', 'sugar', 'diabetes', 'insulin', 
                    'bmi', 'BMI', 'weight', 'age', 'bp', 'BP', 'pressure']
diabetes_cols = []
for pattern in diabetes_patterns:
    diabetes_cols.extend([col for col in X.columns if pattern.lower() in col.lower()])
diabetes_cols = list(set(diabetes_cols))  # Remove duplicates

print(f"Found {len(diabetes_cols)} potential diabetes-related features")

# Create diabetes risk score from available features
if diabetes_cols:
    # Normalize diabetes-related features and create risk score
    risk_features = []
    for col in diabetes_cols[:5]:  # Use top 5 diabetes features
        if col in X.columns:
            normalized = (X[col] - X[col].min()) / (X[col].max() - X[col].min() + 1e-8)
            risk_features.append(normalized)
    
    if risk_features:
        X['diabetes_risk_score'] = np.mean(risk_features, axis=0)
        print("✅ Created diabetes_risk_score feature")

# HbA1c specific transformations (medical knowledge)
hba1c_cols = [col for col in X.columns if 'hba1c' in col.lower() or 'HBA1C' in col]
for col in hba1c_cols:
    if col in X.columns:
        # HbA1c categories based on medical standards
        X[f'{col}_category'] = pd.cut(X[col], 
                                     bins=[0, 5.7, 6.5, 9.0, float('inf')], 
                                     labels=[0, 1, 2, 3],  # Normal, Prediabetic, Diabetic, Uncontrolled
                                     include_lowest=True).astype(float)
        
        # HbA1c change rate (if multiple HbA1c columns exist)
        other_hba1c = [c for c in hba1c_cols if c != col]
        for other_col in other_hba1c:
            if other_col in X.columns:
                X[f'{col}_change_rate'] = X[col] - X[other_col]
                break

# Age-related interactions (diabetes risk increases with age)
age_cols = [col for col in X.columns if 'age' in col.lower()]
for age_col in age_cols:
    if age_col in X.columns and len(diabetes_cols) > 0:
        main_diabetes_feature = diabetes_cols[0]
        if main_diabetes_feature in X.columns:
            X[f'{age_col}_x_{main_diabetes_feature}'] = X[age_col] * X[main_diabetes_feature]

print(f"Enhanced features after diabetes-specific engineering: {X.shape[1]}")

# Clean data
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"Enhanced features: {X.shape[1]}")

# Outlier detection and removal
from sklearn.ensemble import IsolationForest
print("\n🔍 OUTLIER DETECTION")
print("-" * 25)

# Remove extreme outliers that could hurt performance
isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Remove top 5% outliers
outlier_mask = isolation_forest.fit_predict(X) == 1

print(f"Original dataset size: {len(X)}")
X_clean = X[outlier_mask]
y_clean = y[outlier_mask]
print(f"After outlier removal: {len(X_clean)} (removed {len(X) - len(X_clean)} outliers)")

# Update X and y
X, y = X_clean.copy(), y_clean.copy()

# Advanced feature selection - keep more features for better performance
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Use both correlation and mutual information for feature selection
print("\n🎯 ADVANCED FEATURE SELECTION")
print("-" * 35)

# Mutual information selection
mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(40, X.shape[1]))
X_mi = mi_selector.fit_transform(X, y)
mi_scores = mi_selector.scores_
mi_features = X.columns[mi_selector.get_support()]

# F-regression selection
f_selector = SelectKBest(score_func=f_regression, k=min(40, X.shape[1]))
X_f = f_selector.fit_transform(X, y)
f_scores = f_selector.scores_
f_features = X.columns[f_selector.get_support()]

# Combine both selections (union of features)
combined_features = list(set(mi_features) | set(f_features))
X = X[combined_features]

print(f"Mutual Info selected: {len(mi_features)} features")
print(f"F-regression selected: {len(f_features)} features")
print(f"Combined selection: {len(combined_features)} features")
print(f"Top 5 combined features: {combined_features[:5]}")

# Scale features with robust scaling for better outlier handling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# =============================================================================
# OPTIMIZED BASE MODELS
# =============================================================================
print("\n🚀 OPTIMIZED BASE MODELS")
print("-" * 30)

def optimize_xgb(X, y, n_trials=30):
    """Optimize XGBoost with regularization to prevent overfitting"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 6),  # Reduced depth
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),  # Prevent overfitting
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_lgb(X, y, n_trials=30):
    """Optimize LightGBM with regularization to prevent overfitting"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 6),  # Reduced depth
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Prevent overfitting
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True  # Remove warnings
        }
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# Optimize XGBoost
print("Optimizing XGBoost...")
best_xgb_params, best_xgb_mae = optimize_xgb(X_train, y_train, n_trials=80)
xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)
print(f"✅ XGBoost MAE: {best_xgb_mae:.3f}")

# Optimize LightGBM
print("Optimizing LightGBM...")
best_lgb_params, best_lgb_mae = optimize_lgb(X_train, y_train, n_trials=80)
lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(X_train, y_train)
print(f"✅ LightGBM MAE: {best_lgb_mae:.3f}")

# Random Forest baseline
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"✅ Random Forest MAE: {-rf_cv.mean():.3f}")

# =============================================================================
# OUT-OF-FOLD STACKING ENSEMBLE
# =============================================================================
print("\n🎯 OUT-OF-FOLD STACK RIDGE ENSEMBLE")
print("-" * 40)

# Create out-of-fold predictions
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

base_models = {
    'xgb': xgb.XGBRegressor(**best_xgb_params),
    'lgb': lgb.LGBMRegressor(**best_lgb_params),
    'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
}

# Generate out-of-fold predictions
oof_preds = np.zeros((X_train.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

print("Generating out-of-fold predictions...")
for model_idx, (model_name, model) in enumerate(base_models.items()):
    oof_pred = np.zeros(X_train.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model on fold
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        
        # Predict on validation fold
        val_pred = model_clone.predict(X_fold_val)
        oof_pred[val_idx] = val_pred
        
        # Predict on test set
        test_preds[:, model_idx] += model_clone.predict(X_test) / n_folds
    
    oof_preds[:, model_idx] = oof_pred
    
    # Calculate OOF score
    oof_mae = mean_absolute_error(y_train, oof_pred)
    print(f"  {model_name} OOF MAE: {oof_mae:.3f}")

# Train meta-learner with optimized Ridge regression
print("\nTraining optimized meta-learner...")

# Optimize Ridge alpha parameter
from sklearn.model_selection import GridSearchCV
ridge_param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
ridge_cv = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
ridge_cv.fit(oof_preds, y_train)

meta_model = ridge_cv.best_estimator_
print(f"Best Ridge alpha: {ridge_cv.best_params_['alpha']}")

# Final predictions
final_test_preds = meta_model.predict(test_preds)
final_train_preds = meta_model.predict(oof_preds)

# Calculate final metrics
train_mae = mean_absolute_error(y_train, final_train_preds)
test_mae = mean_absolute_error(y_test, final_test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, final_test_preds))
test_r2 = np.corrcoef(y_test, final_test_preds)[0,1]**2

# Clinical thresholds
errors = np.abs(y_test - final_test_preds)
excellent = (errors <= 0.5).mean() * 100
good = (errors <= 1.0).mean() * 100
fair = (errors <= 1.5).mean() * 100

print(f"\n🏆 FINAL STACK RIDGE RESULTS:")
print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE: {test_mae:.3f}")
print(f"   Test RMSE: {test_rmse:.3f}")
print(f"   Test R²: {test_r2:.3f}")
print(f"   Excellent (±0.5): {excellent:.1f}%")
print(f"   Good (±1.0): {good:.1f}%")
print(f"   Fair (±1.5): {fair:.1f}%")

# Performance assessment
if test_mae < 0.5:
    print("\n🎉 TARGET ACHIEVED! MAE < 0.5!")
elif test_mae < 0.6:
    print("\n🚀 EXCELLENT! MAE < 0.6 achieved!")
elif test_mae < 0.7:
    print("\n✅ GOOD! MAE < 0.7 achieved!")
else:
    print(f"\n⚠️ Current MAE: {test_mae:.3f} - Need further optimization")

# =============================================================================
# SAVE FOCUSED MODEL
# =============================================================================
print("\n💾 SAVING FOCUSED STACK RIDGE MODEL")
print("-" * 40)

# Create final ensemble class
class FocusedStackRidge:
    def __init__(self, base_models, meta_model, scaler, selected_features):
        self.base_models = base_models
        self.meta_model = meta_model
        self.scaler = scaler
        self.selected_features = selected_features
    
    def predict(self, X_new):
        # Apply same preprocessing
        X_processed = X_new[self.selected_features]
        X_scaled = self.scaler.transform(X_processed)
        
        # Get base model predictions
        base_preds = np.zeros((X_scaled.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            base_preds[:, i] = model.predict(X_scaled)
        
        # Meta-learner prediction
        return self.meta_model.predict(base_preds)

# Create final model
final_ensemble = FocusedStackRidge(
    base_models=base_models,
    meta_model=meta_model,
    scaler=scaler,
    selected_features=combined_features
)

# Save model
os.makedirs('models', exist_ok=True)
model_filename = f"models/focused_stack_ridge_mae_{test_mae:.3f}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(final_ensemble, f)

print(f"✅ Model saved: {model_filename}")

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Actual_HbA1c': y_test.values,
    'Predicted_HbA1c': final_test_preds,
    'Absolute_Error': errors,
    'Clinical_Agreement': ['Excellent' if e <= 0.5 else 'Good' if e <= 1.0 else 'Fair' if e <= 1.5 else 'Poor' for e in errors]
})

# Save predictions
os.makedirs('outputs', exist_ok=True)
pred_filename = f"outputs/focused_predictions_mae_{test_mae:.3f}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
predictions_df.to_csv(pred_filename, index=False)
print(f"✅ Predictions saved: {pred_filename}")

print(f"\n🎯 FOCUSED STACK RIDGE COMPLETE!")
print(f"📁 Files: {model_filename}, {pred_filename}")
print("=" * 60)