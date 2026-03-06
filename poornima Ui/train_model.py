"""
DiabeSense+ Model Training Pipeline (Deep Training Mode)
Predicts post-intervention HbA1c (PostBLHBA1C) using:
  1. Imputation Tournament (Mean, Median, KNN, MICE, Zero)
  2. 10-Fold Cross-Validation throughout
  3. Hyperparameter Tuning for ALL base learners via RandomizedSearchCV
  4. Polynomial Interaction Features for key numeric columns
  5. Stacking Ensemble (tuned Ridge, Lasso, ElasticNet, BayesianRidge,
     SVR, RF, GBR, KNN, XGB → Ridge meta-learner)
  6. Standalone XGBRegressor (tuned) for SHAP explanations
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
from scipy.stats import uniform, randint, loguniform
import shap
import os
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
N_FOLDS = 10
TUNE_FOLDS = 5          # CV folds inside each hyperopt (faster)
XGB_SEARCH_ITER = 100   # XGB gets the most search budget
TREE_SEARCH_ITER = 50   # RF, GBR
OTHER_SEARCH_ITER = 30  # SVR, Ridge, Lasso, ElasticNet, KNN
print(f"[CONFIG] {N_JOBS} threads | {N_FOLDS}-fold eval | "
      f"XGB:{XGB_SEARCH_ITER} RF/GBR:{TREE_SEARCH_ITER} others:{OTHER_SEARCH_ITER} HP iterations")

DATA_FILES = [
    os.path.join(BASE_DIR, "apolloCombined.csv"),
    os.path.join(BASE_DIR, "ApolloFormat_nmbfinalDiabetes (4).csv"),
    os.path.join(BASE_DIR, "ApolloFormat_nmbfinalnewDiabetes (3).csv"),
    os.path.join(BASE_DIR, "ApolloFormat_PrePostFinal (3).csv"),
]

TARGET_COL = "PostBLHBA1C"
OUTPUT_PATH = os.path.join(BASE_DIR, "diabesense_artifacts.pkl")

COLUMN_RENAMES = {
    "postblage": "PostBLAge",
    "PreRgender": "PreBLGender",
}

DESIRED_FEATURES = [
    "PostBLAge", "PreBLGender", "PreRarea", "PreRmaritalstatus",
    "PreReducation", "PreRpresentoccupation",
    "PreRdiafather", "PreRdiamother", "PreRdiabrother", "PreRdiasister",
    "current_smoking", "current_alcohol",
    "PreRsleepquality",
    "PreRmildactivityduration",
    "PreRmoderate", "PreRmoderateduration",
    "PreRvigorous", "PreRvigorousduration",
    "PreRskipbreakfast", "PreRlessfruit", "PreRlessvegetable",
    "PreRmilk", "PreRmeat", "PreRfriedfood", "PreRsweet",
    "PreRwaist", "PreRBMI",
    "PreRsystolicfirst", "PreRdiastolicfirst",
    "PreBLPPBS", "PreBLFBS", "PreBLHBA1C",
    "PreBLCHOLESTEROL", "PreBLTRIGLYCERIDES",
    "Diabetic_Duration", "PostRgroupname",
]

NUMERIC_FEATURES = [
    "PostBLAge", "PreRwaist", "PreRBMI",
    "PreRsystolicfirst", "PreRdiastolicfirst",
    "PreBLPPBS", "PreBLFBS", "PreBLHBA1C",
    "PreBLCHOLESTEROL", "PreBLTRIGLYCERIDES",
    "Diabetic_Duration",
    "PreRmildactivityduration", "PreRmoderateduration", "PreRvigorousduration",
]

CATEGORICAL_FEATURES = [
    "PreBLGender", "PreRarea", "PreRmaritalstatus",
    "PreReducation", "PreRpresentoccupation",
    "PreRdiafather", "PreRdiamother", "PreRdiabrother", "PreRdiasister",
    "current_smoking", "current_alcohol",
    "PreRsleepquality",
    "PreRmoderate", "PreRvigorous",
    "PreRskipbreakfast", "PreRlessfruit", "PreRlessvegetable",
    "PreRmilk", "PreRmeat", "PreRfriedfood", "PreRsweet",
    "PostRgroupname",
]

POLY_FEATURES = [
    "PreBLHBA1C", "PreBLFBS", "PreBLPPBS", "PreRBMI",
    "PostBLAge", "Diabetic_Duration",
]


# ============================================================================
# DATA LOADING & MERGING
# ============================================================================

def load_and_merge_data(file_list):
    print("\n[1/8] Loading and merging datasets...")
    frames = []
    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"    [!] Missing: {os.path.basename(fpath)}")
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        df.rename(columns=COLUMN_RENAMES, inplace=True)
        if "PreRBMI" in df.columns:
            df["PreRBMI"] = pd.to_numeric(df["PreRBMI"], errors="coerce")
        available = [c for c in DESIRED_FEATURES + [TARGET_COL] if c in df.columns]
        df = df[available]
        df["_source"] = os.path.basename(fpath)
        frames.append(df)
        print(f"    [OK] {os.path.basename(fpath)}: {df.shape[0]} rows, {len(available)} cols")

    if not frames:
        raise ValueError("No data files loaded!")

    combined = pd.concat(frames, ignore_index=True)
    combined.drop(columns=["_source"], inplace=True)

    n0 = len(combined)
    combined = combined.dropna(subset=[TARGET_COL])
    n1 = len(combined)
    combined = combined.drop_duplicates()
    n2 = len(combined)
    print(f"    Combined: {n0} -> {n1} (nulls) -> {n2} (dedup)")
    return combined


# ============================================================================
# IMPUTATION TOURNAMENT
# ============================================================================

def run_imputation_tournament(df, numeric_cols):
    print("\n[2/8] Imputation Tournament...")
    df_valid = df[numeric_cols].dropna()

    if df_valid.shape[0] < 10:
        print("    [!] Sparse data, defaulting to MICE.")
        try:
            imp = IterativeImputer(estimator=RandomForestRegressor(
                n_jobs=N_JOBS, max_depth=5, n_estimators=10), max_iter=10, random_state=42)
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])
        except Exception:
            df[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])
        return df

    np.random.seed(42)
    X_true = df_valid.values
    mask = np.random.rand(*X_true.shape) < 0.1
    X_masked = X_true.copy()
    X_masked[mask] = np.nan

    strategies = {
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'KNN_5': KNNImputer(n_neighbors=5),
        'MICE_RF': IterativeImputer(estimator=RandomForestRegressor(
            n_jobs=N_JOBS, max_depth=5, n_estimators=10), max_iter=10, random_state=42),
        'Zero': SimpleImputer(strategy='constant', fill_value=0),
    }

    best_rmse, best_name, best_model = float('inf'), 'Mean', strategies['Mean']
    for name, model in strategies.items():
        try:
            t0 = time.time()
            X_imp = model.fit_transform(X_masked)
            rmse = np.sqrt(mean_squared_error(X_true[mask], X_imp[mask]))
            dt = time.time() - t0
            marker = ""
            if rmse < best_rmse:
                best_rmse, best_name, best_model = rmse, name, model
                marker = " <-- best"
            print(f"    {name:12s}: RMSE={rmse:.4f}  ({dt:.1f}s){marker}")
        except Exception as e:
            print(f"    {name:12s}: FAILED ({e})")

    print(f"    WINNER: {best_name} (RMSE: {best_rmse:.4f})")
    try:
        df[numeric_cols] = clone(best_model).fit_transform(df[numeric_cols])
    except Exception as e2:
        print(f"    [!] Winner failed ({e2}). Falling back to Mean.")
        df[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])
    return df


# ============================================================================
# PREPROCESSING + POLYNOMIAL FEATURES
# ============================================================================

def create_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL_FEATURES),
    ], remainder='drop')


def get_feature_names_after_encoding(preprocessor):
    ohe_names = preprocessor.named_transformers_['cat']['encoder'] \
        .get_feature_names_out(CATEGORICAL_FEATURES)
    return list(NUMERIC_FEATURES) + list(ohe_names)


def add_polynomial_features(X_df, feature_names):
    """Add degree-2 interaction features for key numeric columns."""
    poly_indices = [i for i, f in enumerate(feature_names) if f in POLY_FEATURES]
    if not poly_indices:
        return X_df, feature_names

    X_poly_src = X_df.iloc[:, poly_indices].values
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_poly_src)

    poly_names = poly.get_feature_names_out([feature_names[i] for i in poly_indices])
    new_poly_names = [n for n in poly_names if ' ' in n]
    new_poly_data = X_poly[:, len(poly_indices):]

    poly_df = pd.DataFrame(new_poly_data, columns=new_poly_names, index=X_df.index)
    X_augmented = pd.concat([X_df, poly_df], axis=1)
    augmented_names = list(feature_names) + list(new_poly_names)
    return X_augmented, augmented_names


# ============================================================================
# HYPERPARAMETER TUNING FOR ALL BASE LEARNERS
# ============================================================================

def _run_search(name, estimator, param_dist, X, y, n_iter, n_jobs_cv):
    """Run RandomizedSearchCV for one estimator."""
    t0 = time.time()
    kf = KFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=n_iter, cv=kf,
        scoring='neg_mean_absolute_error', n_jobs=n_jobs_cv,
        random_state=42, verbose=0
    )
    search.fit(X, y)
    dt = time.time() - t0
    mae = -search.best_score_
    print(f"    {name:18s}: CV-MAE={mae:.4f}  ({dt:.1f}s, {n_iter}×{TUNE_FOLDS} fits)")
    return search.best_estimator_


def tune_all_learners(X_train, y_train):
    """Tune all 9 base learners with RandomizedSearchCV."""
    print(f"\n[5/8] Hyperparameter Tuning for ALL base learners...")
    print(f"    {TUNE_FOLDS}-fold CV for each | {N_JOBS} threads")

    # Split jobs: give tree models more CPU, linear models less
    half_jobs = max(1, N_JOBS // 2)
    tuned = {}

    # --- 1. Ridge ---
    tuned['ridge'] = _run_search('Ridge',
        Pipeline([('sc', StandardScaler()), ('m', Ridge())]),
        {'m__alpha': loguniform(1e-3, 100)},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 2. Lasso ---
    tuned['lasso'] = _run_search('Lasso',
        Pipeline([('sc', StandardScaler()), ('m', Lasso(max_iter=10000))]),
        {'m__alpha': loguniform(1e-5, 1)},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 3. ElasticNet ---
    tuned['elastic'] = _run_search('ElasticNet',
        Pipeline([('sc', StandardScaler()), ('m', ElasticNet(max_iter=10000))]),
        {'m__alpha': loguniform(1e-5, 1), 'm__l1_ratio': uniform(0.1, 0.8)},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 4. BayesianRidge ---
    tuned['bayesian'] = _run_search('BayesianRidge',
        Pipeline([('sc', StandardScaler()), ('m', BayesianRidge())]),
        {'m__alpha_1': loguniform(1e-7, 1e-3), 'm__alpha_2': loguniform(1e-7, 1e-3),
         'm__lambda_1': loguniform(1e-7, 1e-3), 'm__lambda_2': loguniform(1e-7, 1e-3),
         'm__max_iter': randint(100, 1000)},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 5. SVR (Optimized: LinearSVR is much faster for N=5000+) ---
    tuned['svr'] = _run_search('LinearSVR',
        Pipeline([('sc', StandardScaler()), ('m', LinearSVR(max_iter=10000, dual='auto'))]),
        {'m__C': loguniform(0.1, 100), 'm__epsilon': uniform(0, 0.5),
         'm__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 6. Random Forest ---
    tuned['rf'] = _run_search('RandomForest',
        RandomForestRegressor(random_state=42, n_jobs=half_jobs),
        {'n_estimators': randint(200, 1000), 'max_depth': [None, 10, 15, 20, 30],
         'min_samples_split': randint(2, 10), 'min_samples_leaf': randint(1, 5),
         'max_features': ['sqrt', 'log2', 0.5, 0.8, None]},
        X_train, y_train, TREE_SEARCH_ITER, half_jobs)

    # --- 7. GradientBoosting ---
    tuned['gbr'] = _run_search('GradientBoosting',
        GradientBoostingRegressor(random_state=42),
        {'n_estimators': randint(200, 1000), 'learning_rate': uniform(0.005, 0.1),
         'max_depth': randint(3, 8), 'subsample': uniform(0.6, 0.4),
         'min_samples_split': randint(2, 10), 'min_samples_leaf': randint(1, 5)},
        X_train, y_train, TREE_SEARCH_ITER, N_JOBS)

    # --- 8. KNN ---
    tuned['knn'] = _run_search('KNN',
        Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor())]),
        {'m__n_neighbors': randint(3, 30), 'm__weights': ['uniform', 'distance'],
         'm__p': [1, 2], 'm__leaf_size': randint(10, 50)},
        X_train, y_train, OTHER_SEARCH_ITER, N_JOBS)

    # --- 9. XGBoost (gets the biggest search budget) ---
    print(f"\n    XGBoost tuning ({XGB_SEARCH_ITER} iterations)...")
    tuned['xgb'] = _run_search('XGBoost',
        xgb.XGBRegressor(random_state=42, eval_metric='mae', n_jobs=half_jobs),
        {'n_estimators': randint(500, 2000), 'learning_rate': uniform(0.005, 0.1),
         'max_depth': randint(3, 10), 'min_child_weight': randint(1, 10),
         'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.5, 0.5),
         'gamma': uniform(0, 0.5), 'reg_alpha': uniform(0, 1.0),
         'reg_lambda': uniform(0.5, 2.0)},
        X_train, y_train, XGB_SEARCH_ITER, half_jobs)

    return tuned


# ============================================================================
# STACKING ENSEMBLE (all tuned learners)
# ============================================================================

def train_stacking_ensemble(X_train, y_train, tuned_learners):
    """Stack all tuned learners with Ridge meta-learner."""
    print(f"\n[6/8] Stacking Ensemble (all tuned learners)...")

    estimators = [
        ('ridge', tuned_learners['ridge']),
        ('lasso', tuned_learners['lasso']),
        ('elastic', tuned_learners['elastic']),
        ('bayesian', tuned_learners['bayesian']),
        ('svr', tuned_learners['svr']),
        ('rf', tuned_learners['rf']),
        ('gbr', tuned_learners['gbr']),
        ('knn', tuned_learners['knn']),
        ('xgb', clone(tuned_learners['xgb'])),
    ]

    t0 = time.time()
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=TUNE_FOLDS,
        n_jobs=N_JOBS,
        passthrough=False,
    )
    stack.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"    [OK] Stacking trained in {dt:.1f}s ({len(estimators)} tuned learners + Ridge meta)")
    return stack


# ============================================================================
# K-FOLD CROSS-VALIDATION EVALUATION
# ============================================================================

def kfold_evaluate(model, X, y, model_name):
    """10-fold CV with MAE, RMSE, R²."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=N_JOBS)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=N_JOBS))
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=N_JOBS)

    print(f"    [{model_name}] {N_FOLDS}-Fold CV:")
    print(f"      MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"      RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"      R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"      (per-fold MAE: {', '.join(f'{s:.3f}' for s in mae_scores)})")
    return {'mae': mae_scores.mean(), 'rmse': rmse_scores.mean(), 'r2': r2_scores.mean()}


def evaluate_holdout(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"    [{model_name}] Holdout: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(preprocessor, stack_model, xgb_shap_model, feature_names,
                   poly_transformer, poly_indices, output_path):
    print(f"\n[8/8] Saving artifacts...")
    artifacts = {
        'preprocessor': preprocessor,
        'model': stack_model,
        'shap_model': xgb_shap_model,
        'feature_names': feature_names,
        'poly_transformer': poly_transformer,
        'poly_indices': poly_indices,
    }
    joblib.dump(artifacts, output_path)
    print(f"    [OK] Saved to {output_path}")
    print(f"    - model: StackingRegressor (9 tuned learners + Ridge meta)")
    print(f"    - shap_model: XGBRegressor (tuned)")
    print(f"    - features: {len(feature_names)} (incl. poly interactions)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 64)
    print("    DiabeSense+ DEEP Training Pipeline")
    print("    All learners hyperopt-tuned | 10-fold CV | Poly features")
    print(f"    {N_JOBS} threads | XGB:{XGB_SEARCH_ITER} RF/GBR:{TREE_SEARCH_ITER} "
          f"others:{OTHER_SEARCH_ITER} search iters")
    print("=" * 64)
    total_t0 = time.time()

    # 1. Load
    df = load_and_merge_data(DATA_FILES)
    available_features = [c for c in DESIRED_FEATURES if c in df.columns]
    missing = [c for c in DESIRED_FEATURES if c not in df.columns]
    if missing:
        print(f"    [!] Missing: {missing}")
    print(f"    {len(available_features)} features | "
          f"Target: mean={df[TARGET_COL].mean():.2f}, std={df[TARGET_COL].std():.2f}, "
          f"range=[{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}]")

    # 2. Imputation
    numeric_in_data = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = run_imputation_tournament(df, numeric_in_data)

    # 3. Split
    X = df[available_features]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[3/8] Split: {len(X_train)} train / {len(X_test)} test")

    # 4. Preprocess + poly features
    print("\n[4/8] Preprocessing + Polynomial Features...")
    preprocessor = create_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    base_names = get_feature_names_after_encoding(preprocessor)
    X_train_df = pd.DataFrame(X_train_proc, columns=base_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=base_names)

    X_train_df, feature_names = add_polynomial_features(X_train_df, base_names)
    X_test_df, _ = add_polynomial_features(X_test_df, base_names)
    poly_indices = [i for i, f in enumerate(base_names) if f in POLY_FEATURES]
    print(f"    Base: {len(base_names)} | +Poly interactions: {len(feature_names)} total")

    # Writable numpy arrays
    X_train_np = np.ascontiguousarray(X_train_df.values)
    X_test_np = np.ascontiguousarray(X_test_df.values)
    y_train_np = np.ascontiguousarray(y_train.values)

    # 5. Tune ALL learners
    tuned = tune_all_learners(X_train_np, y_train_np)

    # 6. Stacking
    stack_model = train_stacking_ensemble(X_train_np, y_train_np, tuned)

    # 7. Evaluation
    print(f"\n[7/8] Evaluation")
    print("=" * 64)

    print("\n  Holdout Test Set:")
    stack_h = evaluate_holdout(stack_model, X_test_np, y_test, "Stack(9 tuned)")
    xgb_h = evaluate_holdout(tuned['xgb'], X_test_np, y_test, "Tuned XGB")

    print(f"\n  {N_FOLDS}-Fold CV (full data):")
    X_full_np = np.ascontiguousarray(pd.concat([X_train_df, X_test_df], ignore_index=True).values)
    y_full_np = np.ascontiguousarray(pd.concat([y_train, y_test], ignore_index=True).values)
    kfold_evaluate(clone(tuned['xgb']), X_full_np, y_full_np, "Tuned XGB")

    # Feature importances
    fi = pd.DataFrame({'feature': feature_names, 'importance': tuned['xgb'].feature_importances_
                        }).sort_values('importance', ascending=False)
    print(f"\n    Top 15 Feature Importances:")
    for _, row in fi.head(15).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # SHAP
    print("\n    SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(tuned['xgb'])
    sample_shap = explainer.shap_values(X_test_df.iloc[:1])
    print(f"    [OK] SHAP shape: {sample_shap.shape}")

    # 8. Save
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly.fit(X_train_df.iloc[:, poly_indices].values)
    save_artifacts(preprocessor, stack_model, tuned['xgb'], feature_names,
                   poly, poly_indices, OUTPUT_PATH)

    total_t = time.time() - total_t0
    print("\n" + "=" * 64)
    print(f"    DEEP TRAINING COMPLETE! {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"    Stack  -> MAE: {stack_h['mae']:.4f} | R²: {stack_h['r2']:.4f}")
    print(f"    XGB    -> MAE: {xgb_h['mae']:.4f} | R²: {xgb_h['r2']:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
