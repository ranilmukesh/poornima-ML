"""
DiabeSense+ Model Training Pipeline
Predicts post-intervention HbA1c (PostBLHBA1C) using:
  1. Imputation Tournament (Mean, Median, KNN, MICE, Zero)
  2. Stacking Ensemble (Ridge, Lasso, ElasticNet, BayesianRidge,
     SVR, RF, GB, KNN, XGB → StackingRegressor)
  3. Standalone XGBRegressor for SHAP explanations
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
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
print(f"[CONFIG] Using {N_JOBS} CPU threads (total cores: {os.cpu_count()})")

DATA_FILES = [
    os.path.join(BASE_DIR, "apolloCombined.csv"),
    os.path.join(BASE_DIR, "ApolloFormat_nmbfinalDiabetes (4).csv"),
    os.path.join(BASE_DIR, "ApolloFormat_nmbfinalnewDiabetes (3).csv"),
    os.path.join(BASE_DIR, "ApolloFormat_PrePostFinal (3).csv"),
]

TARGET_COL = "PostBLHBA1C"
OUTPUT_PATH = os.path.join(BASE_DIR, "diabesense_artifacts.pkl")

# Column rename mapping (actual CSV names -> desired names)
COLUMN_RENAMES = {
    "postblage": "PostBLAge",
    "PreRgender": "PreBLGender",
}

# All desired feature columns (excluding target)
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

# Feature type definitions
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


# ============================================================================
# DATA LOADING & MERGING
# ============================================================================

def load_and_merge_data(file_list: list) -> pd.DataFrame:
    """Load all CSV files, rename columns, filter to desired features, merge."""
    print("[*] Loading and merging datasets...")
    frames = []

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"    [!] File not found, skipping: {os.path.basename(fpath)}")
            continue

        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()

        # Apply column renames
        df.rename(columns=COLUMN_RENAMES, inplace=True)

        # Ensure BMI is numeric (some files have it as string)
        if "PreRBMI" in df.columns:
            df["PreRBMI"] = pd.to_numeric(df["PreRBMI"], errors="coerce")

        # Keep only columns that exist in this file
        available = [c for c in DESIRED_FEATURES + [TARGET_COL] if c in df.columns]
        df = df[available]

        # Add source tag
        df["_source"] = os.path.basename(fpath)
        frames.append(df)
        print(f"    [OK] {os.path.basename(fpath)}: {df.shape[0]} rows, {len(available)} cols")

    if not frames:
        raise ValueError("No data files loaded!")

    # Merge all
    combined = pd.concat(frames, ignore_index=True)
    combined.drop(columns=["_source"], inplace=True)

    # Drop rows where target is missing
    before = len(combined)
    combined = combined.dropna(subset=[TARGET_COL])
    after = len(combined)
    print(f"\n[*] Combined: {before} rows -> {after} rows (dropped {before - after} null targets)")

    # Drop duplicate rows
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    print(f"[*] Dedup: {before} rows -> {after} rows (dropped {before - after} duplicates)")

    return combined


# ============================================================================
# IMPUTATION TOURNAMENT
# ============================================================================

def run_imputation_tournament(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Compare Mean, Median, KNN, MICE, Zero imputation strategies.
    Pick the one with lowest RMSE on artificially masked data.
    Apply the winner to the full dataset.
    """
    print("\n[*] Running Imputation Tournament...")

    # Get rows with no nulls in numeric columns for validation
    df_valid = df[numeric_cols].dropna()

    if df_valid.shape[0] < 10:
        print("    [!] Not enough clean rows for tournament. Defaulting to MICE.")
        try:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_jobs=N_JOBS, max_depth=5, n_estimators=10),
                max_iter=5, random_state=42
            )
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        except Exception:
            print("    [!] MICE failed. Falling back to Mean.")
            imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df

    # Create validation: artificially mask 10% of known values
    np.random.seed(42)
    X_true = df_valid.values
    mask = np.random.rand(*X_true.shape) < 0.1
    X_masked = X_true.copy()
    X_masked[mask] = np.nan

    strategies = {
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'KNN_5': KNNImputer(n_neighbors=5),
        'MICE_RF': IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=N_JOBS, max_depth=5, n_estimators=10),
            max_iter=5, random_state=42
        ),
        'Zero': SimpleImputer(strategy='constant', fill_value=0),
    }

    best_rmse = float('inf')
    best_name = 'Mean'
    best_model = strategies['Mean']

    for name, model in strategies.items():
        try:
            start_t = time.time()
            X_imp = model.fit_transform(X_masked)
            rmse = np.sqrt(mean_squared_error(X_true[mask], X_imp[mask]))
            duration = time.time() - start_t
            marker = ""

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_model = model
                marker = " <-- best"

            print(f"    {name:12s}: RMSE={rmse:.4f}  ({duration:.1f}s){marker}")
        except Exception as e:
            print(f"    {name:12s}: FAILED ({e})")

    print(f"    WINNER: {best_name} (RMSE: {best_rmse:.4f})")

    # Apply winner to the full dataset
    try:
        # Re-fit a fresh clone on full data (not the masked version)
        fresh = clone(best_model)
        df[numeric_cols] = fresh.fit_transform(df[numeric_cols])
    except Exception as e2:
        print(f"    [!] Winner failed on full data ({e2}). Falling back to Mean.")
        fallback = SimpleImputer(strategy='mean')
        df[numeric_cols] = fallback.fit_transform(df[numeric_cols])

    return df


# ============================================================================
# PREPROCESSING
# ============================================================================

def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor


def get_feature_names_after_encoding(preprocessor) -> list:
    """Extract feature names after OneHotEncoding."""
    ohe_names = preprocessor.named_transformers_['cat']['encoder'] \
        .get_feature_names_out(CATEGORICAL_FEATURES)
    return list(NUMERIC_FEATURES) + list(ohe_names)


INTERACTION_PAIRS = [
    # (groupname OHE column prefix, clinical feature, output name)
    ('PostRgroupname', 'PreBLHBA1C', 'group_x_hba1c'),
    ('PostRgroupname', 'PreBLFBS',   'group_x_fbs'),
    ('PostRgroupname', 'PreBLPPBS',  'group_x_ppbs'),
]


def create_interaction_features(df: pd.DataFrame, feature_names: list) -> tuple:
    """
    Create interaction features between PostRgroupname and key clinical markers.
    This forces the model to learn how the care plan modulates outcomes.
    Returns: (augmented_df, updated_feature_names)
    """
    new_cols = []
    for group_prefix, clinical_feat, out_name in INTERACTION_PAIRS:
        # Find all OHE columns for PostRgroupname
        group_cols = [c for c in feature_names if c.startswith(f'cat__{group_prefix}') or c.startswith(group_prefix)]
        if not group_cols or clinical_feat not in feature_names:
            continue
        # Use the first group column (binary indicator) multiplied by clinical value
        for gc in group_cols:
            suffix = gc.split('_')[-1]  # e.g. '1' or '2'
            col_name = f"{out_name}_{suffix}"
            df[col_name] = df[gc].values * df[clinical_feat].values
            new_cols.append(col_name)

    updated_names = list(feature_names) + new_cols
    return df, updated_names


# ============================================================================
# MODEL TRAINING: STACKING ENSEMBLE
# ============================================================================

def train_stacking_ensemble(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a StackingRegressor with 9 base estimators + Ridge meta-learner.
    Also trains a standalone XGBRegressor for SHAP explanations.
    Returns: (stacking_model, xgb_shap_model)
    """
    print(f"\n[*] Training Stacking Ensemble ({N_JOBS} threads)...")
    print("    Base estimators: Ridge, Lasso, ElasticNet, BayesianRidge, SVR,")
    print("                     RF, GBR, KNN, XGBoost")
    print("    Meta-learner: Ridge")

    estimators = [
        ('ridge', Pipeline([('sc', StandardScaler()), ('m', Ridge())])),
        ('lasso', Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.01))])),
        ('elastic', Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=0.01, l1_ratio=0.5))])),
        ('bayesian', Pipeline([('sc', StandardScaler()), ('m', BayesianRidge())])),
        ('svr', Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf', C=1.0))])),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=N_JOBS, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
        ('knn', Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=7, n_jobs=N_JOBS))])),
        ('xgb', xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1,
            random_state=42, eval_metric='mae', n_jobs=N_JOBS,
        )),
    ]

    t0 = time.time()

    # Ensure writable numpy arrays (fixes joblib read-only array issue on Windows)
    X_np = np.ascontiguousarray(X_train.values if hasattr(X_train, 'values') else X_train)
    y_np = np.ascontiguousarray(y_train.values if hasattr(y_train, 'values') else y_train)

    # Train stacking ensemble
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=N_JOBS,
        passthrough=False,
    )
    stack.fit(X_np, y_np)
    stack_time = time.time() - t0
    print(f"    [OK] Stacking ensemble trained in {stack_time:.1f}s")

    # Also train a standalone XGBRegressor for SHAP (TreeExplainer needs a tree model)
    print("\n[*] Training standalone XGBRegressor for SHAP explanations...")
    t1 = time.time()
    xgb_shap = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1,
        random_state=42, eval_metric='mae', n_jobs=N_JOBS,
    )
    xgb_shap.fit(X_np, y_np)
    xgb_time = time.time() - t1
    print(f"    [OK] XGB SHAP model trained in {xgb_time:.1f}s")

    return stack, xgb_shap


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
    """Evaluate regression model performance."""
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_pred = model.predict(X_test_np)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n    [{model_name}] MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")
    print(f"    Target range: {y_test.min():.1f} - {y_test.max():.1f}")
    print(f"    Pred range:   {y_pred.min():.1f} - {y_pred.max():.1f}")
    print(f"    Mean target:  {y_test.mean():.2f}  |  Mean pred: {y_pred.mean():.2f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def print_feature_importances(xgb_model, feature_names, top_n=10):
    """Print top N feature importances from XGBRegressor."""
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n    Top {top_n} Feature Importances (XGB):")
    for _, row in fi.head(top_n).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(preprocessor, stack_model, xgb_shap_model, feature_names: list, output_path: str):
    """Save model artifacts for API deployment."""
    print(f"\n[*] Saving artifacts to '{output_path}'...")

    artifacts = {
        'preprocessor': preprocessor,
        'model': stack_model,       # StackingRegressor for predictions
        'shap_model': xgb_shap_model,  # XGBRegressor for SHAP explanations
        'feature_names': feature_names,
    }

    joblib.dump(artifacts, output_path)
    print(f"    [OK] Successfully saved!")
    print(f"\n    Artifact contents:")
    print(f"    - preprocessor: ColumnTransformer (numeric + categorical)")
    print(f"    - model: StackingRegressor (9 base estimators + Ridge meta)")
    print(f"    - shap_model: XGBRegressor (for SHAP TreeExplainer)")
    print(f"    - feature_names: {len(feature_names)} features")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 60)
    print("    DiabeSense+ Model Training Pipeline")
    print("    Target: PostBLHBA1C (HbA1c Regression)")
    print(f"    Threads: {N_JOBS} (CPU-based)")
    print("=" * 60)
    total_t0 = time.time()

    # Step 1: Load and merge data
    df = load_and_merge_data(DATA_FILES)

    # Step 2: Ensure all desired feature columns exist
    available_features = [c for c in DESIRED_FEATURES if c in df.columns]
    missing_features = [c for c in DESIRED_FEATURES if c not in df.columns]
    if missing_features:
        print(f"\n[!] Missing features (will be skipped): {missing_features}")

    print(f"\n[*] Using {len(available_features)} features")
    print(f"[*] Target distribution:")
    print(f"    Mean: {df[TARGET_COL].mean():.2f}")
    print(f"    Std:  {df[TARGET_COL].std():.2f}")
    print(f"    Min:  {df[TARGET_COL].min():.1f}, Max: {df[TARGET_COL].max():.1f}")

    # Step 3: Imputation Tournament on numeric columns
    numeric_in_data = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = run_imputation_tournament(df, numeric_in_data)

    # Step 4: Separate features and target
    X = df[available_features]
    y = df[TARGET_COL]

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n[*] Train-Test Split:")
    print(f"    - Training: {len(X_train)} samples")
    print(f"    - Testing:  {len(X_test)} samples")

    # Step 6: Create and fit preprocessor
    print("\n[*] Creating preprocessing pipeline...")
    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names_after_encoding(preprocessor)
    print(f"    Total features after encoding: {len(feature_names)}")

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # Step 7a: Create interaction features (PostRgroupname × clinical markers)
    print("\n[*] Creating PostRgroupname interaction features...")
    base_feature_names = list(feature_names)  # save before augmentation
    X_train_df, feature_names = create_interaction_features(X_train_df, base_feature_names)
    X_test_df, _ = create_interaction_features(X_test_df, base_feature_names)
    # Ensure test has same columns as train
    for col in feature_names:
        if col not in X_test_df.columns:
            X_test_df[col] = 0.0
    X_test_df = X_test_df[feature_names]  # reorder to match
    print(f"    Total features after interactions: {len(feature_names)}")

    # Step 8: Train Stacking Ensemble + standalone XGB for SHAP
    stack_model, xgb_shap_model = train_stacking_ensemble(X_train_df, y_train)

    # Step 8: Evaluate both models
    print("\n[*] Model Performance Evaluation")
    print("=" * 50)
    stack_metrics = evaluate_model(stack_model, X_test_df, y_test, "Stacking Ensemble")
    xgb_metrics = evaluate_model(xgb_shap_model, X_test_df, y_test, "XGB (SHAP)")

    # Feature importances from XGB
    print_feature_importances(xgb_shap_model, feature_names)

    # Step 9: SHAP test
    print("\n[*] Initializing SHAP TreeExplainer (on XGB model)...")
    explainer = shap.TreeExplainer(xgb_shap_model)
    sample_shap = explainer.shap_values(X_test_df.iloc[:1])
    print(f"    [OK] SHAP working. Sample output shape: {sample_shap.shape}")

    # Step 10: Save artifacts
    save_artifacts(preprocessor, stack_model, xgb_shap_model, feature_names, OUTPUT_PATH)

    total_time = time.time() - total_t0
    print("\n" + "=" * 60)
    print(f"    [OK] Training Pipeline Complete! ({total_time:.1f}s)")
    print("=" * 60)
    print(f"\n    Next steps:")
    print(f"    1. Run: uvicorn main:app --reload")
    print(f"    2. Open: http://127.0.0.1:8000/docs")
    print(f"    3. Open index.html in browser for the UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
