#!/usr/bin/env python3
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
import os
import re
import time
import warnings
import gc
import json
import matplotlib.pyplot as plt
import shap
from joblib import dump, Parallel, delayed

# SKLEARN IMPORTS
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    BayesianRidge, LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, QuantileRegressor
)
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================

INDIVIDUAL_DATASETS = [
    "/content/drive/MyDrive/Yoga_Apollo_Processed_Data/ApolloFormat_nmbfinalDiabetes (4).csv",
    "/content/drive/MyDrive/Yoga_Apollo_Processed_Data/ApolloFormat_nmbfinalnewDiabetes (3).csv",
    "/content/drive/MyDrive/Yoga_Apollo_Processed_Data/ApolloFormat_PrePostFinal (3).csv",
    "/content/drive/MyDrive/Yoga_Apollo_Processed_Data/apolloCombined.csv"
]

TARGET_COL = 'PostBLHBA1C'
OUTPUT_DIR = "/content/drive/MyDrive/Yoga_Apollo_Processed_Data/FINAL_MODELS"

MAX_RUNTIME_PER_TASK = 120
H2O_LIMIT = int(MAX_RUNTIME_PER_TASK * 0.7)
NFOLDS = 10

ALL_IMPUTATION_METRICS = []

# ============================================================================
# 🛠️ DATA LOADERS
# ============================================================================

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    new_cols = []
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def load_single_dataset(path, target):
    print(f"   [LOAD] Reading {os.path.basename(path)}...")
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[target])
    df = clean_column_names(df)

    clean_target = re.sub(r'[^A-Za-z0-9_]+', '_', target).strip('_')
    if clean_target not in df.columns:
        candidates = [c for c in df.columns if 'postblhba1c' in c.lower()]
        if candidates: clean_target = candidates[0]
        else: raise ValueError(f"Target {target} not found!")

    print(f"   [INFO] Shape: {df.shape}")
    return df, clean_target

def load_grand_master_dataset(file_list, target):
    print(f"   [LOAD] Merging ALL {len(file_list)} datasets...")
    frames = []
    clean_target_ref = re.sub(r'[^A-Za-z0-9_]+', '_', target).strip('_')

    for path in file_list:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                df = clean_column_names(df)
                candidates = [c for c in df.columns if clean_target_ref.lower() == c.lower()]
                if candidates:
                    df.rename(columns={candidates[0]: clean_target_ref}, inplace=True)
                    df['source_file'] = os.path.basename(path)
                    frames.append(df)
            except Exception as e:
                print(f"   ❌ Error reading {path}: {e}")

    if not frames: raise ValueError("No data found!")
    common_cols = set(frames[0].columns)
    for f in frames[1:]: common_cols.intersection_update(set(f.columns))

    final_cols = list(common_cols)
    combined = pd.concat([f[final_cols] for f in frames], ignore_index=True)
    combined = combined.dropna(subset=[clean_target_ref])
    print(f"   [INFO] Grand Master Shape: {combined.shape}")
    return combined, clean_target_ref

# ============================================================================
# 🧪 IMPUTATION (FIXED RECURSION ERROR)
# ============================================================================
class ImputationTournament:
    def __init__(self, target_col):
        self.target_col = target_col
        self.encoders = {}
        self.imputer = None
        self.best_method_name = "MICE"

    def _encode_categorical(self, df):
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=['object', 'category']).columns:
            if col == 'source_file': continue
            series = df_enc[col].astype(str)
            mask_null = (df_enc[col].isna()) | (series.str.lower().isin(['nan', 'none', '']))
            if mask_null.all(): continue
            le = LabelEncoder()
            valid_vals = series[~mask_null]
            le.fit(valid_vals)
            self.encoders[col] = le
            df_enc.loc[~mask_null, col] = le.transform(valid_vals)
            df_enc.loc[mask_null, col] = np.nan
            df_enc[col] = pd.to_numeric(df_enc[col])
        return df_enc

    def run_comparison(self, df, task_name):
        print(f"   [IMPUTATION] Running Tournament Comparison for {task_name}...")
        df_encoded = self._encode_categorical(df)
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols: numeric_cols.remove(self.target_col)

        # Validation set creation
        df_valid = df_encoded[numeric_cols].dropna()

        # --- FIX FOR RECURSION ERROR ---
        # If dataset is too sparse (no clean rows), skip tournament and force MICE directly
        if df_valid.shape[0] < 10:
             print("   ⚠️ Not enough clean data for comparison (Data too sparse). Defaulting directly to MICE.")
             self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_jobs=-1, max_depth=5, n_estimators=10),
                max_iter=5, random_state=42
             )
             try:
                 df_encoded[numeric_cols] = self.imputer.fit_transform(df_encoded[numeric_cols])
             except:
                 # Absolute fallback if MICE fails on sparse data
                 print("   ⚠️ MICE failed on sparse data. Falling back to Mean.")
                 self.imputer = SimpleImputer(strategy='mean')
                 df_encoded[numeric_cols] = self.imputer.fit_transform(df_encoded[numeric_cols])
             return df_encoded
        # --------------------------------

        np.random.seed(42)
        mask = np.random.rand(*df_valid.shape) < 0.1
        X_true = df_valid.values
        X_masked = X_true.copy()
        X_masked[mask] = np.nan

        strategies = {
            'Mean': SimpleImputer(strategy='mean'),
            'Median': SimpleImputer(strategy='median'),
            'KNN_5': KNNImputer(n_neighbors=5),
            'MICE_RF': IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1, max_depth=5, n_estimators=10), max_iter=5, random_state=42),
            'Zero': SimpleImputer(strategy='constant', fill_value=0)
        }

        best_rmse = float('inf')
        best_model = None

        for name, model in strategies.items():
            try:
                start_t = time.time()
                X_imp = model.fit_transform(X_masked)
                rmse = np.sqrt(mean_squared_error(X_true[mask], X_imp[mask]))
                duration = time.time() - start_t

                ALL_IMPUTATION_METRICS.append({
                    'Dataset': task_name,
                    'Imputation_Method': name,
                    'RMSE_Score': rmse,
                    'Time_Sec': duration
                })
                print(f"     🔹 {name}: RMSE={rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    self.best_method_name = name
            except Exception as e:
                print(f"     ❌ {name} failed: {e}")

        print(f"   🏆 Winner: {self.best_method_name} (RMSE: {best_rmse:.4f})")

        self.imputer = best_model
        try:
            full_imputed = self.imputer.fit_transform(df_encoded[numeric_cols])
            df_encoded[numeric_cols] = full_imputed
        except:
            fallback = SimpleImputer(strategy='mean')
            df_encoded[numeric_cols] = fallback.fit_transform(df_encoded[numeric_cols])

        return df_encoded

# ============================================================================
# ⚔️ SKLEARN STACKING
# ============================================================================
def train_sklearn_stack(X, y):
    print(f"\n   [SKLEARN] Training Ensemble on FULL DATA...")

    estimators = [
        ('ridge', Pipeline([('sc', StandardScaler()), ('m', Ridge())])),
        ('lasso', Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.01))])),
        ('elastic', Pipeline([('sc', StandardScaler()), ('m', ElasticNet())])),
        ('bayesian', Pipeline([('sc', StandardScaler()), ('m', BayesianRidge())])),
        ('pls', Pipeline([('sc', StandardScaler()), ('m', PLSRegression(n_components=5))])),
        ('svr', Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf', C=1.0))])),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3)),
        ('knn', Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=7))]))
    ]

    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5, n_jobs=-1)
    stack.fit(X, y)
    return stack

# ============================================================================
# 💧 H2O SUITE
# ============================================================================
def train_h2o_suite(df, target, task_name):
    print(f"   [H2O] Training AutoML + Deep Learning on FULL DATA...")
    hf = h2o.H2OFrame(df)
    x = hf.columns
    if target in x: x.remove(target)
    if 'source_file' in x: x.remove('source_file')

    results_dict = {}

    aml = H2OAutoML(max_runtime_secs=H2O_LIMIT, seed=42, project_name=f"AML_{task_name}",
                    stopping_metric="MAE", sort_metric="MAE", nfolds=NFOLDS)
    aml.train(x=x, y=target, training_frame=hf)
    print(f"   [H2O] AutoML Leaderboard:\n{aml.leaderboard.head(10)}")
    results_dict['H2O_AutoML_Leader'] = aml.leader

    print("   [H2O] Training Deep Learning...")
    dl = H2ODeepLearningEstimator(hidden=[200, 200], epochs=50, nfolds=NFOLDS, seed=42, stopping_metric="MAE")
    dl.train(x=x, y=target, training_frame=hf)
    results_dict['H2O_DeepLearning'] = dl

    best_mae = float('inf')
    best_model = None
    for name, model in results_dict.items():
        try:
            mae = model.mae(xval=True)
            if mae < best_mae:
                best_mae = mae
                best_model = model
        except: pass

    return best_model, results_dict

# ============================================================================
# 🔍 METRICS & EXPLAINABILITY
# ============================================================================
def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def generate_shap_plots(model, df_eval, target, task_dir):
    print(f"\n   [SHAP] Generating Summary Plots for {model.model_id}...")

    model_to_explain = model
    if "StackedEnsemble" in model.model_id:
        try:
            base_model_id = model.base_models[0]
            model_to_explain = h2o.get_model(base_model_id)
            print(f"      👉 Using base model for SHAP: {model_to_explain.model_id}")
        except:
            print("      ⚠️ Cannot explain StackedEnsemble directly. Skipping SHAP plots.")
            return

    try:
        hf_eval = h2o.H2OFrame(df_eval)
        contributions = model_to_explain.predict_contributions(hf_eval)
        contributions_df = contributions.as_data_frame()

        shap_values = contributions_df.iloc[:, :-1].values
        feature_names = contributions_df.columns[:-1]
        X_eval = df_eval[feature_names]

        # Plot 1: All Variables
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_eval, show=False)
        plt.title(f"SHAP Summary (All Features) - {model.model_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "shap_summary_all.png"))
        plt.close()

        # Plot 2: Top 15 Variables
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_eval, max_display=15, show=False)
        plt.title(f"SHAP Summary (Top 15) - {model.model_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "shap_summary_top15.png"))
        plt.close()

        print(f"      ✅ Saved SHAP plots to {task_dir}")

    except Exception as e:
        print(f"      ❌ SHAP Plot generation failed: {e}")

def save_artifacts(task_name, sk_model, h2o_model, imputer_obj, df_sample, metrics_list, feature_cols):
    task_dir = os.path.join(OUTPUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    print(f"\n   [SAVING] Artifacts -> {task_dir}")

    dump(sk_model, os.path.join(task_dir, "sklearn_stack.joblib"))
    dump(imputer_obj, os.path.join(task_dir, "imputer.joblib"))
    h2o.save_model(model=h2o_model, path=task_dir, force=True)

    pd.DataFrame(metrics_list).to_csv(os.path.join(task_dir, "model_performance_metrics.csv"), index=False)
    pd.DataFrame({'Features': feature_cols}).to_excel(os.path.join(task_dir, "model_feature_list.xlsx"), index=False)

    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump({"columns": list(df_sample.columns), "h2o_model": h2o_model.model_id}, f)

# ============================================================================
# 🚀 MAIN
# ============================================================================
def main():
    print("="*80 + "\n🏥 GRAND HYBRID ML PIPELINE (Fixed Recursion & Full Fit)\n" + "="*80)
    try:
        h2o.cluster().shutdown()
        time.sleep(3)
    except: pass
    h2o.init(nthreads=-1, max_mem_size='14G')

    tasks = []
    for i, path in enumerate(INDIVIDUAL_DATASETS):
        tasks.append({'type': 'single', 'path': path, 'name': f"DS_{i+1}_{os.path.basename(path)[:10]}"})
    tasks.append({'type': 'combo', 'files': INDIVIDUAL_DATASETS, 'name': "GRAND_MASTER_ALL"})

    for task in tasks:
        print(f"\n\n{'#'*60}\n🚀 STARTING: {task['name']}\n{'#'*60}")
        try:
            # 1. Load Data
            if task['type'] == 'single':
                df, clean_target = load_single_dataset(task['path'], TARGET_COL)
            else:
                df, clean_target = load_grand_master_dataset(task['files'], TARGET_COL)

            if len(df) < 20: continue

            # 2. Imputation
            imputer_wrapper = ImputationTournament(clean_target)
            df_full_clean = imputer_wrapper.run_comparison(df, task['name'])

            # 3. Full Data Prep
            X = df_full_clean.drop(columns=[clean_target, 'source_file'], errors='ignore')
            y = df_full_clean[clean_target]
            for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            df_full_h2o = pd.concat([X, y], axis=1)

            # 4. Train Models
            sk_model = train_sklearn_stack(X, y)
            best_h2o, h2o_models_dict = train_h2o_suite(df_full_h2o, clean_target, task['name'])

            # 5. Evaluate (Full Data)
            print("\n   [EVALUATION] Computing Metrics on Full Data...")
            task_metrics = []

            y_pred_sk = sk_model.predict(X)
            task_metrics.append(calculate_metrics(y, y_pred_sk, "Sklearn_Stacked"))

            hf_full = h2o.H2OFrame(df_full_h2o)
            for m_name, m_obj in h2o_models_dict.items():
                y_pred_h2o = m_obj.predict(hf_full).as_data_frame().values.flatten()
                met = calculate_metrics(y, y_pred_h2o, m_name)
                task_metrics.append(met)

            print(pd.DataFrame(task_metrics))

            # 6. SHAP
            task_dir_path = os.path.join(OUTPUT_DIR, task['name'])
            os.makedirs(task_dir_path, exist_ok=True)
            generate_shap_plots(best_h2o, df_full_h2o, clean_target, task_dir_path)

            # 7. Save
            save_artifacts(
                task['name'],
                sk_model,
                best_h2o,
                imputer_wrapper.imputer,
                df_full_clean,
                task_metrics,
                list(X.columns)
            )

        except Exception as e:
            print(f"❌ Critical Task Failure: {e}")
            import traceback
            traceback.print_exc()

    print("\n   [SAVING] Global Imputation Comparison Metrics...")
    pd.DataFrame(ALL_IMPUTATION_METRICS).to_csv(os.path.join(OUTPUT_DIR, "imputation_comparison_metrics.csv"), index=False)

    print("\n✅ MISSION COMPLETE.")

if __name__ == "__main__":
    main()