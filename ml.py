#!/usr/bin/env python3
"""
PyCaret AutoML Stack Ridge Model for HbAc1 Prediction
=====================================================
Target: PostBLHBA1C prediction with feature importance analysis
Datasets: All 3 diabetes datasets (nmbfinalDiabetes, nmbfinalnewDiabetes, PrePostFinal)
Approach: PyCaret AutoML with stack_models() for Ridge ensemble

Clinical Goal: Achieve MAE < 0.5 for accurate diabetes management
Reference: https://pycaret.gitbook.io/docs/get-started/functions/optimize#stack_models
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')

# PyCaret imports
from pycaret.regression import *
import pycaret.regression as pycaret_reg

# Optional advanced model libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# =============================================================================
# MANUAL ML CODE (COMMENTED OUT - REPLACED WITH PYCARET AUTOML)
# =============================================================================

"""
# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available")
"""

print("🚀 PyCaret AutoML Pipeline for HbA1c Prediction")
print("=" * 60)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Target: PostBLHBA1C prediction with MAE < 0.5")
print("📊 Datasets: 3 diabetes datasets with individual models")
print("🏗️ Architecture: AutoML with Stacked Ridge Ensemble")
print("=" * 60)

class DiabetesAutoMLModel:
    """Comprehensive AutoML Model for Diabetes HbA1c Prediction using PyCaret"""

    def __init__(self, dataset_name, random_state=42):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.final_model = None
        self.results = {}
        self.best_features = []
        self.pycaret_setup = None
        self.ensemble_scores = {}
        self.feature_importance = {}
        self.clinical_status = "UNKNOWN"
        self.target_stats = {}
        self.original_shape = (0, 0)

    def clean_column_names(self, df):
        """Clean column names to avoid PyCaret feature name mismatch issues"""
        # Replace spaces, hyphens, and special characters with underscores
        cleaned_columns = {}
        for col in df.columns:
            # Replace problematic characters
            cleaned_col = col.replace(' ', '_')
            cleaned_col = cleaned_col.replace('-', '_')
            cleaned_col = cleaned_col.replace('(', '_')
            cleaned_col = cleaned_col.replace(')', '_')
            cleaned_col = cleaned_col.replace('/', '_')
            cleaned_col = cleaned_col.replace('\\', '_')
            # Remove multiple consecutive underscores
            while '__' in cleaned_col:
                cleaned_col = cleaned_col.replace('__', '_')
            # Remove leading/trailing underscores
            cleaned_col = cleaned_col.strip('_')
            cleaned_columns[col] = cleaned_col
        
        print(f"🧹 Cleaned {len(cleaned_columns)} column names for PyCaret compatibility")
        return df.rename(columns=cleaned_columns)

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess diabetes dataset"""
        print(f"\n📂 Loading dataset: {self.dataset_name}")
        print(f"📄 File: {os.path.basename(file_path)}")

        df = pd.read_csv(file_path)
        
        # Clean column names first to avoid PyCaret issues
        df = self.clean_column_names(df)
        
        target_col = 'PostBLHBA1C'

        print(f"📊 Original shape: {df.shape}")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        initial_rows = len(df)
        df = df.dropna(subset=[target_col])

        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        z_scores = np.abs((df[target_col] - target_mean) / target_std)
        df = df[z_scores <= 4]

        outliers_removed = initial_rows - len(df)
        print(f"🧹 Cleaned shape: {df.shape} (removed {outliers_removed} outliers)")
        print(f"📈 HbA1c range: {df[target_col].min():.2f} - {df[target_col].max():.2f}")

        self.original_shape = df.shape
        self.target_stats = {
            'mean': df[target_col].mean(),
            'std': df[target_col].std(),
            'min': df[target_col].min(),
            'max': df[target_col].max()
        }
        return df

    def run_automl_pipeline(self, df):
        """Run the complete PyCaret AutoML pipeline"""
        print(f"\n🤖 Running PyCaret AutoML for {self.dataset_name}")
        target_col = 'PostBLHBA1C'

        # 1. Setup PyCaret environment with robust configuration
        print("🔧 Setting up PyCaret environment...")
        self.pycaret_setup = pycaret_reg.setup(
            data=df,
            target=target_col,
            session_id=self.random_state,
            train_size=0.8,
            
            # Preprocessing configuration
            normalize=True,
            normalize_method='robust',
            transformation=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,  # Less aggressive
            
            # Feature selection (more features for better performance)
            feature_selection=True,
            n_features_to_select=0.5,  # Keep 50% of features instead of 30%
            feature_selection_method='univariate',
            
            # Handle missing values
            imputation_type='simple',
            numeric_imputation='mean',
            categorical_imputation='mode',
            
            # Cross-validation
            fold_strategy='kfold',
            fold=5
        )

        # 2. Compare models to find the best ones
        print("🔍 Comparing base models...")
        
        # Enhanced model selection - removed Lasso (poor performer), added better models
        models_to_compare = [
            'ridge',     # Ridge Regression (good baseline)
            'rf',        # Random Forest (robust performer)
            'et',        # Extra Trees (your best performer!)
            'gbr',       # Gradient Boosting Regressor
            'ada',       # AdaBoost Regressor
            'dt',        # Decision Tree Regressor
            'en',        # Elastic Net (better than Lasso)
            'mlp',       # Multi Layer Perceptron (Neural Network)
            'knn',       # K Neighbors Regressor
            'br'         # Bayesian Ridge
        ]
        
        # Add advanced boosting models if available
        if XGBOOST_AVAILABLE:
            models_to_compare.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models_to_compare.append('lightgbm')
        if CATBOOST_AVAILABLE:
            models_to_compare.append('catboost')
        
        print(f"   📋 Enhanced models to compare: {models_to_compare}")
        print(f"   ❌ Removed: Lasso (poor MAE: 1.6194)")
        print(f"   ✅ Focus on: Tree-based and ensemble methods")
        
        try:
            best_models = compare_models(
                include=models_to_compare,
                sort='MAE',
                n_select=min(5, len(models_to_compare)),  # Select top 5 models for better ensemble
                fold=5
            )
            
            if not isinstance(best_models, list):
                best_models = [best_models]
            
            print(f"🏆 Selected {len(best_models)} top models for enhanced stacking")
            
        except Exception as e:
            print(f"⚠️ Model comparison failed, using best fallback models: {str(e)}")
            # Use the best performing models from your previous results
            fallback_models = ['et', 'rf', 'catboost'] if CATBOOST_AVAILABLE else ['et', 'rf', 'ridge']
            best_models = [create_model(model) for model in fallback_models]
            print(f"🔄 Using fallback models: {fallback_models}")

        # 3. Tune the best models
        print("🔧 Tuning top models...")
        tuned_models = [tune_model(model, optimize='MAE') for model in best_models]

        # 4. Create a Stacked Ensemble with Ridge as meta-model
        print("🏗️ Creating Stacked Ridge Ensemble...")
        stacked_model = stack_models(
            estimator_list=tuned_models,
            meta_model=create_model('ridge'),
            optimize='MAE'
        )

        # 5. Finalize the model (train on full dataset)
        print("✅ Finalizing the model...")
        self.final_model = finalize_model(stacked_model)

        # 6. Evaluate the final model
        print("\n📊 Evaluating final model on hold-out set...")
        
        try:
            # Use evaluate_model for more reliable evaluation
            evaluation_results = evaluate_model(stacked_model)
            
            # Get the last row of evaluation results which contains the mean scores
            if len(evaluation_results) > 0:
                mean_row = evaluation_results.iloc[-1]  # Last row is usually the mean
                mae = float(mean_row['MAE'])
                rmse = float(mean_row['RMSE']) 
                r2 = float(mean_row['R2'])
            else:
                # Fallback: try with predictions
                predictions = predict_model(stacked_model)
                y_test = predictions[target_col]
                y_pred = predictions['prediction_label']
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
            
            self.ensemble_scores = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            print(f"   📈 MAE: {mae:.4f}")
            print(f"   📊 RMSE: {rmse:.4f}")
            print(f"   🎯 R²: {r2:.4f}")
            
            # Clinical assessment
            if mae < 0.5:
                print("   🎉 SUCCESS: Clinical target MAE < 0.5 ACHIEVED!")
                clinical_status = "EXCELLENT"
            elif mae < 0.75:
                print("   ✅ GOOD: Clinically acceptable performance")
                clinical_status = "GOOD"
            elif mae < 1.0:
                print("   ⚠️ FAIR: Moderate performance, improvement needed")
                clinical_status = "FAIR"
            else:
                print(f"   📈 NEEDS IMPROVEMENT: {((mae / 0.5 - 1) * 100):.1f}% improvement needed")
                clinical_status = "POOR"
            
            self.clinical_status = clinical_status
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {str(e)}")
            # Set default values
            self.ensemble_scores = {'MAE': 999.0, 'RMSE': 999.0, 'R2': 0.0}
            self.clinical_status = "ERROR"

        return self.final_model

    def analyze_feature_importance(self):
        """Analyze and store feature importance from the best models"""
        print(f"\n🔍 Feature Importance Analysis for {self.dataset_name}")

        try:
            # Try to get feature importance from the final model
            if hasattr(self.final_model, 'feature_importances_'):
                importances = self.final_model.feature_importances_
                feature_names = get_config('X_train').columns
                
                fi_df = pd.DataFrame({
                    'Feature': feature_names, 
                    'Importance': importances
                })
                fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)
                
                print("📈 Top 15 features from the ensemble:")
                for idx, row in fi_df.iterrows():
                    print(f"    {idx+1:2d}. {row['Feature']}: {row['Importance']:.4f}")
                
                self.best_features = fi_df['Feature'].tolist()
                self.feature_importance = fi_df.set_index('Feature').to_dict()['Importance']
                
            else:
                # Alternative: Use permutation importance for any model
                print("   🔄 Using permutation importance analysis...")
                
                from sklearn.inspection import permutation_importance
                from sklearn.metrics import make_scorer, mean_absolute_error
                
                # Get test data from PyCaret
                X_test = get_config('X_test')
                y_test = get_config('y_test')
                
                # Calculate permutation importance
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                perm_importance = permutation_importance(
                    self.final_model, X_test, y_test,
                    scoring=scorer,
                    n_repeats=3,
                    random_state=self.random_state
                )
                
                fi_df = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': np.abs(perm_importance.importances_mean),  # Use absolute values
                    'Std': perm_importance.importances_std
                }).sort_values('Importance', ascending=False).head(15)
                
                print("� Top 15 features by permutation importance:")
                for idx, row in fi_df.iterrows():
                    print(f"    {idx+1:2d}. {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})")
                
                self.best_features = fi_df['Feature'].tolist()
                self.feature_importance = fi_df.set_index('Feature').to_dict()['Importance']
            
        except Exception as e:
            print(f"⚠️ Feature importance analysis failed: {str(e)}")
            # Fallback: Use available feature names
            try:
                feature_names = get_config('X_train').columns[:15].tolist()
                self.best_features = feature_names
                self.feature_importance = {name: 0.1 for name in feature_names}
                print(f"   🔄 Using fallback feature list: {len(self.best_features)} features")
            except:
                self.best_features = []
                self.feature_importance = {}

    def save_model_and_results(self, save_dir='models'):
        """Save the final PyCaret model and results"""
        os.makedirs(save_dir, exist_ok=True)

        model_filename = f"{save_dir}/pycaret_automl_{self.dataset_name.lower()}"
        results_filename = f"{save_dir}/pycaret_results_{self.dataset_name.lower()}.pkl"

        # Save the PyCaret pipeline
        save_model(self.final_model, model_filename)

        results = {
            'dataset_name': self.dataset_name,
            'ensemble_scores': self.ensemble_scores,
            'feature_importance': self.feature_importance,
            'best_features': self.best_features,
            'target_stats': self.target_stats,
            'original_shape': self.original_shape
        }

        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n💾 Model pipeline saved: {model_filename}.pkl")
        print(f"💾 Results saved: {results_filename}")

        return f"{model_filename}.pkl", results_filename

    def generate_summary_report(self):
        """Generate a summary report for the AutoML run"""
        print(f"\n" + "="*80)
        print(f"📋 PYCARET AUTOML SUMMARY - {self.dataset_name.upper()}")
        print(f"="*80)

        print(f"📊 Dataset Information:")
        print(f"    • Original shape: {self.original_shape}")
        print(f"    • Target range: {self.target_stats.get('min', 'N/A'):.2f} - {self.target_stats.get('max', 'N/A'):.2f}")
        print(f"    • Target mean: {self.target_stats.get('mean', 'N/A'):.2f} (±{self.target_stats.get('std', 'N/A'):.2f})")

        print(f"\n🏆 Final Model Performance:")
        print(f"    • Ensemble MAE: {self.ensemble_scores.get('MAE', 'N/A'):.4f}")
        print(f"    • Ensemble RMSE: {self.ensemble_scores.get('RMSE', 'N/A'):.4f}")
        print(f"    • Ensemble R²: {self.ensemble_scores.get('R2', 'N/A'):.4f}")

        print(f"\n🎯 Clinical Assessment:")
        mae = self.ensemble_scores.get('MAE', 999.0)
        
        if mae < 0.5:
            status_emoji = "🎉"
            status_text = "EXCELLENT - Ready for clinical use"
            grade = "A"
        elif mae < 0.75:
            status_emoji = "✅"
            status_text = "GOOD - Clinically acceptable"
            grade = "B"
        elif mae < 1.0:
            status_emoji = "⚠️"
            status_text = "FAIR - Needs improvement"
            grade = "C"
        else:
            status_emoji = "❌"
            status_text = "POOR - Significant improvement needed"
            grade = "D"
        
        print(f"    • Status: {status_emoji} {status_text}")
        print(f"    • Clinical Grade: {grade}")
        print(f"    • Target: MAE < 0.5 for clinical accuracy")

        if self.best_features:
            print(f"\n🔍 Top 10 Most Important Features:")
            for i, feature in enumerate(self.best_features[:10]):
                importance = self.feature_importance.get(feature, 0.0)
                print(f"    {i+1:2d}. {feature}: {importance:.4f}")
        else:
            print(f"\n⚠️ Feature importance analysis not available")

        print(f"\n💡 Recommendations:")
        if mae < 0.5:
            print("    • Model is ready for clinical validation")
            print("    • Consider prospective validation study")
            print("    • Prepare for deployment pipeline")
        elif mae < 1.0:
            print("    • Further feature engineering recommended")
            print("    • Consider ensemble with additional models")
            print("    • Validate on external dataset")
        else:
            print("    • Collect more high-quality training data")
            print("    • Review feature selection strategy")
            print("    • Consider domain expert consultation")
            
        print(f"="*80)

def run_pipeline():
    """Run the AutoML pipeline for all configured datasets"""
    datasets = [
        {
            'name': 'nmbfinalDiabetes_4',
            'file': '/content/nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Primary diabetes dataset'
        },
        {
            'name': 'nmbfinalnewDiabetes_3',
            'file': '/content/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Secondary diabetes dataset'
        },
        {
            'name': 'PrePostFinal_3',
            'file': '/content/PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Pre/Post intervention dataset'
        }
    ]

    all_results = {}

    for i, config in enumerate(datasets, 1):
        print(f"\n🔥 PROCESSING DATASET {i}/{len(datasets)}: {config['name']}")
        print(f"📝 {config['description']}")
        print("-" * 60)

        try:
            if not os.path.exists(config['file']):
                print(f"❌ File not found: {config['file']}")
                continue

            model_runner = DiabetesAutoMLModel(dataset_name=config['name'])
            df = model_runner.load_and_preprocess_data(config['file'])

            # For PyCaret, we don't need manual feature engineering as it's part of the setup
            # df_enhanced = model_runner.advanced_feature_engineering(df)

            final_model = model_runner.run_automl_pipeline(df)
            model_runner.analyze_feature_importance()
            model_file, results_file = model_runner.save_model_and_results()
            model_runner.generate_summary_report()

            all_results[config['name']] = {
                'model': model_runner,
                'scores': model_runner.ensemble_scores,
                'model_file': model_file
            }

        except Exception as e:
            print(f"❌ Error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*100)
    print("🏆 FINAL SUMMARY - ALL DATASETS")
    print("="*100)

    if all_results:
        best_dataset = min(all_results.items(), key=lambda x: x[1]['scores']['MAE'])
        print(f"🥇 Best Performing Dataset: {best_dataset[0]} (MAE: {best_dataset[1]['scores']['MAE']:.4f})")
    else:
        print("No datasets processed successfully.")

    print(f"🎉 AutoML Pipeline Complete! Models saved in 'models/' directory.")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance improvement recommendations
    print(f"\n💡 PERFORMANCE IMPROVEMENT RECOMMENDATIONS:")
    print(f"="*60)
    print(f"🔍 Data Quality:")
    print(f"   • Increase dataset size (current: 885, 546, 5531 samples)")
    print(f"   • Feature engineering: create domain-specific interactions")
    print(f"   • Remove low-variance and redundant features")
    print(f"\n🤖 Model Optimization:")
    print(f"   • Try ensemble of top 5 models instead of 3")
    print(f"   • Use advanced meta-learners (Neural Networks, XGBoost)")
    print(f"   • Hyperparameter tuning with more iterations")
    print(f"\n🎯 Clinical Focus:")
    print(f"   • Focus on PreBLHBA1C (strongest predictor: 1.19 importance)")
    print(f"   • Include interaction terms with diabetes duration")
    print(f"   • Consider time-series modeling for longitudinal data")
    print(f"="*60)

if __name__ == "__main__":
    run_pipeline()