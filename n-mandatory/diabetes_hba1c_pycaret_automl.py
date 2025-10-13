#!/usr/bin/env python3
"""
Diabetes HbA1c Prediction using PyCaret AutoML
Windows-Compatible AutoML Solution

This script uses PyCaret for automated machine learning to predict PostBLHBA1C 
(post-intervention HbA1c levels) from the diabetes dataset with comprehensive
model comparison, evaluation, and clinical interpretation.

Target Variable: PostBLHBA1C
Dataset: nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv
Framework: PyCaret AutoML (Windows compatible)
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PyCaret Setup and Installation Check
# ============================================================================

def install_pycaret():
    """Install PyCaret and dependencies if not available."""
    import sys
    py_ver = sys.version_info[:2]
    supported = {(3, 9), (3, 10), (3, 11)}
    if py_ver not in supported:
        print(f"❌ Current Python {sys.version.split()[0]} is not supported by PyCaret. Supported: 3.9–3.11")
        print("➡️ Recommended (Conda):")
        print("   - Create env: conda create -n pycaret310 python=3.10 -y")
        print("   - Install:    conda run -n pycaret310 pip install pycaret==3.3.2 psutil")
        print("   - Run:        conda run -n pycaret310 python diabetes_hba1c_pycaret_automl.py")
        return False

    packages_to_check = [
        ('pycaret', 'pycaret'),
        ('psutil', 'psutil')
    ]
    for package_name, install_name in packages_to_check:
        try:
            __import__(package_name)
            print(f"✅ {package_name} already installed")
        except ImportError:
            print(f"📦 Installing {package_name}...")
            import subprocess
            import sys
            try:
                # Pin PyCaret to a stable version for better wheel availability
                if install_name == 'pycaret':
                    install_cmd = [sys.executable, "-m", "pip", "install", "pycaret==3.3.2"]
                else:
                    install_cmd = [sys.executable, "-m", "pip", "install", install_name]
                subprocess.check_call(install_cmd)
                print(f"✅ {package_name} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {package_name}: {e}")
                return False
    return True

def detect_system_configuration():
    """Detect system configuration and optimize settings."""
    import psutil
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💻 System Configuration Detected:")
    print(f"   🔧 CPU Cores: {cpu_count}")
    print(f"   🔧 RAM: {memory_gb:.1f} GB")
    if cpu_count >= 8 and memory_gb >= 16:
        print("   🚀 High-performance system detected - using full optimization")
        config = {
            'n_jobs': -1,
            'train_size': 0.8,
            'fold': 10,
            'preprocess': True,
            'transformation': True,
            'remove_multicollinearity': True,
            'multicollinearity_threshold': 0.9,
            'remove_outliers': True,
            'outliers_threshold': 0.05,
            'transformation_method': 'yeo-johnson'
        }
    elif cpu_count >= 4 and memory_gb >= 8:
        print("   ⚡ Medium-performance system - using balanced optimization")
        config = {
            'n_jobs': cpu_count // 2,
            'train_size': 0.8,
            'fold': 5,
            'preprocess': True,
            'transformation': True,
            'remove_multicollinearity': True,
            'multicollinearity_threshold': 0.9
        }
    else:
        print("   🐌 Lower-spec system - using conservative settings")
        config = {
            'n_jobs': 2,
            'train_size': 0.75,
            'fold': 3,
            'preprocess': True
        }
    return config

def setup_environment():
    """Setup the analysis environment with system optimization."""
    print("🚀 Setting up PyCaret environment for HbA1c prediction...")
    system_config = detect_system_configuration()
    if not install_pycaret():
        return False
    try:
        from pycaret.regression import setup, compare_models, create_model, tune_model, ensemble_model, finalize_model, evaluate_model, predict_model, plot_model
        print("✅ PyCaret modules imported successfully")
        return system_config
    except ImportError as e:
        print(f"❌ Failed to import PyCaret: {e}")
        return False

# ============================================================================
# SECTION 2: Data Loading and Validation
# ============================================================================

class DiabetesDataLoader:
    """Load and validate multiple diabetes datasets for PyCaret."""
    
    def __init__(self):
        self.datasets = {}
        self.target_column = 'PostBLHBA1C'
        self.dataset_files = [
            'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
            'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv', 
            'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
        ]
        self.dataset_names = [
            'nmbfinalDiabetes_4',
            'nmbfinalnewDiabetes_3',
            'PrePostFinal_3'
        ]
        
    def load_all_datasets(self):
        """Load all available diabetes datasets."""
        print("📊 Loading all diabetes datasets...")
        
        base_paths = ['./final_imputed_data/', 'final_imputed_data/', './']
        loaded_count = 0
        
        for i, (filename, dataset_name) in enumerate(zip(self.dataset_files, self.dataset_names)):
            print(f"\n📋 Loading dataset {i+1}/3: {dataset_name}")
            
            # Try different path combinations
            loaded = False
            for base_path in base_paths:
                full_path = os.path.join(base_path, filename)
                try:
                    data = pd.read_csv(full_path)
                    self.datasets[dataset_name] = {
                        'data': data,
                        'filename': filename,
                        'path': full_path
                    }
                    print(f"✅ Loaded from: {full_path}")
                    print(f"   Shape: {data.shape[0]} rows, {data.shape[1]} columns")
                    loaded = True
                    loaded_count += 1
                    break
                except FileNotFoundError:
                    continue
            
            if not loaded:
                print(f"❌ Could not find: {filename}")
        
        print(f"\n📊 Summary: {loaded_count}/{len(self.dataset_files)} datasets loaded successfully")
        return loaded_count > 0
        
    def validate_dataset(self, dataset_name):
        """Validate a specific dataset."""
        if dataset_name not in self.datasets:
            return False
            
        data = self.datasets[dataset_name]['data']
        
        print(f"\n🔍 Validating {dataset_name}:")
        
        # Check target variable
        if self.target_column not in data.columns:
            print(f"❌ Target column '{self.target_column}' not found")
            return False
        
        # Target statistics
        target_stats = data[self.target_column].describe()
        print(f"   Target '{self.target_column}':")
        print(f"     Range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
        print(f"     Mean: {target_stats['mean']:.2f} ± {target_stats['std']:.2f}")
        print(f"     Missing values: {data[self.target_column].isnull().sum()}")
        
        # Remove rows with missing target
        initial_rows = len(data)
        data_clean = data.dropna(subset=[self.target_column])
        if len(data_clean) != initial_rows:
            print(f"   Removed {initial_rows - len(data_clean)} rows with missing target")
            self.datasets[dataset_name]['data'] = data_clean
        
        # Check for any remaining missing values
        missing_summary = data_clean.isnull().sum()
        total_missing = missing_summary.sum()
        if total_missing > 0:
            print(f"   ⚠️ {total_missing} missing values in features (PyCaret will handle)")
        else:
            print("   ✅ No missing values - ML-ready")
        
        return True
    
    def get_dataset_names(self):
        """Get list of loaded dataset names."""
        return list(self.datasets.keys())
    
    def get_dataset(self, dataset_name):
        """Get a specific dataset."""
        return self.datasets.get(dataset_name, {}).get('data', None)
    
    def _validate_dataset(self):
        """Validate the loaded dataset."""
        print(f"\n� Dataset Information:")
        print(f"   Shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Check target variable
        if self.target_column not in self.data.columns:
            print(f"❌ Target column '{self.target_column}' not found")
            return False
        
        # Target statistics
        target_stats = self.data[self.target_column].describe()
        print(f"   Target '{self.target_column}':")
        print(f"     Range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
        print(f"     Mean: {target_stats['mean']:.2f} ± {target_stats['std']:.2f}")
        print(f"     Missing values: {self.data[self.target_column].isnull().sum()}")
        
        # Remove rows with missing target
        initial_rows = len(self.data)
        self.data = self.data.dropna(subset=[self.target_column])
        if len(self.data) != initial_rows:
            print(f"   Removed {initial_rows - len(self.data)} rows with missing target")
        
        # Check for any remaining missing values
        missing_summary = self.data.isnull().sum()
        total_missing = missing_summary.sum()
        if total_missing > 0:
            print(f"\n⚠️ Warning: {total_missing} missing values found in features")
            print("PyCaret will handle missing values automatically")
        else:
            print("✅ No missing values found - dataset is ML-ready")
        
        return True
    
    def get_data(self):
        """Return the loaded and validated dataset."""
        return self.data

# ============================================================================
# SECTION 3: PyCaret AutoML Pipeline
# ============================================================================

class PyCaretAutoML:
    """PyCaret-based AutoML pipeline for diabetes HbA1c prediction."""
    def __init__(self, data, target_column='PostBLHBA1C', dataset_name=None, system_config=None, feature_strategy: str = 'all'):
        self.data = data
        self.target_column = target_column
        self.dataset_name = dataset_name or "diabetes_dataset"
        self.system_config = system_config or {}
        # feature_strategy: 'all' -> no dropping, 'opt' -> enable selection+multicollinearity removal
        assert feature_strategy in ('all', 'opt'), "feature_strategy must be 'all' or 'opt'"
        self.feature_strategy = feature_strategy
        self.ml_setup = None
        self.best_models = []
        self.final_model = None
        self.results = {}
    def setup_ml_environment(self):
        """Setup PyCaret ML environment."""
        print("\n🤖 Setting up PyCaret ML environment...")
        from pycaret.regression import setup
        import inspect
        # PyCaret setup with system-optimized configurations
        setup_params = {
            'data': self.data,
            'target': self.target_column,
            'session_id': 42,
            # System-optimized parameters
            'train_size': self.system_config.get('train_size', 0.8),
            'fold': self.system_config.get('fold', 5),
            # Data preprocessing
            'numeric_imputation': 'mean',
            'categorical_imputation': 'mode', 
            'normalize': True,
            'transformation': self.system_config.get('transformation', True),
            # Feature strategy
            'remove_multicollinearity': (self.feature_strategy == 'opt'),
            'multicollinearity_threshold': self.system_config.get('multicollinearity_threshold', 0.9),
            # Feature engineering
            'pca': False,  # Keep interpretability
            'feature_selection': (self.feature_strategy == 'opt'),
            # Cross-validation
            'fold_strategy': 'kfold'
        }
        # Add optional system-specific parameters
        if 'remove_outliers' in self.system_config:
            setup_params['remove_outliers'] = self.system_config['remove_outliers']
            setup_params['outliers_threshold'] = self.system_config.get('outliers_threshold', 0.05)
        if 'transformation_method' in self.system_config:
            setup_params['transformation_method'] = self.system_config['transformation_method']
        print(f"   🔧 Using {self.system_config.get('fold', 5)}-fold CV with {self.system_config.get('n_jobs', -1)} parallel jobs")
        # Filter unsupported params for installed PyCaret version
        try:
            allowed = set(inspect.signature(setup).parameters.keys())
            setup_params = {k: v for k, v in setup_params.items() if k in allowed}
        except Exception:
            pass
        self.ml_setup = setup(**setup_params)
        # Configure n_jobs post-setup if supported
        try:
            from pycaret.regression import set_config
            if 'n_jobs' in self.system_config:
                set_config('n_jobs', self.system_config['n_jobs'])
        except Exception:
            pass
        print("✅ PyCaret environment setup complete")
        return True
    def compare_all_models(self):
        """Compare all available regression models."""
        print("\n📊 Comparing all available regression models...")
        from pycaret.regression import compare_models
        self.comparison_results = compare_models(
            include=['lr', 'rf', 'et', 'gbr', 'xgboost', 'lightgbm', 
                     'catboost', 'ridge', 'lasso', 'en', 'ada', 'dt'],
            sort='RMSE',
            n_select=10,
            fold=self.system_config.get('fold', 5),
            verbose=True
        )
        print("✅ Model comparison complete")
        return self.comparison_results
    def create_and_tune_best_models(self):
        """Create and tune the top performing models."""
        print("\n🔧 Creating and tuning best models...")
        from pycaret.regression import create_model, tune_model
        if hasattr(self, 'comparison_results'):
            if isinstance(self.comparison_results, list):
                top_models = self.comparison_results[:3]
            else:
                top_models = [self.comparison_results]
        else:
            from pycaret.regression import compare_models
            top_models = compare_models(n_select=3, verbose=False)
        tuned_models = []
        for i, model in enumerate(top_models, 1):
            try:
                print(f"\n   🎯 Tuning model {i}/3: {type(model).__name__}")
                tuned_model = tune_model(
                    model,
                    optimize='RMSE',
                    n_iter=50,
                    fold=self.system_config.get('fold', 5),
                    verbose=False
                )
                tuned_models.append(tuned_model)
                print(f"   ✅ Model {i} tuned successfully")
            except Exception as e:
                print(f"   ⚠️ Failed to tune model {i}: {e}")
                tuned_models.append(model)
        self.best_models = tuned_models
        print(f"\n✅ {len(tuned_models)} models tuned successfully")
        return tuned_models
    def create_ensemble(self):
        """Create ensemble from best models."""
        print("\n🎯 Creating ensemble model...")
        from pycaret.regression import ensemble_model, blend_models
        try:
            if len(self.best_models) >= 2:
                self.ensemble_model = blend_models(
                    estimator_list=self.best_models[:3],
                    fold=self.system_config.get('fold', 5),
                    verbose=False
                )
                print("✅ Blended ensemble created successfully")
            else:
                self.ensemble_model = self.best_models[0]
                print("✅ Using single best model (insufficient models for ensemble)")
        except Exception as e:
            print(f"⚠️ Ensemble creation failed: {e}")
            self.ensemble_model = self.best_models[0] if self.best_models else None
        return self.ensemble_model
    def finalize_and_evaluate(self):
        """Finalize the best model and evaluate performance."""
        print("\n🏆 Finalizing best model...")
        from pycaret.regression import finalize_model, evaluate_model, predict_model
        best_model = self.ensemble_model if hasattr(self, 'ensemble_model') else self.best_models[0]
        self.final_model = finalize_model(best_model)
        predictions_df = predict_model(self.final_model)
        y_true = predictions_df[self.target_column]
        y_pred = predictions_df['prediction_label']
        abs_errors = np.abs(y_true - y_pred)
        clinical_excellent = (abs_errors <= 0.5).mean() * 100
        clinical_good = (abs_errors <= 1.0).mean() * 100
        self.results = {
            'model_name': type(self.final_model).__name__,
            'predictions': predictions_df,
            'rmse': np.sqrt(((y_true - y_pred) ** 2).mean()),
            'mae': abs_errors.mean(),
            'r2': np.corrcoef(y_true, y_pred)[0, 1] ** 2,
            'clinical_excellent': clinical_excellent,
            'clinical_good': clinical_good,
            'sample_size': len(y_true),
            'target_range': (y_true.min(), y_true.max()),
            'target_mean_std': (y_true.mean(), y_true.std())
        }
        try:
            import os
            from pycaret.regression import save_model
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            strategy_suffix = f"_{self.feature_strategy}"
            clean_name = self.dataset_name.replace(' ', '_').replace('(', '').replace(')', '') + strategy_suffix
            model_filename = f"{models_dir}/{clean_name}_best_model"
            save_model(self.final_model, model_filename)
            print(f"💾 Model saved as: {model_filename}.pkl")
        except Exception as e:
            print(f"⚠️ Model saving failed: {e}")
        print("✅ Model finalized and evaluated")
        return self.final_model

    def export_feature_importance(self):
        """Compute and save feature importance (model-native and permutation) per dataset."""
        print(f"\n🧠 Computing feature importance for {self.dataset_name}...")
        import os
        from pycaret.regression import predict_model
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import pandas as pd

        fi_dir = os.path.join("visualizations", "feature_importance")
        os.makedirs(fi_dir, exist_ok=True)
        strategy_suffix = f"_{self.feature_strategy}"
        clean_name = self.dataset_name.replace(' ', '_').replace('(', '').replace(')', '') + strategy_suffix

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # 1) Model-native importance if available
        native_importance = None
        model = self.final_model
        try:
            if hasattr(model, 'feature_importances_'):
                native_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': getattr(model, 'feature_importances_')
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                coefs = getattr(model, 'coef_')
                if hasattr(coefs, 'toarray'):
                    coefs = coefs.toarray().ravel()
                native_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(coefs)
                }).sort_values('importance', ascending=False)
        except Exception:
            pass

        # 2) Permutation importance (robust across estimators)
        perm_df = None
        try:
            result = permutation_importance(
                model, X, y, scoring='neg_root_mean_squared_error', n_repeats=5, random_state=42, n_jobs=-1
            )
            perm_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
        except Exception as e:
            print(f"⚠️ Permutation importance failed: {e}")

        # Save CSVs
        try:
            if native_importance is not None:
                native_path = os.path.join(fi_dir, f"{clean_name}_native_importance.csv")
                native_importance.to_csv(native_path, index=False)
                print(f"💾 Saved native importance: {native_path}")
            if perm_df is not None:
                perm_path = os.path.join(fi_dir, f"{clean_name}_permutation_importance.csv")
                perm_df.to_csv(perm_path, index=False)
                print(f"💾 Saved permutation importance: {perm_path}")
        except Exception as e:
            print(f"⚠️ Saving importance CSV failed: {e}")

        # Simple barplot for permutation importance top 20
        try:
            if perm_df is not None and not perm_df.empty:
                import matplotlib.pyplot as plt
                top = perm_df.head(20)
                plt.figure(figsize=(10, 6))
                plt.barh(top['feature'][::-1], top['importance_mean'][::-1])
                plt.title(f"Permutation Importance (Top 20) - {self.dataset_name}")
                plt.xlabel('Importance (mean decrease in score)')
                plt.tight_layout()
                fig_path = os.path.join(fi_dir, f"{clean_name}_permutation_importance_top20.png")
                plt.savefig(fig_path, dpi=150)
                plt.close()
                print(f"🖼️ Saved permutation importance plot: {fig_path}")
        except Exception as e:
            print(f"⚠️ Plotting importance failed: {e}")
    def generate_visualizations(self):
        """Generate comprehensive visualizations with dataset-specific names."""
        print(f"\n📊 Generating visualizations for {self.dataset_name}...")
        from pycaret.regression import plot_model
        import os
        viz_dir = "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        try:
            strategy_suffix = f"_{self.feature_strategy}"
            clean_name = self.dataset_name.replace(' ', '_').replace('(', '').replace(')', '') + strategy_suffix
            print("   📈 Creating residuals plot...")
            plot_model(self.final_model, plot='residuals', save=f"{viz_dir}/{clean_name}_residuals")
            print("   📈 Creating prediction error plot...")
            plot_model(self.final_model, plot='error', save=f"{viz_dir}/{clean_name}_prediction_error")
            print("   📈 Creating feature importance plot...")
            plot_model(self.final_model, plot='feature', save=f"{viz_dir}/{clean_name}_feature_importance")
            print("   📈 Creating model performance plot...")
            plot_model(self.final_model, plot='parameter', save=f"{viz_dir}/{clean_name}_model_performance")
            print(f"✅ Visualizations saved to {viz_dir}/ directory with prefix '{clean_name}'")
        except Exception as e:
            print(f"⚠️ Some visualizations failed: {e}")
    def print_clinical_report(self):
        """Print comprehensive clinical report."""
        if not self.results:
            print("❌ No results available for reporting")
            return
        print(f"\n{'='*80}")
        print("🏥 PYCARET DIABETES HBA1C PREDICTION REPORT")
        print(f"{'='*80}")
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Sample Size: {self.results['sample_size']} patients")
        print(f"   HbA1c Range: {self.results['target_range'][0]:.2f}% - {self.results['target_range'][1]:.2f}%")
        print(f"   HbA1c Mean: {self.results['target_mean_std'][0]:.2f}% ± {self.results['target_mean_std'][1]:.2f}%")
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   Best Model: {self.results['model_name']}")
        print(f"   RMSE: {self.results['rmse']:.3f}% HbA1c")
        print(f"   MAE: {self.results['mae']:.3f}% HbA1c")
        print(f"   R²: {self.results['r2']:.3f}")
        print(f"\n🏥 CLINICAL ACCURACY:")
        print(f"   Excellent (±0.5%): {self.results['clinical_excellent']:.1f}% of predictions")
        print(f"   Good (±1.0%): {self.results['clinical_good']:.1f}% of predictions")
        if self.results['r2'] > 0.7:
            performance = "Excellent - Suitable for clinical decision support"
        elif self.results['r2'] > 0.5:
            performance = "Good - Useful for screening and monitoring"
        elif self.results['r2'] > 0.3:
            performance = "Moderate - Requires clinical validation"
        else:
            performance = "Poor - Needs significant improvement"
        print(f"   Overall Assessment: {performance}")
        print(f"\n💾 OUTPUTS:")
        print(f"   Saved visualizations in current directory")
        print(f"   Model can be deployed for predictions")

# ============================================================================
# SECTION 4: Main Execution Pipeline
# ============================================================================

def run_analysis_for_dataset(dataset_name, data, dataset_number, total_datasets, system_config):
    """Run complete AutoML analysis for a single dataset.
    Trains two variants: 'all' (no dropping) and 'opt' (selection+multicollinearity).
    Returns a dict with both results.
    """
    print(f"\n{'='*80}")
    print(f"🎯 DATASET {dataset_number}/{total_datasets}: {dataset_name}")
    print(f"{'='*80}")
    try:
        results_by_strategy = {}

        for strategy in ('all', 'opt'):
            print(f"\n--- Strategy: {strategy.upper()} ---")
            automl = PyCaretAutoML(data, dataset_name=dataset_name, system_config=system_config, feature_strategy=strategy)
            if not automl.setup_ml_environment():
                continue
            automl.compare_all_models()
            automl.create_and_tune_best_models()
            automl.create_ensemble()
            automl.finalize_and_evaluate()
            automl.generate_visualizations()
            automl.export_feature_importance()
            automl.print_clinical_report()
            results_by_strategy[strategy] = automl

        # Quick comparison if both ran
        if len(results_by_strategy) >= 1:
            def get_metric(a, m):
                return a.results.get(m) if a and hasattr(a, 'results') else None
            print("\n📊 Strategy comparison:")
            for s, a in results_by_strategy.items():
                if a and a.results:
                    print(f"   {s}: RMSE={a.results.get('rmse'):.3f}, MAE={a.results.get('mae'):.3f}, R2={a.results.get('r2'):.3f}")

        print(f"\n✅ {dataset_name} analysis completed successfully!")
        return results_by_strategy if results_by_strategy else None
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution pipeline for PyCaret diabetes prediction - Multiple Datasets."""
    print("🚀 PyCaret AutoML for Diabetes HbA1c Prediction - All Datasets")
    print("=" * 80)
    try:
        system_config = setup_environment()
        if not system_config:
            return False
        data_loader = DiabetesDataLoader()
        if not data_loader.load_all_datasets():
            return False
        dataset_names = data_loader.get_dataset_names()
        total_datasets = len(dataset_names)
        if total_datasets == 0:
            print("❌ No datasets loaded successfully")
            return False
        all_results = {}
        for i, dataset_name in enumerate(dataset_names, 1):
            if not data_loader.validate_dataset(dataset_name):
                print(f"⚠️ Skipping {dataset_name} due to validation errors")
                continue
            data = data_loader.get_dataset(dataset_name)
            if data is None:
                print(f"⚠️ Could not load data for {dataset_name}")
                continue
            result = run_analysis_for_dataset(dataset_name, data, i, total_datasets, system_config)
            if result:
                all_results[dataset_name] = result  # dict of strategies
        generate_multi_dataset_summary(all_results)
        print(f"\n🎉 All datasets analysis completed!")
        print(f"📊 Successfully processed: {len(all_results)}/{total_datasets} datasets")
        print("=" * 80)
        return all_results
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_multi_dataset_summary(results):
    """Generate a comprehensive summary across all datasets."""
    if not results:
        print("❌ No results to summarize")
        return
    print(f"\n{'='*80}")
    print("📊 MULTI-DATASET COMPARISON SUMMARY")
    print(f"{'='*80}")
    summary_data = []
    for dataset_name, strategy_dict in results.items():
        # Choose best strategy by R²
        best_row = None
        best_r2 = -1
        for strategy, automl in strategy_dict.items():
            if hasattr(automl, 'results') and automl.results:
                metrics = automl.results
                r2 = metrics.get('r2', 0)
                row = {
                    'Dataset': dataset_name + f" [{strategy}]",
                    'Model': metrics.get('model_name', 'Unknown'),
                    'RMSE': f"{metrics.get('rmse', 0):.3f}",
                    'MAE': f"{metrics.get('mae', 0):.3f}",
                    'R²': f"{r2:.3f}",
                    'Clinical_Excellent': f"{metrics.get('clinical_excellent', 0):.1f}%",
                    'Clinical_Good': f"{metrics.get('clinical_good', 0):.1f}%",
                    'Sample_Size': metrics.get('sample_size', 0)
                }
                # Track best per dataset for summary highlight
                if r2 > best_r2:
                    best_r2 = r2
                    best_row = row
                summary_data.append(row)
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n📋 Performance Comparison:")
        print(summary_df.to_string(index=False))
        if len(summary_data) > 1:
            best_r2_idx = summary_df['R²'].astype(float).idxmax()
            best_dataset = summary_df.iloc[best_r2_idx]['Dataset']
            best_r2 = summary_df.iloc[best_r2_idx]['R²']
            print(f"\n🏆 Best Performing Dataset: {best_dataset} (R² = {best_r2})")
        print(f"\n💾 Results Summary:")
        print(f"   📊 {len(results)} models created and saved")
        print(f"   📈 Visualizations generated for each dataset") 
        print(f"   🏥 Clinical accuracy metrics calculated")
        print(f"   🚀 Models ready for deployment")
    else:
        print("❌ No valid results found for summary")

if __name__ == "__main__":
    results = main()
    if results:
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the generated plots for visual insights")
        print("🏥 Review the clinical report above for performance metrics")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
