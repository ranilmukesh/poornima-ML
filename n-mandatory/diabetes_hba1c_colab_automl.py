#!/usr/bin/env python3
"""
Diabetes HbA1c Prediction using PyCaret AutoML - Google Colab Optimized
Colab-Compatible AutoML Solution

This script is optimized for Google Colab and uses PyCaret for automated 
machine learning to predict PostBLHBA1C (post-intervention HbA1c levels) 
from multiple diabetes datasets with comprehensive model comparison, 
evaluation, and clinical interpretation.

Target Variable: PostBLHBA1C
Datasets: All three diabetes datasets from final_imputed_data folder
Framework: PyCaret AutoML (Colab optimized)

Instructions for Google Colab:
1. Upload your datasets to Colab files or mount Google Drive
2. Update the file paths in the COLAB_CONFIG section
3. Run all cells sequentially
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLAB CONFIGURATION - Update these paths for your setup
# ============================================================================

COLAB_CONFIG = {
    # Set to True when running in Google Colab
    'IS_COLAB': True,
    
    # Dataset paths - Update these based on your Colab setup
    'DATASET_PATHS': {
        # Option 1: If datasets are uploaded directly to Colab files
        'direct_upload': [
            'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
            'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv', 
            'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
        ],
        
        # Option 2: If datasets are in Google Drive (uncomment and modify)
        # 'google_drive': [
        #     '/content/drive/MyDrive/diabetes_data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
        #     '/content/drive/MyDrive/diabetes_data/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
        #     '/content/drive/MyDrive/diabetes_data/PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
        # ]
    },
    
    # Colab-optimized settings
    'USE_GPU': True,  # Set to False if you want to use CPU only
    'MOUNT_DRIVE': False,  # Set to True if using Google Drive
}

# ============================================================================
# SECTION 1: Colab Environment Setup
# ============================================================================

def check_colab_environment():
    """Check if running in Google Colab and setup accordingly."""
    try:
        import google.colab
        print("✅ Running in Google Colab")
        return True
    except ImportError:
        print("ℹ️ Not running in Google Colab - using local environment")
        return False

def setup_colab_environment():
    """Setup Google Colab environment with optimizations."""
    is_colab = check_colab_environment()
    
    if is_colab or COLAB_CONFIG['IS_COLAB']:
        print("🔧 Setting up Colab environment...")
        
        # Mount Google Drive if needed
        if COLAB_CONFIG['MOUNT_DRIVE']:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                print("✅ Google Drive mounted successfully")
            except Exception as e:
                print(f"⚠️ Failed to mount Google Drive: {e}")
        
        # Check Python version compatibility
        python_version = sys.version_info
        print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major == 3 and python_version.minor in [9, 10, 11]:
            print("✅ Python version compatible with PyCaret")
            return True
        else:
            print("⚠️ Python version might have compatibility issues with PyCaret")
            print("   Colab typically uses Python 3.10 which should work")
            return True  # Continue anyway as Colab usually works
    
    return True

def install_requirements():
    """Install required packages for Colab."""
    print("📦 Installing required packages...")
    
    packages = [
        'pycaret[full]',  # Full PyCaret installation
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'ipywidgets'  # For interactive widgets
    ]
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            os.system(f"pip install {package} --quiet")
        except Exception as e:
            print(f"   ⚠️ Warning installing {package}: {e}")
    
    print("✅ Package installation completed")

def detect_colab_resources():
    """Detect available Colab resources and optimize accordingly."""
    import psutil
    
    # Get system specs
    cpu_count = os.cpu_count()
    memory_info = psutil.virtual_memory()
    memory_gb = memory_info.total / (1024**3)
    
    print(f"💻 Colab Resources Detected:")
    print(f"   🔧 CPU Cores: {cpu_count}")
    print(f"   🔧 RAM: {memory_gb:.1f} GB")
    
    # Check for GPU
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   🚀 GPU: {gpu_name}")
            gpu_available = True
    except ImportError:
        pass
    
    if not gpu_available:
        print("   💻 GPU: Not available (CPU mode)")
    
    # Optimize configuration based on resources
    if memory_gb >= 12:  # High-RAM Colab
        config = {
            'n_jobs': -1,
            'train_size': 0.8,
            'fold': 10,
            'preprocess': True,
            'transformation': True,
            'remove_multicollinearity': True,
            'multicollinearity_threshold': 0.9,
            'remove_outliers': True,
            'outliers_threshold': 0.05
        }
        print("   🚀 High-RAM Colab detected - full optimization")
    else:  # Standard Colab
        config = {
            'n_jobs': cpu_count,
            'train_size': 0.8,
            'fold': 5,
            'preprocess': True,
            'transformation': True,
            'remove_multicollinearity': True,
            'multicollinearity_threshold': 0.9
        }
        print("   ⚡ Standard Colab - balanced optimization")
    
    return config

# ============================================================================
# SECTION 2: Data Loading for Colab
# ============================================================================

class ColabDiabetesDataLoader:
    """Load and validate diabetes datasets in Google Colab."""
    
    def __init__(self):
        self.datasets = {}
        self.target_column = 'PostBLHBA1C'
        self.dataset_names = [
            'nmbfinalDiabetes_4',
            'nmbfinalnewDiabetes_3',
            'PrePostFinal_3'
        ]
        
    def load_all_datasets(self):
        """Load all available diabetes datasets in Colab."""
        print("📊 Loading diabetes datasets in Colab...")
        
        # Get file paths based on configuration
        file_paths = self._get_dataset_paths()
        
        loaded_count = 0
        
        for i, (file_path, dataset_name) in enumerate(zip(file_paths, self.dataset_names)):
            print(f"\n📋 Loading dataset {i+1}/3: {dataset_name}")
            
            try:
                # Load dataset
                data = pd.read_csv(file_path)
                self.datasets[dataset_name] = {
                    'data': data,
                    'path': file_path
                }
                print(f"✅ Loaded from: {file_path}")
                print(f"   Shape: {data.shape[0]} rows, {data.shape[1]} columns")
                loaded_count += 1
                
            except FileNotFoundError:
                print(f"❌ File not found: {file_path}")
                print("   💡 Make sure to upload the file to Colab or update the path")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        print(f"\n📊 Summary: {loaded_count}/{len(self.dataset_names)} datasets loaded successfully")
        
        if loaded_count == 0:
            print("\n🆘 NO DATASETS LOADED!")
            print("Please ensure your datasets are uploaded to Colab:")
            print("1. Click the folder icon in the left sidebar")
            print("2. Upload your CSV files")
            print("3. Update the file paths in COLAB_CONFIG if needed")
        
        return loaded_count > 0
    
    def _get_dataset_paths(self):
        """Get dataset paths based on Colab configuration."""
        paths = []
        
        # Use direct upload paths by default
        base_files = COLAB_CONFIG['DATASET_PATHS']['direct_upload']
        
        # Check if Google Drive paths are configured
        if (COLAB_CONFIG['MOUNT_DRIVE'] and 
            'google_drive' in COLAB_CONFIG['DATASET_PATHS']):
            return COLAB_CONFIG['DATASET_PATHS']['google_drive']
        
        # Look for files in common Colab locations
        for filename in base_files:
            # Try different possible locations
            possible_paths = [
                filename,  # Current directory
                f'/content/{filename}',  # Colab content directory
                f'./content/{filename}',  # Relative content directory
            ]
            
            file_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    paths.append(path)
                    file_found = True
                    break
            
            if not file_found:
                paths.append(filename)  # Default - will cause FileNotFound
        
        return paths
    
    def validate_dataset(self, dataset_name):
        """Validate a specific dataset."""
        if dataset_name not in self.datasets:
            return False
            
        data = self.datasets[dataset_name]['data']
        
        print(f"\n🔍 Validating {dataset_name}:")
        
        # Check target variable
        if self.target_column not in data.columns:
            print(f"❌ Target column '{self.target_column}' not found")
            print(f"   Available columns: {list(data.columns)}")
            return False
        
        # Target statistics
        target_stats = data[self.target_column].describe()
        print(f"   Target '{self.target_column}':")
        print(f"     Range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
        print(f"     Mean: {target_stats['mean']:.2f} ± {target_stats['std']:.2f}")
        print(f"     Missing values: {data[self.target_column].isnull().sum()}")
        
        # Clean missing target values
        initial_rows = len(data)
        data_clean = data.dropna(subset=[self.target_column])
        if len(data_clean) != initial_rows:
            print(f"   Removed {initial_rows - len(data_clean)} rows with missing target")
            self.datasets[dataset_name]['data'] = data_clean
        
        # Check for missing values in features
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

# ============================================================================
# SECTION 3: PyCaret AutoML Pipeline for Colab
# ============================================================================

class ColabPyCaretAutoML:
    """Colab-optimized PyCaret AutoML pipeline for diabetes HbA1c prediction."""
    
    def __init__(self, data, target_column='PostBLHBA1C', dataset_name=None, colab_config=None):
        self.data = data
        self.target_column = target_column
        self.dataset_name = dataset_name or "diabetes_dataset"
        self.colab_config = colab_config or {}
        self.ml_setup = None
        self.best_models = []
        self.final_model = None
        self.results = {}
        
    def setup_ml_environment(self):
        """Setup PyCaret ML environment optimized for Colab."""
        print(f"\n🤖 Setting up PyCaret ML environment for {self.dataset_name}...")
        
        try:
            from pycaret.regression import setup
        except ImportError as e:
            print(f"❌ Failed to import PyCaret: {e}")
            print("💡 Try restarting the runtime and running the installation cell again")
            return False
        
        # Colab-optimized setup parameters
        setup_params = {
            'data': self.data,
            'target': self.target_column,
            'session_id': 42,  # Reproducibility
            
            # Colab-optimized parameters
            'train_size': self.colab_config.get('train_size', 0.8),
            'fold': self.colab_config.get('fold', 5),
            'n_jobs': self.colab_config.get('n_jobs', -1),
            
            # Data preprocessing
            'numeric_imputation': 'mean',
            'categorical_imputation': 'mode', 
            'normalize': True,
            'transformation': self.colab_config.get('transformation', True),
            'remove_multicollinearity': self.colab_config.get('remove_multicollinearity', True),
            'multicollinearity_threshold': self.colab_config.get('multicollinearity_threshold', 0.9),
            
            # Feature engineering
            'pca': False,  # Keep interpretability
            'feature_selection': True,
            'feature_selection_threshold': 0.8,
            
            # Cross-validation
            'fold_strategy': 'kfold',
            
            # Colab display settings
            'html': True,  # Enable HTML display for Colab
            'silent': False  # Show progress
        }
        
        # Add optional parameters based on system resources
        if self.colab_config.get('remove_outliers', False):
            setup_params['remove_outliers'] = True
            setup_params['outliers_threshold'] = self.colab_config.get('outliers_threshold', 0.05)
        
        print(f"   🔧 Using {self.colab_config.get('fold', 5)}-fold CV with {self.colab_config.get('n_jobs', -1)} jobs")
        
        try:
            self.ml_setup = setup(**setup_params)
            print("✅ PyCaret environment setup complete")
            return True
        except Exception as e:
            print(f"❌ PyCaret setup failed: {e}")
            return False
    
    def compare_all_models(self):
        """Compare all available regression models with Colab-friendly display."""
        print("\n📊 Comparing regression models (this may take a few minutes)...")
        
        from pycaret.regression import compare_models
        
        try:
            # Compare models - Colab optimized
            self.comparison_results = compare_models(
                include=['lr', 'rf', 'et', 'gbr', 'xgboost', 'lightgbm', 
                        'ridge', 'lasso', 'en', 'ada', 'dt'],  # Removed catboost for compatibility
                sort='RMSE',
                n_select=5,  # Reduced for Colab performance
                fold=self.colab_config.get('fold', 5),
                verbose=True,
                display_format='html'  # Better for Colab display
            )
            
            print("✅ Model comparison complete")
            return self.comparison_results
            
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            # Fallback to basic models
            try:
                self.comparison_results = compare_models(
                    include=['lr', 'rf', 'ridge'],
                    n_select=3,
                    verbose=False
                )
                print("✅ Fallback model comparison complete")
                return self.comparison_results
            except:
                return None
    
    def create_and_tune_best_models(self):
        """Create and tune top models with Colab optimization."""
        print("\n🔧 Tuning best models for Colab...")
        
        from pycaret.regression import tune_model
        
        if not hasattr(self, 'comparison_results') or self.comparison_results is None:
            print("❌ No comparison results available")
            return []
        
        # Get top models
        if isinstance(self.comparison_results, list):
            top_models = self.comparison_results[:3]  # Top 3 models
        else:
            top_models = [self.comparison_results]
        
        tuned_models = []
        
        for i, model in enumerate(top_models, 1):
            try:
                print(f"   🎯 Tuning model {i}/{len(top_models)}: {type(model).__name__}")
                
                # Colab-friendly tuning
                tuned_model = tune_model(
                    model,
                    optimize='RMSE',
                    n_iter=20,  # Reduced iterations for Colab
                    fold=3,     # Reduced folds for faster execution
                    verbose=False
                )
                
                tuned_models.append(tuned_model)
                print(f"   ✅ Model {i} tuned successfully")
                
            except Exception as e:
                print(f"   ⚠️ Tuning failed for model {i}: {e}")
                tuned_models.append(model)  # Use original model
        
        self.best_models = tuned_models
        print(f"\n✅ {len(tuned_models)} models ready")
        return tuned_models
    
    def create_ensemble(self):
        """Create ensemble optimized for Colab."""
        print("\n🎯 Creating ensemble model...")
        
        try:
            from pycaret.regression import blend_models
            
            if len(self.best_models) >= 2:
                # Create blend with top 2-3 models (Colab friendly)
                models_to_blend = self.best_models[:min(3, len(self.best_models))]
                
                self.ensemble_model = blend_models(
                    estimator_list=models_to_blend,
                    fold=3,  # Reduced for Colab
                    verbose=False
                )
                print("✅ Ensemble created successfully")
            else:
                self.ensemble_model = self.best_models[0] if self.best_models else None
                print("✅ Using single best model")
                
        except Exception as e:
            print(f"⚠️ Ensemble creation failed: {e}")
            self.ensemble_model = self.best_models[0] if self.best_models else None
            
        return self.ensemble_model
    
    def finalize_and_evaluate(self):
        """Finalize model and evaluate with Colab-friendly output."""
        print("\n🏆 Finalizing best model...")
        
        from pycaret.regression import finalize_model, predict_model
        
        # Get best model
        best_model = getattr(self, 'ensemble_model', None) or (self.best_models[0] if self.best_models else None)
        
        if best_model is None:
            print("❌ No model available for finalization")
            return None
        
        try:
            # Finalize model
            self.final_model = finalize_model(best_model)
            
            # Generate predictions
            predictions_df = predict_model(self.final_model)
            
            # Calculate metrics
            y_true = predictions_df[self.target_column]
            y_pred = predictions_df['prediction_label']
            
            # Clinical accuracy
            abs_errors = np.abs(y_true - y_pred)
            clinical_excellent = (abs_errors <= 0.5).mean() * 100
            clinical_good = (abs_errors <= 1.0).mean() * 100
            
            # Store results
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
            
            print("✅ Model finalized and evaluated")
            return self.final_model
            
        except Exception as e:
            print(f"❌ Model finalization failed: {e}")
            return None
    
    def generate_visualizations(self):
        """Generate Colab-friendly visualizations."""
        print(f"\n📊 Generating visualizations for {self.dataset_name}...")
        
        if not hasattr(self, 'final_model') or self.final_model is None:
            print("❌ No model available for visualization")
            return
        
        from pycaret.regression import plot_model
        import matplotlib.pyplot as plt
        
        try:
            # Generate key plots
            plots_to_create = [
                ('residuals', 'Residuals Plot'),
                ('error', 'Prediction Error Plot'), 
                ('feature', 'Feature Importance')
            ]
            
            for plot_type, plot_name in plots_to_create:
                try:
                    print(f"   📈 Creating {plot_name}...")
                    plot_model(self.final_model, plot=plot_type, display_format='streamlit')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ {plot_name} failed: {e}")
            
            print("✅ Visualizations generated")
            
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
    
    def print_clinical_report(self):
        """Print comprehensive clinical report optimized for Colab."""
        if not self.results:
            print("❌ No results available for reporting")
            return
        
        print(f"\n{'='*70}")
        print(f"🏥 CLINICAL REPORT - {self.dataset_name}")
        print(f"{'='*70}")
        
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
        
        # Performance interpretation
        r2 = self.results['r2']
        if r2 > 0.7:
            performance = "🟢 Excellent - Clinical decision support ready"
        elif r2 > 0.5:
            performance = "🟡 Good - Useful for monitoring"
        elif r2 > 0.3:
            performance = "🟠 Moderate - Needs validation"
        else:
            performance = "🔴 Poor - Requires improvement"
        
        print(f"   Assessment: {performance}")

# ============================================================================
# SECTION 4: Colab Main Execution Pipeline
# ============================================================================

def run_colab_analysis_for_dataset(dataset_name, data, dataset_number, total_datasets, colab_config):
    """Run analysis for a single dataset in Colab."""
    
    print(f"\n{'='*70}")
    print(f"🎯 DATASET {dataset_number}/{total_datasets}: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        # Initialize AutoML
        automl = ColabPyCaretAutoML(data, dataset_name=dataset_name, colab_config=colab_config)
        
        # Setup ML environment
        if not automl.setup_ml_environment():
            return None
        
        # Compare models
        automl.compare_all_models()
        
        # Tune models
        automl.create_and_tune_best_models()
        
        # Create ensemble
        automl.create_ensemble()
        
        # Finalize and evaluate
        automl.finalize_and_evaluate()
        
        # Generate visualizations
        automl.generate_visualizations()
        
        # Print report
        automl.print_clinical_report()
        
        print(f"\n✅ {dataset_name} analysis completed!")
        return automl
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main_colab():
    """Main execution pipeline optimized for Google Colab."""
    
    print("🚀 PyCaret AutoML for Diabetes HbA1c - Google Colab Edition")
    print("=" * 70)
    
    try:
        # 1. Setup Colab environment
        if not setup_colab_environment():
            print("❌ Colab environment setup failed")
            return False
        
        # 2. Install requirements
        install_requirements()
        
        # 3. Detect resources and get configuration
        colab_config = detect_colab_resources()
        
        # 4. Load datasets
        data_loader = ColabDiabetesDataLoader()
        if not data_loader.load_all_datasets():
            print("❌ No datasets loaded - please check file paths")
            return False
        
        dataset_names = data_loader.get_dataset_names()
        total_datasets = len(dataset_names)
        
        print(f"\n📊 Ready to analyze {total_datasets} datasets")
        
        # 5. Process each dataset
        all_results = {}
        
        for i, dataset_name in enumerate(dataset_names, 1):
            # Validate dataset
            if not data_loader.validate_dataset(dataset_name):
                print(f"⚠️ Skipping {dataset_name} - validation failed")
                continue
            
            # Get dataset
            data = data_loader.get_dataset(dataset_name)
            if data is None:
                print(f"⚠️ Could not load {dataset_name}")
                continue
            
            # Run analysis
            result = run_colab_analysis_for_dataset(
                dataset_name, data, i, total_datasets, colab_config
            )
            
            if result:
                all_results[dataset_name] = result
        
        # 6. Generate summary
        generate_colab_summary(all_results)
        
        print(f"\n🎉 Colab analysis completed!")
        print(f"📊 Successfully processed: {len(all_results)}/{total_datasets} datasets")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Colab analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_colab_summary(results):
    """Generate Colab-friendly summary."""
    
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"\n{'='*70}")
    print("📊 MULTI-DATASET SUMMARY")
    print(f"{'='*70}")
    
    # Create summary table
    summary_data = []
    for dataset_name, automl in results.items():
        if hasattr(automl, 'results') and automl.results:
            metrics = automl.results
            summary_data.append({
                'Dataset': dataset_name,
                'Model': metrics.get('model_name', 'Unknown'),
                'RMSE': f"{metrics.get('rmse', 0):.3f}",
                'R²': f"{metrics.get('r2', 0):.3f}",
                'Clinical Excellent': f"{metrics.get('clinical_excellent', 0):.1f}%",
                'Clinical Good': f"{metrics.get('clinical_good', 0):.1f}%",
                'Sample Size': metrics.get('sample_size', 0)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n📋 Performance Comparison:")
        print(summary_df.to_string(index=False))
        
        # Find best dataset
        if len(summary_data) > 1:
            best_idx = summary_df['R²'].astype(float).idxmax()
            best_dataset = summary_df.iloc[best_idx]['Dataset']
            best_r2 = summary_df.iloc[best_idx]['R²']
            print(f"\n🏆 Best Performing Dataset: {best_dataset} (R² = {best_r2})")
        
        print(f"\n💾 Colab Results:")
        print(f"   📊 {len(results)} models created")
        print(f"   📈 Visualizations displayed above")
        print(f"   🏥 Clinical metrics calculated")
        print(f"   🚀 Ready for further analysis in Colab")

# ============================================================================
# SECTION 5: Colab Execution
# ============================================================================

# For Jupyter/Colab execution
if __name__ == "__main__":
    # Run the analysis
    results = main_colab()
    
    if results:
        print(f"\n✅ SUCCESS! Analysis completed for {len(results)} datasets")
        print("📊 Scroll up to see visualizations and detailed reports")
        print("🔬 Models are ready for further experimentation in Colab")
    else:
        print("\n❌ Analysis failed - check error messages above")
        print("💡 Tips for troubleshooting:")
        print("   1. Ensure datasets are uploaded to Colab")
        print("   2. Check file paths in COLAB_CONFIG")
        print("   3. Try restarting runtime if PyCaret import fails")

# Display instructions for Colab users
print("""
🎯 GOOGLE COLAB USAGE INSTRUCTIONS:

1. UPLOAD DATASETS:
   - Click the folder icon in the left sidebar
   - Upload your 3 diabetes CSV files
   - Or mount Google Drive and update paths in COLAB_CONFIG

2. RUN THE ANALYSIS:
   - Execute all cells in order
   - The analysis will process all datasets automatically
   - Visualizations will appear inline

3. CUSTOMIZE SETTINGS:
   - Modify COLAB_CONFIG at the top for your setup
   - Adjust resource settings based on your Colab type
   - Enable GPU if available for faster processing

4. TROUBLESHOOTING:
   - If PyCaret import fails, restart runtime
   - Ensure Python version is 3.9-3.11 (Colab default is fine)
   - Check that CSV files have 'PostBLHBA1C' column

Happy analyzing! 🚀
""")