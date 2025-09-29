#!/usr/bin/env python3
"""
Diabetes HbA1c Prediction using Auto-sklearn
Google Colab Optimized Script

This script implements Auto-sklearn for predicting PostBLHBA1C (post-intervention HbA1c levels)
from three diabetes intervention datasets using k-fold cross-validation with separate models
for each dataset.

Author: Auto-generated for Diabetes Research
Date: 2025
Target Variable: PostBLHBA1C
Datasets: 3 diabetes intervention studies
Framework: Auto-sklearn with scikit-learn ecosystem
"""

# ============================================================================
# SECTION 1: Google Colab Setup and Package Installation
# ============================================================================

import sys
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_packages():
    """Install required packages for Google Colab environment."""
    print("🔧 Installing required packages for Auto-sklearn...")
    
    packages = [
        'auto-sklearn',
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'joblib'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Could not install {package}: {e}")
    
    print("🎉 Package installation completed!")

def mount_google_drive():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("⚠️ Not running in Google Colab environment - skipping Drive mount")
        return False
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def setup_colab_environment():
    """Setup Google Colab environment with optimizations."""
    print("🚀 Setting up Google Colab environment...")
    
    # Install packages
    install_packages()
    
    # Mount Google Drive
    drive_mounted = mount_google_drive()
    
    # Memory optimization
    import gc
    gc.collect()
    
    # Set environment variables for better performance
    os.environ['JOBLIB_MULTIPROCESSING'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    print("✅ Colab environment setup completed!")
    return drive_mounted

# ============================================================================
# SECTION 2: Import Libraries
# ============================================================================

# Core data science libraries
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import joblib
from pathlib import Path

# Auto-sklearn installation and import
try:
    import autosklearn.regression
    from autosklearn.metrics import mean_squared_error, mean_absolute_error, r2
    print("✅ Auto-sklearn successfully imported")
except ImportError:
    print("⚠️ Auto-sklearn not found. Installing latest versions...")
    
    # Clean installation sequence for latest versions
    print("🧹 Step 1: Uninstalling conflicting packages...")
    conflicting_packages = [
        "Cython", "scipy", "pyparsing", "scikit_learn", 
        "scikit-learn", "imbalanced-learn", "mlxtend", "yellowbrick"
    ]
    for package in conflicting_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass  # Continue if package not found
    
    print("📦 Step 2: Installing latest dependency versions...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyparsing"])
    
    print("📦 Step 3: Installing latest scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    
    print("📦 Step 4: Installing Auto-sklearn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "auto-sklearn"])
    
    print("🔄 Step 5: Importing Auto-sklearn after installation...")
    import autosklearn.regression
    from autosklearn.metrics import mean_squared_error, mean_absolute_error, r2
    print("✅ Auto-sklearn successfully installed and imported")

# Scikit-learn modules for preprocessing and evaluation
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - using matplotlib only")

# ============================================================================
# Auto-sklearn Configuration and Setup
# ============================================================================

# ============================================================================
# SECTION 3: Configuration and Constants
# ============================================================================

class Config:
    """Configuration class for the Auto-sklearn diabetes prediction pipeline."""
    
    # Dataset configuration
    DATASETS = {
        'nmbfinalDiabetes4': 'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
        'nmbfinalnewDiabetes3': 'nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
        'PrePostFinal3': 'PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv'
    }
    
    # Target variable
    TARGET_COLUMN = 'PostBLHBA1C'
    
    # Cross-validation configuration
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Auto-sklearn configuration (Colab-optimized)
    AUTOSKLEARN_CONFIG = {
        'time_left_for_this_task': 300,  # 5 minutes per model (Colab resource constraint)
        'per_run_time_limit': 30,       # 30 seconds per individual model
        'memory_limit': 3072,           # 3GB memory limit for Colab
        'ensemble_size': 50,            # Smaller ensemble for faster inference
        'n_jobs': 2,                    # Limited cores in Colab
        'resampling_strategy': 'cv',    # Cross-validation
        'resampling_strategy_arguments': {'folds': 3},  # Reduced folds for speed
        'delete_tmp_folder_after_terminate': True,
        'seed': RANDOM_STATE
    }
    
    # Clinical thresholds for HbA1c accuracy
    CLINICAL_THRESHOLDS = {
        'excellent': 0.5,  # Within ±0.5% HbA1c
        'good': 1.0        # Within ±1.0% HbA1c
    }
    
    # Visualization configuration
    FIGURE_SIZE = (12, 8)
    DPI = 100
    
    # File paths (will be updated based on environment)
    DATA_PATH = None
    OUTPUT_PATH = None

# ============================================================================
# SECTION 4: Data Loading and Preprocessing Pipeline
# ============================================================================

class DataLoader:
    """Data loading and preprocessing for Google Colab environment."""
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.preprocessors = {}
        
    def setup_paths(self, drive_mounted=False):
        """Setup data paths based on environment."""
        if drive_mounted:
            # Google Drive paths
            base_path = '/content/drive/MyDrive/diabetes_data/'
            self.config.DATA_PATH = base_path
            self.config.OUTPUT_PATH = '/content/drive/MyDrive/diabetes_results/'
            print(f"📁 Using Google Drive paths: {base_path}")
        else:
            # Local paths for testing
            self.config.DATA_PATH = './'
            self.config.OUTPUT_PATH = './results/'
            print(f"📁 Using local paths: ./")
            
        # Create output directory
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)
    
    def load_datasets(self):
        """Load all three diabetes datasets."""
        print("📊 Loading diabetes datasets...")
        
        for dataset_name, filename in self.config.DATASETS.items():
            filepath = os.path.join(self.config.DATA_PATH, filename)
            
            try:
                df = pd.read_csv(filepath)
                self.datasets[dataset_name] = df
                print(f"✅ Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Quick data validation
                if self.config.TARGET_COLUMN not in df.columns:
                    print(f"⚠️ Warning: Target column '{self.config.TARGET_COLUMN}' not found in {dataset_name}")
                else:
                    target_stats = df[self.config.TARGET_COLUMN].describe()
                    print(f"   Target range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
                    
            except FileNotFoundError:
                print(f"❌ Error: Could not find file {filepath}")
                print(f"   Please ensure the file exists in {self.config.DATA_PATH}")
            except Exception as e:
                print(f"❌ Error loading {dataset_name}: {e}")
        
        return len(self.datasets) > 0
    
    def get_feature_types(self, df):
        """Identify numerical and categorical features."""
        # Exclude target column
        features = [col for col in df.columns if col != self.config.TARGET_COLUMN]
        
        numerical_features = []
        categorical_features = []
        
        for col in features:
            # Check if column is numerical
            if df[col].dtype in ['int64', 'float64']:
                # Additional check for categorical variables encoded as numbers
                unique_values = df[col].nunique()
                if unique_values <= 10 and df[col].dtype == 'int64':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        return numerical_features, categorical_features
    
    def create_preprocessor(self, df):
        """Create preprocessing pipeline for Auto-sklearn."""
        numerical_features, categorical_features = self.get_feature_types(df)
        
        print(f"   📈 Numerical features: {len(numerical_features)}")
        print(f"   📊 Categorical features: {len(categorical_features)}")
        
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing - with compatibility handling
        try:
            # Try new parameter name (scikit-learn >= 1.2)
            onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # Fall back to old parameter name (scikit-learn < 1.2)
            onehot_encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', onehot_encoder)
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor, numerical_features, categorical_features
    
    def prepare_dataset(self, dataset_name):
        """Prepare individual dataset for training."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        df = self.datasets[dataset_name].copy()
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[self.config.TARGET_COLUMN])
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"   🧹 Removed {initial_rows - final_rows} rows with missing target values")
        
        # Separate features and target
        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN]
        
        # Create preprocessor
        preprocessor, num_features, cat_features = self.create_preprocessor(df)
        
        # Store preprocessor for later use
        self.preprocessors[dataset_name] = {
            'preprocessor': preprocessor,
            'numerical_features': num_features,
            'categorical_features': cat_features
        }
        
        return X, y, preprocessor

# ============================================================================
# SECTION 5: K-Fold Cross-Validation Framework
# ============================================================================

class CrossValidationFramework:
    """K-fold cross-validation framework for Auto-sklearn with progress tracking."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def calculate_clinical_accuracy(self, y_true, y_pred, threshold):
        """Calculate clinical accuracy within specified HbA1c threshold."""
        absolute_errors = np.abs(y_true - y_pred)
        within_threshold = np.sum(absolute_errors <= threshold)
        accuracy = within_threshold / len(y_true)
        return accuracy
    
    def evaluate_predictions(self, y_true, y_pred, dataset_name, fold=None):
        """Comprehensive evaluation of predictions."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'clinical_excellent': self.calculate_clinical_accuracy(
                y_true, y_pred, self.config.CLINICAL_THRESHOLDS['excellent']
            ),
            'clinical_good': self.calculate_clinical_accuracy(
                y_true, y_pred, self.config.CLINICAL_THRESHOLDS['good']
            )
        }
        
        # Calculate additional statistics
        residuals = y_true - y_pred
        metrics.update({
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_error': np.min(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals))
        })
        
        return metrics
    
    def perform_cross_validation(self, X, y, dataset_name, preprocessor):
        """Perform k-fold cross-validation with Auto-sklearn."""
        print(f"🔄 Starting {self.config.N_FOLDS}-fold cross-validation for {dataset_name}...")
        
        # Initialize KFold
        kfold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        fold_results = []
        fold_predictions = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"   📊 Training fold {fold}/{self.config.N_FOLDS}...")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            
            # Create and train Auto-sklearn model
            automl = autosklearn.regression.AutoSklearnRegressor(**self.config.AUTOSKLEARN_CONFIG)
            
            try:
                # Fit the model
                automl.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred = automl.predict(X_val_processed)
                
                # Evaluate fold
                fold_metrics = self.evaluate_predictions(y_val, y_pred, dataset_name, fold)
                fold_metrics['fold'] = fold
                fold_results.append(fold_metrics)
                
                # Store predictions and model
                fold_predictions.append({
                    'y_true': y_val.values,
                    'y_pred': y_pred,
                    'indices': val_idx
                })
                fold_models.append(automl)
                
                print(f"   ✅ Fold {fold} completed - RMSE: {fold_metrics['rmse']:.3f}, "
                      f"R²: {fold_metrics['r2']:.3f}, Clinical(±0.5%): {fold_metrics['clinical_excellent']:.1%}")
                
            except Exception as e:
                print(f"   ❌ Error in fold {fold}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
        
        # Aggregate results
        if fold_results:
            cv_results = self._aggregate_cv_results(fold_results, fold_predictions, dataset_name)
            cv_results['models'] = fold_models
            return cv_results
        else:
            print(f"❌ All folds failed for {dataset_name}")
            return None
    
    def _aggregate_cv_results(self, fold_results, fold_predictions, dataset_name):
        """Aggregate cross-validation results across folds."""
        df_results = pd.DataFrame(fold_results)
        
        # Calculate mean and std for each metric
        metrics = ['rmse', 'mae', 'r2', 'clinical_excellent', 'clinical_good']
        aggregated = {}
        
        for metric in metrics:
            aggregated[f'{metric}_mean'] = df_results[metric].mean()
            aggregated[f'{metric}_std'] = df_results[metric].std()
            aggregated[f'{metric}_cv'] = df_results[metric].std() / df_results[metric].mean()
        
        # Combine all predictions
        all_y_true = np.concatenate([fold['y_true'] for fold in fold_predictions])
        all_y_pred = np.concatenate([fold['y_pred'] for fold in fold_predictions])
        
        # Overall evaluation
        overall_metrics = self.evaluate_predictions(all_y_true, all_y_pred, dataset_name)
        
        results = {
            'dataset': dataset_name,
            'fold_results': df_results,
            'aggregated_metrics': aggregated,
            'overall_metrics': overall_metrics,
            'predictions': {
                'y_true': all_y_true,
                'y_pred': all_y_pred
            }
        }
        
        return results

# ============================================================================
# SECTION 6: Model Training Orchestrator
# ============================================================================

class AutoSklearnOrchestrator:
    """Main orchestrator for training Auto-sklearn models on multiple datasets."""
    
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.cv_framework = CrossValidationFramework(config)
        self.results = {}
        
    def setup_environment(self, drive_mounted=False):
        """Setup the complete training environment."""
        print("🚀 Setting up Auto-sklearn training environment...")
        
        # Setup paths
        self.data_loader.setup_paths(drive_mounted)
        
        # Load datasets
        success = self.data_loader.load_datasets()
        if not success:
            raise RuntimeError("Failed to load datasets")
        
        print("✅ Environment setup completed!")
        return True
    
    def train_dataset(self, dataset_name):
        """Train Auto-sklearn model on a single dataset."""
        print(f"\n🎯 Training Auto-sklearn model for {dataset_name}...")
        
        try:
            # Prepare dataset
            X, y, preprocessor = self.data_loader.prepare_dataset(dataset_name)
            
            print(f"   Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"   Target range: {y.min():.2f} - {y.max():.2f}")
            
            # Perform cross-validation
            cv_results = self.cv_framework.perform_cross_validation(X, y, dataset_name, preprocessor)
            
            if cv_results is not None:
                self.results[dataset_name] = cv_results
                self._print_dataset_summary(dataset_name, cv_results)
                return True
            else:
                print(f"❌ Failed to train model for {dataset_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            return False
    
    def train_all_datasets(self):
        """Train Auto-sklearn models on all datasets sequentially."""
        print("\n🚀 Starting sequential training on all datasets...")
        
        successful_trainings = 0
        total_datasets = len(self.config.DATASETS)
        
        for dataset_name in self.config.DATASETS.keys():
            success = self.train_dataset(dataset_name)
            if success:
                successful_trainings += 1
                
        print(f"\n🎉 Training completed! {successful_trainings}/{total_datasets} datasets trained successfully.")
        
        if successful_trainings > 0:
            self._print_overall_summary()
            
        return successful_trainings > 0
    
    def _print_dataset_summary(self, dataset_name, results):
        """Print summary for a single dataset."""
        metrics = results['overall_metrics']
        
        print(f"\n📊 {dataset_name} Results Summary:")
        print(f"   🎯 RMSE: {metrics['rmse']:.3f}")
        print(f"   📐 MAE: {metrics['mae']:.3f}")
        print(f"   📈 R²: {metrics['r2']:.3f}")
        print(f"   🏥 Clinical Accuracy (±0.5%): {metrics['clinical_excellent']:.1%}")
        print(f"   🏥 Clinical Accuracy (±1.0%): {metrics['clinical_good']:.1%}")
    
    def _print_overall_summary(self):
        """Print overall summary across all datasets."""
        print(f"\n{'='*60}")
        print("🏆 OVERALL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        for dataset_name, results in self.results.items():
            metrics = results['overall_metrics']
            print(f"\n📊 {dataset_name}:")
            print(f"   RMSE: {metrics['rmse']:.3f} | R²: {metrics['r2']:.3f} | "
                  f"Clinical(±0.5%): {metrics['clinical_excellent']:.1%}")

# ============================================================================
# SECTION 7: Visualization and Reporting
# ============================================================================

class ResultsVisualizer:
    """Comprehensive visualization and reporting for Auto-sklearn results."""
    
    def __init__(self, config, results):
        self.config = config
        self.results = results
        
    def create_prediction_plots(self):
        """Create prediction vs actual plots for all datasets."""
        n_datasets = len(self.results)
        
        if PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=1, cols=n_datasets,
                subplot_titles=list(self.results.keys()),
                specs=[[{"secondary_y": False}] * n_datasets]
            )
            
            for i, (dataset_name, results) in enumerate(self.results.items(), 1):
                y_true = results['predictions']['y_true']
                y_pred = results['predictions']['y_pred']
                
                # Scatter plot
                fig.add_scatter(
                    x=y_true, y=y_pred,
                    mode='markers',
                    name=f'{dataset_name} Predictions',
                    row=1, col=i
                )
                
                # Perfect prediction line
                min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                fig.add_scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red'),
                    row=1, col=i
                )
            
            title = "Auto-sklearn Predictions vs Actual HbA1c Values"
            fig.update_layout(
                title=title,
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Actual HbA1c")
            fig.update_yaxes(title_text="Predicted HbA1c")
            
            fig.show()
            
        else:
            # Matplotlib fallback
            fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))
            if n_datasets == 1:
                axes = [axes]
                
            for i, (dataset_name, results) in enumerate(self.results.items()):
                y_true = results['predictions']['y_true']
                y_pred = results['predictions']['y_pred']
                
                axes[i].scatter(y_true, y_pred, alpha=0.6)
                
                # Perfect prediction line
                min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[i].set_xlabel('Actual HbA1c')
                axes[i].set_ylabel('Predicted HbA1c')
                axes[i].set_title(f'{dataset_name}\nR² = {results["overall_metrics"]["r2"]:.3f}')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def create_metrics_comparison(self):
        """Create metrics comparison chart across datasets."""
        metrics_data = []
        
        for dataset_name, results in self.results.items():
            metrics = results['overall_metrics']
            metrics_data.append({
                'Dataset': dataset_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'Clinical (±0.5%)': metrics['clinical_excellent'],
                'Clinical (±1.0%)': metrics['clinical_good']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        if PLOTLY_AVAILABLE:
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=['RMSE', 'MAE', 'R²', 'Clinical (±0.5%)', 'Clinical (±1.0%)', 'Summary'],
                specs=[[{"secondary_y": False}]*3, [{"secondary_y": False}]*3]
            )
            
            metrics_to_plot = ['RMSE', 'MAE', 'R²', 'Clinical (±0.5%)', 'Clinical (±1.0%)']
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            
            for metric, (row, col) in zip(metrics_to_plot, positions):
                fig.add_bar(
                    x=df_metrics['Dataset'], y=df_metrics[metric],
                    name=metric,
                    row=row, col=col
                )
            
            title = "Auto-sklearn Performance Metrics Comparison"
            fig.update_layout(
                title=title,
                height=600,
                showlegend=False
            )
            
            fig.show()
            
        else:
            # Matplotlib fallback
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics_to_plot = ['RMSE', 'MAE', 'R²', 'Clinical (±0.5%)', 'Clinical (±1.0%)']
            
            for i, metric in enumerate(metrics_to_plot):
                axes[i].bar(df_metrics['Dataset'], df_metrics[metric])
                axes[i].set_title(metric)
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
            # Summary table in last subplot
            axes[5].axis('tight')
            axes[5].axis('off')
            table = axes[5].table(cellText=df_metrics.round(3).values,
                                 colLabels=df_metrics.columns,
                                 cellLoc='center',
                                 loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            axes[5].set_title('Results Summary')
            
            plt.tight_layout()
            plt.show()
        
        return df_metrics
    
    def create_residual_analysis(self):
        """Create residual analysis plots."""
        n_datasets = len(self.results)
        
        fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 8))
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (dataset_name, results) in enumerate(self.results.items()):
            y_true = results['predictions']['y_true']
            y_pred = results['predictions']['y_pred']
            residuals = y_true - y_pred
            
            # Residuals vs predicted
            axes[0, i].scatter(y_pred, residuals, alpha=0.6)
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_xlabel('Predicted HbA1c')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].set_title(f'{dataset_name}\nResiduals vs Predicted')
            axes[0, i].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1, i].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, i].axvline(x=0, color='r', linestyle='--')
            axes[1, i].set_xlabel('Residuals')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'Residuals Distribution\nMean: {np.mean(residuals):.3f}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_clinical_report(self):
        """Generate clinical interpretation report."""
        print(f"\n{'='*80}")
        print("🏥 CLINICAL INTERPRETATION REPORT")
        print(f"{'='*80}")
        
        for dataset_name, results in self.results.items():
            metrics = results['overall_metrics']
            y_true = results['predictions']['y_true']
            y_pred = results['predictions']['y_pred']
            
            print(f"\n📊 {dataset_name.upper()} ANALYSIS:")
            print(f"   Sample Size: {len(y_true)} patients")
            print(f"   HbA1c Range: {y_true.min():.1f}% - {y_true.max():.1f}%")
            
            print(f"\n   🎯 Prediction Accuracy:")
            print(f"   • RMSE: {metrics['rmse']:.3f}% HbA1c")
            print(f"   • MAE: {metrics['mae']:.3f}% HbA1c")
            print(f"   • R²: {metrics['r2']:.3f}")
            
            print(f"\n   🏥 Clinical Relevance:")
            excellent_pct = metrics['clinical_excellent'] * 100
            good_pct = metrics['clinical_good'] * 100
            print(f"   • Excellent accuracy (±0.5%): {excellent_pct:.1f}% of predictions")
            print(f"   • Good accuracy (±1.0%): {good_pct:.1f}% of predictions")
            
            # Clinical recommendations
            if metrics['r2'] > 0.7:
                performance = "Excellent"
            elif metrics['r2'] > 0.5:
                performance = "Good"
            elif metrics['r2'] > 0.3:
                performance = "Moderate"
            else:
                performance = "Poor"
                
            print(f"   • Model Performance: {performance}")
            
            if excellent_pct > 70:
                clinical_utility = "High clinical utility - suitable for clinical decision support"
            elif excellent_pct > 50:
                clinical_utility = "Moderate clinical utility - useful for screening and monitoring"
            else:
                clinical_utility = "Limited clinical utility - requires further improvement"
                
            print(f"   • Clinical Utility: {clinical_utility}")
    
    def save_results(self):
        """Save results to files."""
        output_dir = self.config.OUTPUT_PATH
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics summary
        metrics_data = []
        for dataset_name, results in self.results.items():
            metrics = results['overall_metrics'].copy()
            metrics['dataset'] = dataset_name
            metrics_data.append(metrics)
        
        df_summary = pd.DataFrame(metrics_data)
        summary_file = os.path.join(output_dir, f'autosklearn_results_summary_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"📁 Results summary saved to: {summary_file}")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f'autosklearn_detailed_results_{timestamp}.pkl')
        joblib.dump(self.results, results_file)
        print(f"📁 Detailed results saved to: {results_file}")

# ============================================================================
# SECTION 8: Main Execution Pipeline
# ============================================================================

def main():
    """Main execution pipeline for Google Colab diabetes HbA1c prediction."""
    
    print("🚀 Diabetes HbA1c Prediction with Auto-sklearn")
    print("=" * 60)
    
    # Setup Google Colab environment
    drive_mounted = setup_colab_environment()
    
    # Initialize configuration
    config = Config()
    
    # Initialize orchestrator
    orchestrator = AutoSklearnOrchestrator(config)
    
    try:
        # Setup environment and load data
        orchestrator.setup_environment(drive_mounted)
        
        # Train models on all datasets
        success = orchestrator.train_all_datasets()
        
        if success:
            # Create visualizations and reports
            visualizer = ResultsVisualizer(config, orchestrator.results)
            
            print("\n📊 Creating visualizations...")
            visualizer.create_prediction_plots()
            
            print("\n📈 Creating metrics comparison...")
            metrics_df = visualizer.create_metrics_comparison()
            
            print("\n📉 Creating residual analysis...")
            visualizer.create_residual_analysis()
            
            print("\n🏥 Generating clinical report...")
            visualizer.generate_clinical_report()
            
            print("\n💾 Saving results...")
            visualizer.save_results()
            
            print("\n🎉 Analysis completed successfully!")
            
        else:
            print("❌ Training failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SECTION 9: Interactive Functions for Colab
# ============================================================================

def quick_analysis(dataset_path=None, time_limit=180):
    """
    Quick analysis function for Google Colab with minimal configuration.
    
    Parameters:
    -----------
    dataset_path : str, optional
        Path to the directory containing CSV files
    time_limit : int, default=180
        Time limit in seconds for each Auto-sklearn run
    """
    
    print("🚀 Quick Auto-sklearn Analysis for Diabetes HbA1c Prediction")
    
    # Setup environment
    drive_mounted = setup_colab_environment()
    
    # Initialize with quick config
    config = Config()
    config.AUTOSKLEARN_CONFIG['time_left_for_this_task'] = time_limit
    config.AUTOSKLEARN_CONFIG['per_run_time_limit'] = min(30, time_limit // 10)
    
    # Override dataset path if provided
    if dataset_path:
        config.DATA_PATH = dataset_path
    
    orchestrator = AutoSklearnOrchestrator(config)
    orchestrator.setup_environment(drive_mounted)
    
    # Train models
    success = orchestrator.train_all_datasets()
    
    if success:
        # Quick visualization
        visualizer = ResultsVisualizer(config, orchestrator.results)
        visualizer.create_prediction_plots()
        visualizer.generate_clinical_report()
        
        return orchestrator.results
    else:
        return None

def load_and_predict(model_path, dataset_path, dataset_name):
    """
    Load a trained model and make predictions on new data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    dataset_path : str
        Path to the CSV file for prediction
    dataset_name : str
        Name of the dataset for preprocessing
    """
    
    try:
        # Load results
        results = joblib.load(model_path)
        
        if dataset_name not in results:
            print(f"❌ Dataset {dataset_name} not found in saved results")
            return None
        
        # Load new data
        df_new = pd.read_csv(dataset_path)
        print(f"📊 Loaded new data: {df_new.shape[0]} rows, {df_new.shape[1]} columns")
        
        # Get the best model from CV results
        models = results[dataset_name]['models']
        best_model = models[0]  # Use first fold model as representative
        
        # Make predictions (would need preprocessor restoration)
        print("⚠️ Note: Full prediction pipeline requires preprocessor restoration")
        print("   This is a placeholder for the prediction functionality")
        
        return df_new
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return None

# ============================================================================
# SECTION 10: Execution
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    main()

# ============================================================================
# COLAB USAGE INSTRUCTIONS
# ============================================================================

"""
🚀 GOOGLE COLAB USAGE INSTRUCTIONS:

1. BASIC SETUP:
   - Upload this script to Google Colab
   - Upload your CSV files to Google Drive in a folder named 'diabetes_data'
   - Run the entire script or use the functions below

2. QUICK START:
   # Run complete analysis
   main()
   
   # Or run quick analysis with custom settings
   results = quick_analysis(time_limit=300)

3. FILE STRUCTURE IN GOOGLE DRIVE:
   MyDrive/
   ├── diabetes_data/
   │   ├── nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv
   │   ├── nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv
   │   └── PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv
   └── diabetes_results/ (created automatically)

4. CUSTOMIZATION:
   - Modify Config class for different settings
   - Adjust AUTOSKLEARN_CONFIG for different time/memory limits
   - Change N_FOLDS for different cross-validation strategies

5. TROUBLESHOOTING:
   - Ensure CSV files have 'PostBLHBA1C' column
   - Check file paths and permissions
   - Monitor memory usage in Colab
   - Reduce time limits if running into memory issues

6. EXPECTED OUTPUTS:
   - Prediction vs actual plots
   - Metrics comparison charts
   - Residual analysis
   - Clinical interpretation report
   - Saved results files in Google Drive

📊 The script automatically handles:
   ✅ Package installation
   ✅ Google Drive mounting
   ✅ Data preprocessing
   ✅ Feature encoding
   ✅ K-fold cross-validation
   ✅ Model training with Auto-sklearn
   ✅ Performance evaluation
   ✅ Clinical metrics calculation
   ✅ Comprehensive visualization
   ✅ Results saving
"""