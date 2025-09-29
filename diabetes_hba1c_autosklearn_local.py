#!/usr/bin/env python3
"""
Diabetes HbA1c Prediction using Auto-sklearn - Local System Optimized
Single Dataset Analysis for Local Development

This script implements Auto-sklearn for predicting PostBLHBA1C (post-intervention HbA1c levels)
from the nmbfinalDiabetes (4) dataset using k-fold cross-validation.

Optimized for: 8-core system with 16GB RAM
Target Variable: PostBLHBA1C
Dataset: nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv
Framework: Auto-sklearn with local system optimization
"""

# ============================================================================
# SECTION 1: Local System Setup and Package Installation
# ============================================================================

import sys
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_packages():
    """Install required packages for local environment."""
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

def setup_local_environment():
    """Setup local environment with optimizations for 8-core/16GB system."""
    print("🚀 Setting up local environment for 8-core system...")
    
    # Memory optimization for local system
    import gc
    gc.collect()
    
    # Set environment variables for better performance on local system
    os.environ['JOBLIB_MULTIPROCESSING'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Use 4 threads for BLAS operations
    os.environ['MKL_NUM_THREADS'] = '4'       # Use 4 threads for Intel MKL
    os.environ['OMP_NUM_THREADS'] = '4'       # Use 4 threads for OpenMP
    
    print("✅ Local environment setup completed!")
    return True

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
# SECTION 3: Configuration and Constants - Optimized for Local System
# ============================================================================

class Config:
    """Configuration class for the Auto-sklearn diabetes prediction pipeline - Local System Optimized."""
    
    # Dataset configuration - Single dataset focus
    DATASET_NAME = 'nmbfinalDiabetes4'
    DATASET_FILE = 'nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv'
    
    # Target variable
    TARGET_COLUMN = 'PostBLHBA1C'
    
    # Cross-validation configuration - More folds for better evaluation on local system
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Auto-sklearn configuration - Optimized for 8-core/16GB system
    AUTOSKLEARN_CONFIG = {
        'time_left_for_this_task': 1800,     # 30 minutes total (more time on local system)
        'per_run_time_limit': 120,           # 2 minutes per individual model (longer for local)
        'memory_limit': 3072,                # 3GB memory limit per model (recommended)
        'ensemble_size': 50,                 # Standard ensemble size
        'n_jobs': 4,                         # Use 4 cores (recommended for 8-core system)
        'resampling_strategy': 'cv',         # Cross-validation
        'resampling_strategy_arguments': {'folds': 5},  # 5 folds for thorough evaluation
        'delete_tmp_folder_after_terminate': True,
        'seed': RANDOM_STATE,
        'initial_configurations_via_metalearning': 25,   # More initial configs
        'smac_scenario_args': {
            'runcount_limit': 50             # Allow more model evaluations
        }
    }
    
    # Alternative high-performance configuration (uncomment if you want more aggressive settings)
    AUTOSKLEARN_CONFIG_AGGRESSIVE = {
        'time_left_for_this_task': 3600,     # 60 minutes total
        'per_run_time_limit': 180,           # 3 minutes per model
        'memory_limit': 4096,                # 4GB memory limit
        'ensemble_size': 100,                # Larger ensemble
        'n_jobs': 6,                         # Use 6 cores (more aggressive)
        'resampling_strategy': 'cv',
        'resampling_strategy_arguments': {'folds': 5},
        'delete_tmp_folder_after_terminate': True,
        'seed': RANDOM_STATE,
        'initial_configurations_via_metalearning': 50,
        'smac_scenario_args': {
            'runcount_limit': 100
        }
    }
    
    # Clinical thresholds for HbA1c accuracy
    CLINICAL_THRESHOLDS = {
        'excellent': 0.5,  # Within ±0.5% HbA1c
        'good': 1.0        # Within ±1.0% HbA1c
    }
    
    # Visualization configuration
    FIGURE_SIZE = (12, 8)
    DPI = 100
    
    # File paths - Local system paths
    DATA_PATH = './final_imputed_data/'
    OUTPUT_PATH = './results/'

# ============================================================================
# SECTION 4: Data Loading and Preprocessing Pipeline
# ============================================================================

class DataLoader:
    """Data loading and preprocessing for local environment."""
    
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.preprocessor = None
        
    def setup_paths(self):
        """Setup data paths for local environment."""
        print(f"📁 Using local paths:")
        print(f"   Data: {self.config.DATA_PATH}")
        print(f"   Output: {self.config.OUTPUT_PATH}")
            
        # Create output directory
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)
    
    def load_dataset(self):
        """Load the diabetes dataset."""
        print("📊 Loading diabetes dataset...")
        
        filepath = os.path.join(self.config.DATA_PATH, self.config.DATASET_FILE)
        
        try:
            df = pd.read_csv(filepath)
            self.dataset = df
            print(f"✅ Loaded {self.config.DATASET_NAME}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Quick data validation
            if self.config.TARGET_COLUMN not in df.columns:
                print(f"⚠️ Warning: Target column '{self.config.TARGET_COLUMN}' not found")
                return False
            else:
                target_stats = df[self.config.TARGET_COLUMN].describe()
                print(f"   Target range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
                print(f"   Missing values in target: {df[self.config.TARGET_COLUMN].isnull().sum()}")
                
        except FileNotFoundError:
            print(f"❌ Error: Could not find file {filepath}")
            print(f"   Please ensure the file exists in {self.config.DATA_PATH}")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
        
        return True
    
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
        
        # Display feature details for debugging
        if len(numerical_features) <= 20:  # Only show if reasonable number
            print(f"   Numerical: {numerical_features}")
        if len(categorical_features) <= 20:
            print(f"   Categorical: {categorical_features}")
        
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
    
    def prepare_dataset(self):
        """Prepare dataset for training."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        df = self.dataset.copy()
        
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
        self.preprocessor = {
            'preprocessor': preprocessor,
            'numerical_features': num_features,
            'categorical_features': cat_features
        }
        
        print(f"   📋 Final dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        
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
    
    def evaluate_predictions(self, y_true, y_pred, fold=None):
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
    
    def perform_cross_validation(self, X, y, preprocessor, use_aggressive_config=False):
        """Perform k-fold cross-validation with Auto-sklearn."""
        print(f"🔄 Starting {self.config.N_FOLDS}-fold cross-validation...")
        
        # Choose configuration
        autosklearn_config = (self.config.AUTOSKLEARN_CONFIG_AGGRESSIVE 
                             if use_aggressive_config 
                             else self.config.AUTOSKLEARN_CONFIG)
        
        print(f"⚙️  Configuration:")
        print(f"   Time limit: {autosklearn_config['time_left_for_this_task']} seconds")
        print(f"   Per run limit: {autosklearn_config['per_run_time_limit']} seconds")
        print(f"   Memory limit: {autosklearn_config['memory_limit']} MB")
        print(f"   n_jobs: {autosklearn_config['n_jobs']}")
        
        # Initialize KFold
        kfold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        fold_results = []
        fold_predictions = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\n   📊 Training fold {fold}/{self.config.N_FOLDS}...")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"      Train size: {len(X_train)}, Validation size: {len(X_val)}")
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            
            print(f"      Processed shape: Train {X_train_processed.shape}, Val {X_val_processed.shape}")
            
            # Create and train Auto-sklearn model
            automl = autosklearn.regression.AutoSklearnRegressor(**autosklearn_config)
            
            try:
                # Fit the model
                print(f"      🤖 Training Auto-sklearn model...")
                automl.fit(X_train_processed, y_train)
                
                # Print model statistics
                print(f"      📈 Models evaluated: {len(automl.get_models_with_weights())}")
                
                # Make predictions
                y_pred = automl.predict(X_val_processed)
                
                # Evaluate fold
                fold_metrics = self.evaluate_predictions(y_val, y_pred, fold)
                fold_metrics['fold'] = fold
                fold_results.append(fold_metrics)
                
                # Store predictions and model
                fold_predictions.append({
                    'y_true': y_val.values,
                    'y_pred': y_pred,
                    'indices': val_idx
                })
                fold_models.append(automl)
                
                print(f"      ✅ Fold {fold} completed:")
                print(f"         RMSE: {fold_metrics['rmse']:.3f}")
                print(f"         R²: {fold_metrics['r2']:.3f}")
                print(f"         Clinical(±0.5%): {fold_metrics['clinical_excellent']:.1%}")
                
            except Exception as e:
                print(f"      ❌ Error in fold {fold}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
        
        # Aggregate results
        if fold_results:
            cv_results = self._aggregate_cv_results(fold_results, fold_predictions)
            cv_results['models'] = fold_models
            return cv_results
        else:
            print(f"❌ All folds failed")
            return None
    
    def _aggregate_cv_results(self, fold_results, fold_predictions):
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
        overall_metrics = self.evaluate_predictions(all_y_true, all_y_pred)
        
        results = {
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
    """Main orchestrator for training Auto-sklearn model on single dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.cv_framework = CrossValidationFramework(config)
        self.results = None
        
    def setup_environment(self):
        """Setup the complete training environment."""
        print("🚀 Setting up Auto-sklearn training environment...")
        
        # Setup paths
        self.data_loader.setup_paths()
        
        # Load dataset
        success = self.data_loader.load_dataset()
        if not success:
            raise RuntimeError("Failed to load dataset")
        
        print("✅ Environment setup completed!")
        return True
    
    def train_model(self, use_aggressive_config=False):
        """Train Auto-sklearn model on the dataset."""
        print(f"\n🎯 Training Auto-sklearn model for {self.config.DATASET_NAME}...")
        
        try:
            # Prepare dataset
            X, y, preprocessor = self.data_loader.prepare_dataset()
            
            print(f"   Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"   Target range: {y.min():.2f} - {y.max():.2f}")
            print(f"   Target mean: {y.mean():.2f} ± {y.std():.2f}")
            
            # Perform cross-validation
            cv_results = self.cv_framework.perform_cross_validation(X, y, preprocessor, use_aggressive_config)
            
            if cv_results is not None:
                self.results = cv_results
                self._print_results_summary()
                return True
            else:
                print(f"❌ Failed to train model")
                return False
                
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_results_summary(self):
        """Print summary of results."""
        metrics = self.results['overall_metrics']
        
        print(f"\n{'='*60}")
        print(f"🏆 {self.config.DATASET_NAME} Results Summary")
        print(f"{'='*60}")
        print(f"   🎯 RMSE: {metrics['rmse']:.3f}")
        print(f"   📐 MAE: {metrics['mae']:.3f}")
        print(f"   📈 R²: {metrics['r2']:.3f}")
        print(f"   🏥 Clinical Accuracy (±0.5%): {metrics['clinical_excellent']:.1%}")
        print(f"   🏥 Clinical Accuracy (±1.0%): {metrics['clinical_good']:.1%}")
        
        # Cross-validation statistics
        agg = self.results['aggregated_metrics']
        print(f"\n📊 Cross-Validation Statistics:")
        print(f"   RMSE: {agg['rmse_mean']:.3f} ± {agg['rmse_std']:.3f}")
        print(f"   R²: {agg['r2_mean']:.3f} ± {agg['r2_std']:.3f}")
        print(f"   Clinical(±0.5%): {agg['clinical_excellent_mean']:.1%} ± {agg['clinical_excellent_std']:.1%}")

# ============================================================================
# SECTION 7: Visualization and Reporting
# ============================================================================

class ResultsVisualizer:
    """Comprehensive visualization and reporting for Auto-sklearn results."""
    
    def __init__(self, config, results):
        self.config = config
        self.results = results
        
    def create_prediction_plots(self):
        """Create prediction vs actual plots."""
        y_true = self.results['predictions']['y_true']
        y_pred = self.results['predictions']['y_pred']
        
        if PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Predictions vs Actual', 'Residual Distribution'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Scatter plot
            fig.add_scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig.add_scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red'),
                row=1, col=1
            )
            
            # Residuals histogram
            residuals = y_true - y_pred
            fig.add_histogram(
                x=residuals,
                name='Residuals',
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Auto-sklearn Results - {self.config.DATASET_NAME} (R² = {self.results['overall_metrics']['r2']:.3f})",
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Actual HbA1c", row=1, col=1)
            fig.update_yaxes(title_text="Predicted HbA1c", row=1, col=1)
            fig.update_xaxes(title_text="Residuals", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.show()
            
        else:
            # Matplotlib fallback
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Prediction plot
            axes[0].scatter(y_true, y_pred, alpha=0.6)
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0].set_xlabel('Actual HbA1c')
            axes[0].set_ylabel('Predicted HbA1c')
            axes[0].set_title(f'Predictions vs Actual\nR² = {self.results["overall_metrics"]["r2"]:.3f}')
            axes[0].grid(True, alpha=0.3)
            
            # Residuals histogram
            residuals = y_true - y_pred
            axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='r', linestyle='--')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Residuals Distribution\nMean: {np.mean(residuals):.3f}')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def create_cv_results_plot(self):
        """Create cross-validation results visualization."""
        fold_results = self.results['fold_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # RMSE across folds
        axes[0,0].bar(fold_results['fold'], fold_results['rmse'])
        axes[0,0].set_title('RMSE by Fold')
        axes[0,0].set_xlabel('Fold')
        axes[0,0].set_ylabel('RMSE')
        
        # R² across folds
        axes[0,1].bar(fold_results['fold'], fold_results['r2'])
        axes[0,1].set_title('R² by Fold')
        axes[0,1].set_xlabel('Fold')
        axes[0,1].set_ylabel('R²')
        
        # Clinical accuracy across folds
        axes[1,0].bar(fold_results['fold'], fold_results['clinical_excellent'] * 100)
        axes[1,0].set_title('Clinical Accuracy (±0.5%) by Fold')
        axes[1,0].set_xlabel('Fold')
        axes[1,0].set_ylabel('Accuracy (%)')
        
        # MAE across folds
        axes[1,1].bar(fold_results['fold'], fold_results['mae'])
        axes[1,1].set_title('MAE by Fold')
        axes[1,1].set_xlabel('Fold')
        axes[1,1].set_ylabel('MAE')
        
        plt.tight_layout()
        plt.show()
    
    def generate_clinical_report(self):
        """Generate clinical interpretation report."""
        metrics = self.results['overall_metrics']
        y_true = self.results['predictions']['y_true']
        y_pred = self.results['predictions']['y_pred']
        
        print(f"\n{'='*80}")
        print("🏥 CLINICAL INTERPRETATION REPORT")
        print(f"{'='*80}")
        
        print(f"\n📊 {self.config.DATASET_NAME.upper()} ANALYSIS:")
        print(f"   Sample Size: {len(y_true)} patients")
        print(f"   HbA1c Range: {y_true.min():.1f}% - {y_true.max():.1f}%")
        print(f"   HbA1c Mean: {y_true.mean():.1f}% ± {y_true.std():.1f}%")
        
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
        
        # Cross-validation stability
        agg = self.results['aggregated_metrics']
        cv_stability = agg['r2_cv']  # Coefficient of variation
        if cv_stability < 0.1:
            stability = "Very stable"
        elif cv_stability < 0.2:
            stability = "Stable"
        else:
            stability = "Variable"
        print(f"   • Cross-validation Stability: {stability} (CV = {cv_stability:.3f})")
    
    def save_results(self):
        """Save results to files."""
        output_dir = self.config.OUTPUT_PATH
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics summary
        metrics_data = []
        metrics = self.results['overall_metrics'].copy()
        metrics['dataset'] = self.config.DATASET_NAME
        metrics['timestamp'] = timestamp
        
        # Add cross-validation statistics
        agg = self.results['aggregated_metrics']
        for key, value in agg.items():
            metrics[f'cv_{key}'] = value
        
        metrics_data.append(metrics)
        
        df_summary = pd.DataFrame(metrics_data)
        summary_file = os.path.join(output_dir, f'autosklearn_results_summary_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"📁 Results summary saved to: {summary_file}")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f'autosklearn_detailed_results_{timestamp}.pkl')
        joblib.dump(self.results, results_file)
        print(f"📁 Detailed results saved to: {results_file}")
        
        # Save fold-by-fold results
        fold_results_file = os.path.join(output_dir, f'autosklearn_fold_results_{timestamp}.csv')
        self.results['fold_results'].to_csv(fold_results_file, index=False)
        print(f"📁 Fold results saved to: {fold_results_file}")

# ============================================================================
# SECTION 8: Main Execution Pipeline
# ============================================================================

def main(use_aggressive_config=False):
    """Main execution pipeline for local diabetes HbA1c prediction."""
    
    print("🚀 Diabetes HbA1c Prediction with Auto-sklearn - Local System")
    print("=" * 70)
    print(f"Dataset: {Config.DATASET_NAME}")
    print(f"Configuration: {'Aggressive' if use_aggressive_config else 'Balanced'}")
    print("=" * 70)
    
    # Setup local environment
    setup_local_environment()
    
    # Initialize configuration
    config = Config()
    
    # Initialize orchestrator
    orchestrator = AutoSklearnOrchestrator(config)
    
    try:
        # Setup environment and load data
        orchestrator.setup_environment()
        
        # Train model
        success = orchestrator.train_model(use_aggressive_config)
        
        if success:
            # Create visualizations and reports
            visualizer = ResultsVisualizer(config, orchestrator.results)
            
            print("\n📊 Creating visualizations...")
            visualizer.create_prediction_plots()
            visualizer.create_cv_results_plot()
            
            print("\n🏥 Generating clinical report...")
            visualizer.generate_clinical_report()
            
            print("\n💾 Saving results...")
            visualizer.save_results()
            
            print("\n🎉 Analysis completed successfully!")
            
            return orchestrator.results
            
        else:
            print("❌ Training failed. Please check the error messages above.")
            return None
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# SECTION 9: Interactive Functions
# ============================================================================

def quick_analysis(time_limit=600):
    """
    Quick analysis function with reduced time limits.
    
    Parameters:
    -----------
    time_limit : int, default=600
        Time limit in seconds for Auto-sklearn run (10 minutes)
    """
    
    print("🚀 Quick Auto-sklearn Analysis for Diabetes HbA1c Prediction")
    
    # Modify config for quick analysis
    config = Config()
    config.AUTOSKLEARN_CONFIG['time_left_for_this_task'] = time_limit
    config.AUTOSKLEARN_CONFIG['per_run_time_limit'] = min(60, time_limit // 10)
    
    orchestrator = AutoSklearnOrchestrator(config)
    orchestrator.setup_environment()
    
    # Train model
    success = orchestrator.train_model()
    
    if success:
        # Quick visualization and report
        visualizer = ResultsVisualizer(config, orchestrator.results)
        visualizer.create_prediction_plots()
        visualizer.generate_clinical_report()
        
        return orchestrator.results
    else:
        return None

def compare_configurations():
    """Compare balanced vs aggressive configurations."""
    
    print("🔄 Comparing Balanced vs Aggressive Configurations")
    print("=" * 60)
    
    results = {}
    
    # Run balanced configuration
    print("\n1️⃣ Running Balanced Configuration...")
    results['balanced'] = main(use_aggressive_config=False)
    
    # Run aggressive configuration
    print("\n2️⃣ Running Aggressive Configuration...")
    results['aggressive'] = main(use_aggressive_config=True)
    
    # Compare results
    if results['balanced'] and results['aggressive']:
        print("\n📊 Configuration Comparison:")
        print("=" * 40)
        
        for config_name, result in results.items():
            metrics = result['overall_metrics']
            print(f"\n{config_name.upper()} Configuration:")
            print(f"   RMSE: {metrics['rmse']:.3f}")
            print(f"   R²: {metrics['r2']:.3f}")
            print(f"   Clinical (±0.5%): {metrics['clinical_excellent']:.1%}")
    
    return results

# ============================================================================
# SECTION 10: System Configuration Recommendations
# ============================================================================

def print_system_recommendations():
    """Print system-specific recommendations."""
    
    print("\n" + "="*70)
    print("🖥️  SYSTEM CONFIGURATION RECOMMENDATIONS")
    print("="*70)
    
    print("\n📊 Current Configuration (Balanced):")
    print("   • n_jobs: 4 (uses 4 of 8 cores)")
    print("   • memory_limit: 3072 MB (3GB per model)")
    print("   • time_left_for_this_task: 1800 seconds (30 minutes)")
    print("   • per_run_time_limit: 120 seconds (2 minutes per model)")
    
    print("\n⚡ Aggressive Configuration (if you want faster results):")
    print("   • n_jobs: 6 (uses 6 of 8 cores)")
    print("   • memory_limit: 4096 MB (4GB per model)")
    print("   • time_left_for_this_task: 3600 seconds (60 minutes)")
    print("   • per_run_time_limit: 180 seconds (3 minutes per model)")
    
    print("\n💡 Usage Recommendations:")
    print("   • Use main() for balanced approach (recommended)")
    print("   • Use main(use_aggressive_config=True) for aggressive approach")
    print("   • Use quick_analysis() for fast testing (10 minutes)")
    print("   • Use compare_configurations() to test both approaches")
    
    print("\n⚠️  Monitoring Tips:")
    print("   • Watch Task Manager for CPU and memory usage")
    print("   • If system becomes unresponsive, reduce n_jobs to 2-3")
    print("   • If memory errors occur, reduce memory_limit to 2048 MB")

# ============================================================================
# SECTION 11: Execution
# ============================================================================

if __name__ == "__main__":
    # Print system recommendations
    print_system_recommendations()
    
    # Ask user for configuration choice
    print("\n" + "="*50)
    print("Choose configuration:")
    print("1. Balanced (recommended)")
    print("2. Aggressive (faster, uses more resources)")
    print("3. Quick analysis (10 minutes)")
    print("4. Compare both configurations")
    
    try:
        choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
        if not choice:
            choice = "1"
            
        if choice == "1":
            main(use_aggressive_config=False)
        elif choice == "2":
            main(use_aggressive_config=True)
        elif choice == "3":
            quick_analysis()
        elif choice == "4":
            compare_configurations()
        else:
            print("Invalid choice, running balanced configuration...")
            main(use_aggressive_config=False)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Running default balanced configuration...")
        main(use_aggressive_config=False)