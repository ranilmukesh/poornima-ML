# filepath: c:\Users\maadh\OneDrive\Pictures\poornima-ML\ml_commented.py
#!/usr/bin/env python3
"""
Comprehensive Stack Ridge ML Model for HbA1c Prediction
========================================================
Target: PostBLHBA1C prediction with feature importance analysis
Datasets: All 3 diabetes datasets (nmbfinalDiabetes, nmbfinalnewDiabetes, PrePostFinal)
Approach: Stack Ridge ensemble with advanced feature engineering

Clinical Goal: Achieve MAE < 0.5 for accurate diabetes management
"""

# Import required libraries for data processing and machine learning
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Core ML libraries for model building and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                             GradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
import optuna  # For hyperparameter optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import advanced gradient boosting libraries with error handling
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

# Print pipeline initialization banner with key information
print("🚀 Stack Ridge ML Pipeline for HbA1c Prediction")
print("=" * 60)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Target: PostBLHBA1C prediction with MAE < 0.5")
print("📊 Datasets: 3 diabetes datasets with individual models")
print("🏗️ Architecture: Stack Ridge ensemble with feature importance")
print("=" * 60)

class DiabetesStackRidgeModel:
    """
    Comprehensive Stack Ridge Model for Diabetes HbA1c Prediction
    
    This class implements a complete machine learning pipeline for predicting HbA1c values
    in diabetes patients using a stacked ensemble approach with Ridge regression as the meta-learner.
    """
    
    def __init__(self, dataset_name, random_state=42):
        """
        Initialize the DiabetesStackRidgeModel
        
        Args:
            dataset_name (str): Name identifier for the dataset being processed
            random_state (int): Random seed for reproducible results
        """
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers compared to StandardScaler
        self.models = {}  # Dictionary to store trained base models
        self.stacked_model = None  # The final stacked ensemble model
        self.feature_importance = {}  # Feature importance from different methods
        self.results = {}  # Store evaluation results
        self.best_features = []  # Top performing features
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess diabetes dataset with outlier removal and data cleaning
        
        Args:
            file_path (str): Path to the CSV file containing the dataset
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataset
        """
        print(f"\n📂 Loading dataset: {self.dataset_name}")
        print(f"📄 File: {os.path.basename(file_path)}")
        
        # Load the CSV data into a DataFrame
        df = pd.read_csv(file_path)
        target_col = 'PostBLHBA1C'  # Define the target variable
        
        print(f"📊 Original shape: {df.shape}")
        
        # Validate that the target column exists in the dataset
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Remove rows with missing target values (critical for supervised learning)
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        
        # Conservative outlier removal using z-score method
        # Only remove extreme outliers (>4 standard deviations) to preserve data
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        z_scores = np.abs((df[target_col] - target_mean) / target_std)
        df = df[z_scores <= 4]  # Keep values within 4 standard deviations
        
        # Calculate and report data cleaning statistics
        outliers_removed = initial_rows - len(df)
        print(f"🧹 Cleaned shape: {df.shape} (removed {outliers_removed} outliers)")
        print(f"📈 HbA1c range: {df[target_col].min():.2f} - {df[target_col].max():.2f}")
        print(f"📊 HbA1c mean ± std: {df[target_col].mean():.2f} ± {df[target_col].std():.2f}")
        
        # Store metadata about the original dataset for reporting
        self.original_shape = df.shape
        self.target_stats = {
            'mean': df[target_col].mean(),
            'std': df[target_col].std(),
            'min': df[target_col].min(),
            'max': df[target_col].max()
        }
        
        return df
    
    def advanced_feature_engineering(self, df, target_col='PostBLHBA1C'):
        """
        Create advanced medical domain features for HbA1c prediction
        
        This method implements comprehensive feature engineering including:
        - Categorical encoding
        - Medical risk scores
        - Feature interactions
        - Non-linear transformations
        - Statistical aggregates
        - Domain-specific diabetes features
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: Enhanced dataset with engineered features
        """
        print(f"\n🧬 Advanced Feature Engineering for {self.dataset_name}")
        
        df_enhanced = df.copy()  # Create a copy to avoid modifying original data
        
        # Separate columns by data type for appropriate handling
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)  # Exclude target from feature list
        
        categorical_cols = df_enhanced.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"📊 Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        
        # Handle categorical variables using label encoding
        label_encoders = {}
        for col in categorical_cols:
            if df_enhanced[col].nunique() < 50:  # Only encode if reasonable number of categories
                le = LabelEncoder()
                df_enhanced[col] = le.fit_transform(df_enhanced[col].astype(str))
                label_encoders[col] = le
                numeric_cols.append(col)  # Add to numeric list after encoding
        
        # 1. Medical Risk Scores (diabetes-specific composite features)
        if len(numeric_cols) >= 3:
            print("🏥 Creating diabetes risk composite scores...")
            
            # Find features most correlated with HbA1c for risk scoring
            correlations = df_enhanced[numeric_cols].corrwith(df_enhanced[target_col]).abs()
            top_features = correlations.nlargest(min(10, len(correlations))).index.tolist()
            
            # Create composite risk scores from top correlated features
            for i, feat in enumerate(top_features[:5]):
                df_enhanced[f'diabetes_risk_score_{i+1}'] = df_enhanced[feat]
            
            # Create medical ratios (important clinical indicators in diabetes)
            if len(top_features) >= 2:
                feat1, feat2 = top_features[0], top_features[1]
                df_enhanced[f'medical_ratio_{feat1}_{feat2}'] = (
                    df_enhanced[feat1] / (df_enhanced[feat2] + 1e-8)  # Add small value to avoid division by zero
                )
        
        # 2. Statistical Interactions between important features
        print("🔬 Creating feature interactions...")
        # Focus on features with correlation > 0.1 with target
        high_corr_features = df_enhanced[numeric_cols].corrwith(df_enhanced[target_col]).abs()
        high_corr_features = high_corr_features[high_corr_features > 0.1].index.tolist()
        
        if len(high_corr_features) >= 2:
            # Multiplicative interactions (capture synergistic effects)
            for i, feat1 in enumerate(high_corr_features[:4]):
                for feat2 in high_corr_features[i+1:5]:
                    df_enhanced[f'{feat1}_x_{feat2}'] = df_enhanced[feat1] * df_enhanced[feat2]
            
            # Ratio interactions (capture relative relationships)
            for i, feat1 in enumerate(high_corr_features[:3]):
                for feat2 in high_corr_features[i+1:4]:
                    denominator = df_enhanced[feat2].replace(0, 1e-8)  # Avoid division by zero
                    df_enhanced[f'{feat1}_div_{feat2}'] = df_enhanced[feat1] / denominator
        
        # 3. Non-linear Transformations to capture complex relationships
        print("📈 Creating non-linear transformations...")
        for feat in high_corr_features[:5]:
            # Polynomial features (quadratic and cubic)
            df_enhanced[f'{feat}_squared'] = df_enhanced[feat] ** 2
            df_enhanced[f'{feat}_cubed'] = df_enhanced[feat] ** 3
            
            # Logarithmic transformations (handle negative values with abs)
            df_enhanced[f'{feat}_log1p'] = np.log1p(np.abs(df_enhanced[feat]))
            df_enhanced[f'{feat}_sqrt'] = np.sqrt(np.abs(df_enhanced[feat]))
        
        # 4. Statistical Aggregates across all features
        print("📊 Creating statistical aggregates...")
        current_numeric = df_enhanced.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in current_numeric if col != target_col]
        
        if len(feature_cols) > 0:
            # Create statistical summaries across all features for each patient
            df_enhanced['feature_mean'] = df_enhanced[feature_cols].mean(axis=1)
            df_enhanced['feature_std'] = df_enhanced[feature_cols].std(axis=1)
            df_enhanced['feature_max'] = df_enhanced[feature_cols].max(axis=1)
            df_enhanced['feature_min'] = df_enhanced[feature_cols].min(axis=1)
            df_enhanced['feature_range'] = df_enhanced['feature_max'] - df_enhanced['feature_min']
            df_enhanced['feature_median'] = df_enhanced[feature_cols].median(axis=1)
            df_enhanced['feature_q75'] = df_enhanced[feature_cols].quantile(0.75, axis=1)
            df_enhanced['feature_q25'] = df_enhanced[feature_cols].quantile(0.25, axis=1)
        
        # 5. Diabetes-specific medical features (domain knowledge)
        print("🩺 Creating diabetes-specific medical features...")
        if len(feature_cols) >= 5:
            # Glucose metabolism indicators (identify glucose-related features)
            glucose_like = [col for col in feature_cols if any(term in col.lower() 
                          for term in ['glucose', 'sugar', 'gly', 'hba1c', 'a1c'])]
            if glucose_like:
                df_enhanced['glucose_composite'] = df_enhanced[glucose_like].mean(axis=1)
            
            # Metabolic syndrome indicators (identify metabolic features)
            metabolic_like = [col for col in feature_cols if any(term in col.lower() 
                            for term in ['bmi', 'weight', 'cholesterol', 'triglyc', 'hdl', 'ldl'])]
            if metabolic_like:
                df_enhanced['metabolic_composite'] = df_enhanced[metabolic_like].mean(axis=1)
        
        # 6. Clean infinite and NaN values created during feature engineering
        print("🧹 Cleaning enhanced features...")
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)  # Replace infinite values
        
        # Fill NaN values with median (robust to outliers)
        numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
        df_enhanced[numeric_columns] = df_enhanced[numeric_columns].fillna(
            df_enhanced[numeric_columns].median()
        )
        
        # Report feature engineering results
        features_added = df_enhanced.shape[1] - df.shape[1]
        print(f"✨ Feature engineering complete!")
        print(f"📈 Features: {df.shape[1]} → {df_enhanced.shape[1]} (+{features_added} new features)")
        
        # Store label encoders for future use
        self.label_encoders = label_encoders
        return df_enhanced
    
    def create_base_models(self):
        """
        Create base models for stacking ensemble
        
        This method creates a diverse set of base learners including:
        - Linear models (Ridge, Lasso, ElasticNet)
        - Tree-based models (RandomForest, ExtraTrees, GradientBoosting)
        - Advanced gradient boosting (XGBoost, LightGBM, CatBoost if available)
        
        Returns:
            dict: Dictionary of model names and initialized model objects
        """
        print(f"\n🏗️ Creating base models for {self.dataset_name}")
        
        models = {}
        
        # Classical linear models for different regularization approaches
        models['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        models['lasso'] = Lasso(alpha=0.1, random_state=self.random_state)
        models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        
        # Tree-based ensemble models for non-linear relationships
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=self.random_state
        )
        
        # Advanced gradient boosting models (if libraries are available)
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=self.random_state, n_jobs=-1, verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=self.random_state, n_jobs=-1, verbosity=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostRegressor(
                iterations=100, learning_rate=0.1, depth=6,
                random_state=self.random_state, verbose=False
            )
        
        print(f"✅ Created {len(models)} base models: {list(models.keys())}")
        return models
    
    def optimize_hyperparameters(self, X, y, model_name, model):
        """
        Optimize hyperparameters using Optuna framework
        
        This method uses Bayesian optimization to find optimal hyperparameters
        for key models to improve performance before stacking.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name (str): Name of the model to optimize
            model: Model object to optimize
            
        Returns:
            tuple: (best_parameters, best_score)
        """
        print(f"🔧 Optimizing {model_name} hyperparameters...")
        
        def objective(trial):
            """Objective function for Optuna optimization"""
            try:
                # Define hyperparameter search spaces for different models
                if model_name == 'ridge':
                    alpha = trial.suggest_float('alpha', 0.01, 100, log=True)
                    temp_model = Ridge(alpha=alpha, random_state=self.random_state)
                
                elif model_name == 'random_forest':
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    max_depth = trial.suggest_int('max_depth', 3, 15)
                    temp_model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth,
                        random_state=self.random_state, n_jobs=-1
                    )
                
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    temp_model = xgb.XGBRegressor(
                        learning_rate=learning_rate, max_depth=max_depth,
                        n_estimators=n_estimators, random_state=self.random_state,
                        n_jobs=-1, verbosity=0
                    )
                
                elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    temp_model = lgb.LGBMRegressor(
                        learning_rate=learning_rate, max_depth=max_depth,
                        n_estimators=n_estimators, random_state=self.random_state,
                        n_jobs=-1, verbosity=-1
                    )
                
                else:
                    temp_model = model  # Use default if no optimization defined
                
                # Evaluate using cross-validation
                cv_scores = cross_val_score(temp_model, X, y, cv=5, 
                                          scoring='neg_mean_absolute_error', n_jobs=-1)
                return -cv_scores.mean()  # Minimize MAE (negative because optuna minimizes)
            
            except Exception as e:
                return float('inf')  # Return bad score if optimization fails
        
        # Run Bayesian optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"  ✅ Best MAE: {best_score:.4f}")
        print(f"  🎯 Best params: {best_params}")
        
        return best_params, best_score
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all base models with hyperparameter optimization
        
        This method trains each base model, optimizes key models, and evaluates
        their performance on the test set.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            
        Returns:
            tuple: (trained_models_dict, model_scores_dict)
        """
        print(f"\n🏋️ Training models for {self.dataset_name}")
        
        models = self.create_base_models()
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Optimize hyperparameters for key models
            if name in ['ridge', 'random_forest', 'xgboost', 'lightgbm']:
                try:
                    best_params, _ = self.optimize_hyperparameters(X_train, y_train, name, model)
                    
                    # Update model with optimized parameters
                    if name == 'ridge':
                        model = Ridge(alpha=best_params['alpha'], random_state=self.random_state)
                    elif name == 'random_forest':
                        model = RandomForestRegressor(
                            n_estimators=best_params['n_estimators'],
                            max_depth=best_params['max_depth'],
                            random_state=self.random_state, n_jobs=-1
                        )
                    elif name == 'xgboost' and XGBOOST_AVAILABLE:
                        model = xgb.XGBRegressor(
                            learning_rate=best_params['learning_rate'],
                            max_depth=best_params['max_depth'],
                            n_estimators=best_params['n_estimators'],
                            random_state=self.random_state, n_jobs=-1, verbosity=0
                        )
                    elif name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                        model = lgb.LGBMRegressor(
                            learning_rate=best_params['learning_rate'],
                            max_depth=best_params['max_depth'],
                            n_estimators=best_params['n_estimators'],
                            random_state=self.random_state, n_jobs=-1, verbosity=-1
                        )
                except Exception as e:
                    print(f"  ⚠️ Optimization failed for {name}, using defaults")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate model performance
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            trained_models[name] = model
            model_scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            print(f"  📈 {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Store in instance variables for later use
        self.models = trained_models
        self.model_scores = model_scores
        
        return trained_models, model_scores
    
    def create_stack_ridge_ensemble(self, X_train, X_test, y_train, y_test):
        """
        Create Stack Ridge ensemble with best base models
        
        This is the core stacking implementation where:
        1. Best performing base models are selected
        2. Ridge regression is used as meta-learner
        3. Cross-validation generates meta-features
        4. Final ensemble is trained and evaluated
        
        Args:
            X_train: Training features
            X_test: Testing features  
            y_train: Training targets
            y_test: Testing targets
            
        Returns:
            StackingRegressor: Trained stacking ensemble model
        """
        print(f"\n🏗️ Creating Stack Ridge Ensemble for {self.dataset_name}")
        
        # Select top 5 base models based on MAE performance
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1]['MAE'])
        best_model_names = [name for name, _ in sorted_models[:5]]
        
        print(f"🎯 Selected base models: {best_model_names}")
        
        # Create list of (name, model) tuples for StackingRegressor
        base_learners = [(name, self.models[name]) for name in best_model_names]
        
        # Create Ridge regression as meta-learner (final estimator)
        meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
        
        # Create the stacking ensemble
        stacking_regressor = StackingRegressor(
            estimators=base_learners,          # Base models
            final_estimator=meta_learner,      # Meta-learner
            cv=5,                              # 5-fold CV for meta-feature generation
            n_jobs=-1                          # Use all available cores
        )
        
        # Train the stacking ensemble
        print("🚀 Training Stack Ridge ensemble...")
        stacking_regressor.fit(X_train, y_train)
        
        # Evaluate ensemble performance
        y_pred = stacking_regressor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.stacked_model = stacking_regressor
        self.ensemble_scores = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        # Report performance
        print(f"✅ Stack Ridge Ensemble Performance:")
        print(f"   📈 MAE: {mae:.4f}")
        print(f"   📊 RMSE: {rmse:.4f}")
        print(f"   🎯 R²: {r2:.4f}")
        
        # Check clinical target achievement
        if mae < 0.5:
            print(f"   🎉 SUCCESS: Target MAE < 0.5 ACHIEVED!")
        else:
            improvement_needed = ((mae / 0.5 - 1) * 100)
            print(f"   📈 Need {improvement_needed:.1f}% improvement for clinical target")
        
        return stacking_regressor
    
    def analyze_feature_importance(self, X, y, feature_names):
        """
        Comprehensive feature importance analysis using multiple methods
        
        This method analyzes feature importance using:
        1. Ridge coefficients from meta-learner
        2. Feature importances from tree-based base models
        3. Permutation importance for the ensemble
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance results from different methods
        """
        print(f"\n🔍 Feature Importance Analysis for {self.dataset_name}")
        
        feature_importance = {}
        
        # 1. Ridge Coefficients from meta-learner
        if hasattr(self.stacked_model.final_estimator_, 'coef_'):
            ridge_importance = np.abs(self.stacked_model.final_estimator_.coef_)
            feature_importance['ridge_meta'] = dict(zip(
                [f'base_model_{i}' for i in range(len(ridge_importance))], 
                ridge_importance
            ))
        
        # 2. Base model feature importance analysis
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models have feature_importances_ attribute
                importance = model.feature_importances_
                feature_importance[f'{name}_importance'] = dict(zip(feature_names, importance))
                
                # Display top 10 features for this model
                top_indices = np.argsort(importance)[-10:][::-1]
                print(f"\n📊 Top 10 features for {name}:")
                for i, idx in enumerate(top_indices):
                    print(f"   {i+1:2d}. {feature_names[idx]}: {importance[idx]:.4f}")
            
            elif hasattr(model, 'coef_'):
                # Linear models have coefficients
                importance = np.abs(model.coef_)
                feature_importance[f'{name}_coef'] = dict(zip(feature_names, importance))
        
        # 3. Permutation importance for the ensemble (most reliable method)
        print(f"\n🔄 Computing permutation importance for Stack Ridge ensemble...")
        try:
            # Permutation importance measures decrease in model performance when feature is shuffled
            perm_importance = permutation_importance(
                self.stacked_model, X, y, n_repeats=10, random_state=self.random_state, n_jobs=-1
            )
            
            # Extract importance scores
            perm_scores = perm_importance.importances_mean
            feature_importance['permutation'] = dict(zip(feature_names, perm_scores))
            
            # Display top 15 most important features
            top_indices = np.argsort(perm_scores)[-15:][::-1]
            print(f"📈 Top 15 features by permutation importance:")
            for i, idx in enumerate(top_indices):
                print(f"   {i+1:2d}. {feature_names[idx]}: {perm_scores[idx]:.4f}")
            
            # Store best features for future use
            self.best_features = [feature_names[idx] for idx in top_indices]
            
        except Exception as e:
            print(f"⚠️ Permutation importance failed: {e}")
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def save_model_and_results(self, save_dir='models'):
        """
        Save trained model and results to disk for future use
        
        This method saves:
        1. The trained stacked model
        2. The scaler used for preprocessing
        3. Feature names and label encoders
        4. All evaluation results and metadata
        
        Args:
            save_dir (str): Directory to save files
            
        Returns:
            tuple: (model_filename, results_filename)
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define filenames
        model_filename = f"{save_dir}/stack_ridge_{self.dataset_name.lower()}.pkl"
        results_filename = f"{save_dir}/results_{self.dataset_name.lower()}.pkl"
        
        # Save model and preprocessing objects
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.stacked_model,
                'scaler': self.scaler,
                'feature_names': self.best_features,
                'label_encoders': getattr(self, 'label_encoders', {}),
                'target_stats': self.target_stats
            }, f)
        
        # Save comprehensive results
        results = {
            'dataset_name': self.dataset_name,
            'model_scores': self.model_scores,
            'ensemble_scores': self.ensemble_scores,
            'feature_importance': self.feature_importance,
            'best_features': self.best_features,
            'target_stats': self.target_stats,
            'original_shape': self.original_shape
        }
        
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Model saved: {model_filename}")
        print(f"💾 Results saved: {results_filename}")
        
        return model_filename, results_filename
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report with clinical assessment
        
        This method creates a detailed report including:
        - Dataset information
        - Model performance metrics
        - Clinical assessment against target MAE < 0.5
        - Feature importance insights
        - Recommendations for deployment
        """
        print(f"\n" + "="*80)
        print(f"📋 STACK RIDGE MODEL SUMMARY - {self.dataset_name.upper()}")
        print(f"="*80)
        
        # Dataset information section
        print(f"📊 Dataset Information:")
        print(f"   • Original shape: {self.original_shape}")
        print(f"   • Target statistics: {self.target_stats}")
        
        # Model performance section
        print(f"\n🏆 Model Performance:")
        print(f"   • Stack Ridge Ensemble MAE: {self.ensemble_scores['MAE']:.4f}")
        print(f"   • Stack Ridge Ensemble RMSE: {self.ensemble_scores['RMSE']:.4f}")
        print(f"   • Stack Ridge Ensemble R²: {self.ensemble_scores['R2']:.4f}")
        
        # Base model comparison
        print(f"\n📈 Base Model Performance:")
        for name, scores in sorted(self.model_scores.items(), key=lambda x: x[1]['MAE']):
            print(f"   • {name}: MAE={scores['MAE']:.4f}, R²={scores['R2']:.4f}")
        
        # Clinical assessment based on MAE performance
        print(f"\n🎯 Clinical Assessment:")
        mae = self.ensemble_scores['MAE']
        if mae < 0.5:
            status = "🎉 EXCELLENT - Clinical target achieved!"
            clinical_grade = "A+"
        elif mae < 1.0:
            status = "✅ GOOD - Clinically acceptable"
            clinical_grade = "B+"
        elif mae < 1.5:
            status = "⚠️ FAIR - Needs improvement"
            clinical_grade = "C"
        else:
            status = "❌ POOR - Significant improvement needed"
            clinical_grade = "D"
        
        print(f"   • Status: {status}")
        print(f"   • Clinical Grade: {clinical_grade}")
        print(f"   • Target: MAE < 0.5 for clinical accuracy")
        
        # Feature importance insights
        if len(self.best_features) > 0:
            print(f"\n🔍 Top 10 Most Important Features:")
            for i, feature in enumerate(self.best_features[:10]):
                print(f"   {i+1:2d}. {feature}")
        
        # Actionable recommendations
        print(f"\n💡 Recommendations:")
        if mae < 0.5:
            print(f"   • Model ready for clinical validation")
            print(f"   • Consider deployment for diabetes management")
        else:
            print(f"   • Collect more data for model improvement")
            print(f"   • Feature engineering optimization needed")
            print(f"   • Consider ensemble with other algorithms")
        
        print(f"="*80)


def run_stack_ridge_pipeline():
    """
    Run Stack Ridge pipeline for all diabetes datasets
    
    This is the main execution function that:
    1. Processes each of the 3 diabetes datasets
    2. Applies the complete ML pipeline to each
    3. Compares results across datasets
    4. Saves all models and results
    5. Provides final recommendations
    
    Returns:
        dict: Results for all processed datasets
    """
    
    # Dataset configurations with file paths and descriptions
    datasets = [
        {
            'name': 'nmbfinalDiabetes_4',
            'file': 'final_imputed_data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Primary diabetes dataset with comprehensive features'
        },
        {
            'name': 'nmbfinalnewDiabetes_3', 
            'file': 'final_imputed_data/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Secondary diabetes dataset with extended features'
        },
        {
            'name': 'PrePostFinal_3',
            'file': 'final_imputed_data/PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv',
            'description': 'Pre/Post intervention diabetes dataset'
        }
    ]
    
    all_results = {}  # Store results for all datasets
    
    print("🚀 Starting Stack Ridge Pipeline for All Datasets")
    print("=" * 80)
    
    # Process each dataset sequentially
    for i, dataset_config in enumerate(datasets, 1):
        print(f"\n🔥 PROCESSING DATASET {i}/3: {dataset_config['name']}")
        print(f"📝 Description: {dataset_config['description']}")
        print("-" * 60)
        
        try:
            # Validate file exists before processing
            if not os.path.exists(dataset_config['file']):
                print(f"❌ File not found: {dataset_config['file']}")
                continue
            
            # Initialize model for this dataset
            model = DiabetesStackRidgeModel(
                dataset_name=dataset_config['name'],
                random_state=42  # Ensure reproducible results
            )
            
            # Step 1: Load and preprocess data
            df = model.load_and_preprocess_data(dataset_config['file'])
            
            # Step 2: Advanced feature engineering
            df_enhanced = model.advanced_feature_engineering(df)
            
            # Step 3: Prepare features and target
            target_col = 'PostBLHBA1C'
            X = df_enhanced.drop(columns=[target_col])
            y = df_enhanced[target_col]
            
            # Step 4: Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Step 5: Scale features using robust scaler
            X_train_scaled = model.scaler.fit_transform(X_train)
            X_test_scaled = model.scaler.transform(X_test)
            
            # Convert back to DataFrame for easier handling
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
            
            # Step 6: Train and optimize base models
            trained_models, model_scores = model.train_and_evaluate_models(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # Step 7: Create Stack Ridge ensemble
            stacked_model = model.create_stack_ridge_ensemble(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # Step 8: Analyze feature importance
            feature_importance = model.analyze_feature_importance(
                X_test_scaled, y_test, X.columns.tolist()
            )
            
            # Step 9: Save model and results
            model_file, results_file = model.save_model_and_results()
            
            # Step 10: Generate comprehensive report
            model.generate_summary_report()
            
            # Store results for final comparison
            all_results[dataset_config['name']] = {
                'model': model,
                'scores': model.ensemble_scores,
                'feature_importance': feature_importance,
                'model_file': model_file,
                'results_file': results_file
            }
            
            print(f"✅ Dataset {i} processed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing dataset {i}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            continue
    
    # Generate final summary across all datasets
    print(f"\n" + "="*100)
    print(f"🏆 FINAL SUMMARY - ALL DATASETS COMPARISON")
    print(f"="*100)
    
    if all_results:
        # Find best performing dataset
        best_dataset = min(all_results.items(), key=lambda x: x[1]['scores']['MAE'])
        
        # Performance comparison table
        print(f"📊 Model Performance Comparison:")
        for name, result in all_results.items():
            mae = result['scores']['MAE']
            r2 = result['scores']['R2']
            status = "🎯 TARGET MET" if mae < 0.5 else "📈 NEEDS IMPROVEMENT"
            print(f"   • {name}: MAE={mae:.4f}, R²={r2:.4f} - {status}")
        
        # Highlight best performing model
        print(f"\n🥇 Best Performing Dataset: {best_dataset[0]}")
        print(f"   • MAE: {best_dataset[1]['scores']['MAE']:.4f}")
        print(f"   • R²: {best_dataset[1]['scores']['R2']:.4f}")
        
        # Final recommendations
        print(f"\n📁 All models saved in 'models/' directory")
        print(f"💡 Use the best performing model for clinical deployment")
        
    else:
        print("❌ No datasets processed successfully")
    
    # Pipeline completion
    print(f"="*100)
    print(f"🎉 Stack Ridge Pipeline Complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


# Main execution block
if __name__ == "__main__":
    # Run the complete Stack Ridge pipeline for all datasets
    results = run_stack_ridge_pipeline()

# ========================================
# PyCaret Optimize Function Exploration
# ========================================

"""
PyCaret's optimize() function for stacking models:

The optimize() function in PyCaret is used to tune hyperparameters of models,
including stacked ensembles. For stacking specifically:

Key Features:
1. **Automated Hyperparameter Tuning**: Uses techniques like Random Search, 
   Grid Search, or Bayesian Optimization (TPE, GP)

2. **Stacking Support**: Can optimize the meta-learner and base models in 
   stacked ensembles

3. **Cross-Validation**: Uses robust CV strategies to prevent overfitting
   during optimization

Example Usage for Stacking:

```python
from pycaret.regression import *

# Setup environment
setup(data=diabetes_data, target='PostBLHBA1C', session_id=123)

# Create base models
rf = create_model('rf')
xgb = create_model('xgboost') 
lgb = create_model('lightgbm')

# Create stacked ensemble
stacker = stack_models([rf, xgb, lgb], meta_model='ridge')

# Optimize the stacked model
optimized_stacker = optimize_model(stacker, 
                                  n_iter=50,           # Number of iterations
                                  optimize='MAE',       # Metric to optimize
                                  search_library='optuna', # Optimization library
                                  search_algorithm='tpe')  # Algorithm type

# Advanced optimization with custom search space
custom_grid = {
    'meta_model__alpha': [0.01, 0.1, 1.0, 10.0],  # Ridge alpha values
    'cv': [3, 5, 10]                               # CV folds for stacking
}

optimized_custom = optimize_model(stacker,
                                 custom_grid=custom_grid,
                                 search_library='scikit-learn')