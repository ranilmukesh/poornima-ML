#!/usr/bin/env python3
"""
Enhanced PyCaret AutoML Stack Ridge Model for HbA1c Prediction
=============================================================
Target: PostBLHBA1C prediction with feature importance analysis
Datasets: All 3 diabetes datasets (nmbfinalDiabetes, nmbfinalnewDiabetes, PrePostFinal)
Approach: Enhanced PyCaret AutoML with optimized stack_models() for Ridge ensemble

Clinical Goal: Achieve MAE < 0.5 for accurate diabetes management
Improvements: Better preprocessing, more features, enhanced evaluation
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

print("🚀 Enhanced PyCaret AutoML Pipeline for HbA1c Prediction")
print("=" * 60)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Target: PostBLHBA1C prediction with MAE < 0.5")
print("📊 Datasets: 3 diabetes datasets with individual models")
print("🏗️ Architecture: Enhanced AutoML with Optimized Stacked Ridge Ensemble")
print("💡 Improvements: Better preprocessing, more features, enhanced evaluation")
print("=" * 60)

class EnhancedDiabetesAutoMLModel:
    """Enhanced AutoML Model for Diabetes HbA1c Prediction using PyCaret"""

    def __init__(self, dataset_name, random_state=42):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.final_model = None
        self.stacked_model = None  # Keep reference to non-finalized model
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
        cleaned_columns = {}
        for col in df.columns:
            cleaned_col = col.replace(' ', '_')
            cleaned_col = cleaned_col.replace('-', '_')
            cleaned_col = cleaned_col.replace('(', '_')
            cleaned_col = cleaned_col.replace(')', '_')
            cleaned_col = cleaned_col.replace('/', '_')
            cleaned_col = cleaned_col.replace('\\', '_')
            while '__' in cleaned_col:
                cleaned_col = cleaned_col.replace('__', '_')
            cleaned_col = cleaned_col.strip('_')
            cleaned_columns[col] = cleaned_col
        
        print(f"🧹 Cleaned {len(cleaned_columns)} column names for PyCaret compatibility")
        return df.rename(columns=cleaned_columns)

    def enhanced_data_preprocessing(self, df):
        """Enhanced data preprocessing for better model performance"""
        target_col = 'PostBLHBA1C'
        
        print(f"📊 Original shape: {df.shape}")
        initial_rows = len(df)
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Enhanced outlier detection (less aggressive)
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        z_scores = np.abs((df[target_col] - target_mean) / target_std)
        df = df[z_scores <= 3.5]  # Less aggressive outlier removal
        
        # Remove constant/near-constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_cols = []
        for col in numeric_cols:
            if col != target_col and df[col].std() < 0.01:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            df = df.drop(columns=low_variance_cols)
            print(f"🗑️ Removed {len(low_variance_cols)} low-variance features")
        
        outliers_removed = initial_rows - len(df)
        print(f"🧹 Enhanced cleaned shape: {df.shape} (removed {outliers_removed} outliers)")
        print(f"📈 HbA1c range: {df[target_col].min():.2f} - {df[target_col].max():.2f}")

        self.original_shape = df.shape
        self.target_stats = {
            'mean': df[target_col].mean(),
            'std': df[target_col].std(),
            'min': df[target_col].min(),
            'max': df[target_col].max()
        }
        return df

    def run_enhanced_automl_pipeline(self, df):
        """Run the enhanced PyCaret AutoML pipeline"""
        print(f"\\n🤖 Running Enhanced PyCaret AutoML for {self.dataset_name}")
        target_col = 'PostBLHBA1C'

        # 1. Enhanced Setup PyCaret environment
        print("🔧 Setting up Enhanced PyCaret environment...")
        self.pycaret_setup = pycaret_reg.setup(
            data=df,
            target=target_col,
            session_id=self.random_state,
            train_size=0.85,  # Use more data for training
            
            # Enhanced preprocessing configuration
            normalize=True,
            normalize_method='robust',
            transformation=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,  # Less aggressive
            
            # Enhanced feature selection
            feature_selection=True,
            n_features_to_select=0.6,  # Keep 60% of features
            feature_selection_method='univariate',
            
            # Handle missing values
            imputation_type='simple',
            numeric_imputation='mean',
            categorical_imputation='mode',
            
            # Cross-validation
            fold_strategy='kfold',
            fold=10  # More folds for better validation
        )

        # 2. Enhanced model comparison
        print("🔍 Enhanced model comparison...")
        
        # Include all available models
        models_to_compare = ['lr', 'ridge', 'lasso', 'rf', 'et', 'gbr', 'ada', 'dt']
        
        if XGBOOST_AVAILABLE:
            models_to_compare.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models_to_compare.append('lightgbm')
        if CATBOOST_AVAILABLE:
            models_to_compare.append('catboost')
        
        print(f"   📋 Models to compare: {models_to_compare}")
        
        try:
            best_models = compare_models(
                include=models_to_compare,
                sort='MAE',
                n_select=5,  # Select top 5 models
                fold=10
            )
            
            if not isinstance(best_models, list):
                best_models = [best_models]
            
            print(f"🏆 Selected {len(best_models)} top models for enhanced stacking")
            
        except Exception as e:
            print(f"⚠️ Model comparison failed, using fallback models: {str(e)}")
            best_models = [create_model('ridge'), create_model('rf'), create_model('et')]
            print("🔄 Using Ridge, RF, ET as fallback models")

        # 3. Enhanced tuning with more iterations
        print("⚡ Enhanced hyperparameter tuning...")
        tuned_models = []
        for i, model in enumerate(best_models):
            print(f"   🔧 Tuning model {i+1}/{len(best_models)}...")
            try:
                tuned_model = tune_model(
                    model, 
                    optimize='MAE',
                    n_iter=20,  # More tuning iterations
                    fold=10
                )
                tuned_models.append(tuned_model)
            except Exception as e:
                print(f"   ⚠️ Tuning failed for model {i+1}, using original: {str(e)}")
                tuned_models.append(model)

        # 4. Enhanced Stacked Ensemble
        print("🏗️ Creating Enhanced Stacked Ridge Ensemble...")
        try:
            self.stacked_model = stack_models(
                estimator_list=tuned_models,
                meta_model=create_model('ridge'),
                optimize='MAE',
                fold=10
            )
            print("✅ Enhanced stacking successful")
        except Exception as e:
            print(f"⚠️ Stacking failed, using best single model: {str(e)}")
            self.stacked_model = tuned_models[0]

        # 5. Finalize the model
        print("✅ Finalizing the enhanced model...")
        self.final_model = finalize_model(self.stacked_model)

        # 6. Enhanced evaluation
        print("\\n📊 Enhanced model evaluation...")
        
        try:
            # Use cross-validation scores from the stacked model
            if hasattr(self.stacked_model, 'score_'):
                # Get scores from cross-validation
                cv_results = pull()  # Get the last results
                if len(cv_results) > 0:
                    mean_row = cv_results.loc['Mean']
                    mae = float(mean_row['MAE'])
                    rmse = float(mean_row['RMSE'])
                    r2 = float(mean_row['R2'])
                else:
                    raise ValueError("No CV results available")
            else:
                raise ValueError("No model scores available")
            
            self.ensemble_scores = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            print(f"   📈 Enhanced MAE: {mae:.4f}")
            print(f"   📊 Enhanced RMSE: {rmse:.4f}")
            print(f"   🎯 Enhanced R²: {r2:.4f}")
            
            # Enhanced clinical assessment
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
                improvement_needed = ((mae / 0.5 - 1) * 100)
                print(f"   📈 NEEDS IMPROVEMENT: {improvement_needed:.1f}% improvement needed")
                clinical_status = "POOR"
            
            self.clinical_status = clinical_status
            
        except Exception as e:
            print(f"   ❌ Enhanced evaluation failed: {str(e)}")
            # Set conservative default values
            self.ensemble_scores = {'MAE': 999.0, 'RMSE': 999.0, 'R2': 0.0}
            self.clinical_status = "ERROR"

        return self.final_model

    def enhanced_feature_importance_analysis(self):
        """Enhanced feature importance analysis"""
        print(f"\\n🔍 Enhanced Feature Importance Analysis for {self.dataset_name}")

        try:
            # Method 1: Try PyCaret's interpret_model
            try:
                interpret_model(self.stacked_model, plot='feature', save=True)
                print("📈 Feature importance plot generated")
            except:
                print("   ⚠️ Feature importance plot generation failed")
            
            # Method 2: Use model-specific feature importance
            if hasattr(self.stacked_model, 'feature_importances_'):
                importances = self.stacked_model.feature_importances_
                feature_names = get_config('X_train').columns
                
                fi_df = pd.DataFrame({
                    'Feature': feature_names, 
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)
                
                print("📈 Top 20 Enhanced Features:")
                for idx, row in fi_df.iterrows():
                    print(f"    {idx+1:2d}. {row['Feature']}: {row['Importance']:.4f}")
                
                self.best_features = fi_df['Feature'].tolist()
                self.feature_importance = fi_df.set_index('Feature').to_dict()['Importance']
                
            else:
                # Method 3: Permutation importance
                print("   🔄 Using enhanced permutation importance...")
                
                from sklearn.inspection import permutation_importance
                from sklearn.metrics import make_scorer, mean_absolute_error
                
                X_test = get_config('X_test')
                y_test = get_config('y_test')
                
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                perm_importance = permutation_importance(
                    self.stacked_model, X_test, y_test,
                    scoring=scorer,
                    n_repeats=5,  # More repeats for stability
                    random_state=self.random_state
                )
                
                fi_df = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': np.abs(perm_importance.importances_mean),
                    'Std': perm_importance.importances_std
                }).sort_values('Importance', ascending=False).head(20)
                
                print("🔬 Top 20 Enhanced Features by Permutation Importance:")
                for idx, row in fi_df.iterrows():
                    print(f"    {idx+1:2d}. {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})")
                
                self.best_features = fi_df['Feature'].tolist()
                self.feature_importance = fi_df.set_index('Feature').to_dict()['Importance']
            
        except Exception as e:
            print(f"⚠️ Enhanced feature importance analysis failed: {str(e)}")
            try:
                feature_names = get_config('X_train').columns[:20].tolist()
                self.best_features = feature_names
                self.feature_importance = {name: 0.1 for name in feature_names}
                print(f"   🔄 Using fallback feature list: {len(self.best_features)} features")
            except:
                self.best_features = []
                self.feature_importance = {}

    def save_enhanced_results(self, save_dir='models'):
        """Save enhanced model and results"""
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{save_dir}/enhanced_automl_{self.dataset_name.lower()}_{timestamp}"
        results_filename = f"{save_dir}/enhanced_results_{self.dataset_name.lower()}_{timestamp}.pkl"

        # Save the PyCaret pipeline
        save_model(self.final_model, model_filename)

        results = {
            'dataset_name': self.dataset_name,
            'ensemble_scores': self.ensemble_scores,
            'feature_importance': self.feature_importance,
            'best_features': self.best_features,
            'target_stats': self.target_stats,
            'original_shape': self.original_shape,
            'clinical_status': self.clinical_status,
            'timestamp': timestamp
        }

        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"\\n💾 Enhanced model saved: {model_filename}.pkl")
        print(f"💾 Enhanced results saved: {results_filename}")

        return f"{model_filename}.pkl", results_filename

    def generate_enhanced_summary(self):
        """Generate enhanced summary report"""
        print(f"\\n" + "="*80)
        print(f"📋 ENHANCED PYCARET AUTOML SUMMARY - {self.dataset_name.upper()}")
        print(f"="*80)

        print(f"📊 Dataset Information:")
        print(f"    • Original shape: {self.original_shape}")
        print(f"    • Target range: {self.target_stats.get('min', 'N/A'):.2f} - {self.target_stats.get('max', 'N/A'):.2f}")
        print(f"    • Target mean: {self.target_stats.get('mean', 'N/A'):.2f} (±{self.target_stats.get('std', 'N/A'):.2f})")

        print(f"\\n🏆 Enhanced Model Performance:")
        mae = self.ensemble_scores.get('MAE', 999.0)
        rmse = self.ensemble_scores.get('RMSE', 999.0)
        r2 = self.ensemble_scores.get('R2', 0.0)
        
        print(f"    • Enhanced MAE: {mae:.4f}")
        print(f"    • Enhanced RMSE: {rmse:.4f}")
        print(f"    • Enhanced R²: {r2:.4f}")

        print(f"\\n🎯 Clinical Assessment:")
        
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
            print(f"\\n🔍 Top 15 Most Important Enhanced Features:")
            for i, feature in enumerate(self.best_features[:15]):
                importance = self.feature_importance.get(feature, 0.0)
                print(f"    {i+1:2d}. {feature}: {importance:.4f}")

        print(f"\\n💡 Enhanced Recommendations:")
        if mae < 0.5:
            print("    • Model ready for clinical validation")
            print("    • Conduct prospective validation study")
            print("    • Prepare deployment pipeline")
        elif mae < 1.0:
            print("    • Consider neural network meta-learner")
            print("    • Implement feature interaction terms")
            print("    • Validate on external datasets")
        else:
            print("    • Increase dataset size significantly")
            print("    • Implement domain-specific feature engineering")
            print("    • Consider time-series modeling approach")
            
        print(f"="*80)

def run_enhanced_pipeline():
    """Run the enhanced AutoML pipeline"""
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
        print(f"\\n🔥 PROCESSING ENHANCED DATASET {i}/{len(datasets)}: {config['name']}")
        print(f"📝 {config['description']}")
        print("-" * 60)

        try:
            if not os.path.exists(config['file']):
                print(f"❌ File not found: {config['file']}")
                continue

            model_runner = EnhancedDiabetesAutoMLModel(dataset_name=config['name'])
            
            # Load and enhanced preprocessing
            df = pd.read_csv(config['file'])
            df = model_runner.clean_column_names(df)
            df = model_runner.enhanced_data_preprocessing(df)

            # Run enhanced pipeline
            final_model = model_runner.run_enhanced_automl_pipeline(df)
            model_runner.enhanced_feature_importance_analysis()
            model_file, results_file = model_runner.save_enhanced_results()
            model_runner.generate_enhanced_summary()

            all_results[config['name']] = {
                'model': model_runner,
                'scores': model_runner.ensemble_scores,
                'model_file': model_file,
                'clinical_grade': model_runner.clinical_status
            }

        except Exception as e:
            print(f"❌ Error processing enhanced {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\\n" + "="*100)
    print("🏆 ENHANCED FINAL SUMMARY - ALL DATASETS")
    print("="*100)

    if all_results:
        best_dataset = min(all_results.items(), key=lambda x: x[1]['scores']['MAE'])
        
        print(f"📊 Enhanced Performance Comparison:")
        for name, result in all_results.items():
            mae = result['scores']['MAE']
            r2 = result['scores']['R2']
            grade = result['clinical_grade']
            status = "🎯 TARGET MET" if mae < 0.5 else "📈 NEEDS IMPROVEMENT"
            print(f"   • {name}: MAE={mae:.4f}, R²={r2:.4f}, Grade={grade} - {status}")
        
        print(f"\\n🥇 Best Enhanced Model: {best_dataset[0]}")
        print(f"   • MAE: {best_dataset[1]['scores']['MAE']:.4f}")
        print(f"   • R²: {best_dataset[1]['scores']['R2']:.4f}")
        print(f"   • Clinical Grade: {best_dataset[1]['clinical_grade']}")
    else:
        print("No datasets processed successfully.")

    print(f"\\n💡 ENHANCED PERFORMANCE IMPROVEMENT STRATEGIES:")
    print(f"="*60)
    print(f"🔬 Data Enhancement:")
    print(f"   • Collect longitudinal data (time-series modeling)")
    print(f"   • Include genetic markers and biomarkers")
    print(f"   • Add lifestyle and dietary detailed features")
    print(f"\\n🤖 Model Enhancement:")
    print(f"   • Neural network meta-learners")
    print(f"   • Bayesian optimization for hyperparameters")
    print(f"   • Ensemble of diverse model types")
    print(f"\\n🏥 Clinical Enhancement:")
    print(f"   • Multi-center validation")
    print(f"   • Prospective clinical trial")
    print(f"   • Integration with clinical decision systems")
    print(f"="*60)

    print(f"🎉 Enhanced AutoML Pipeline Complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_enhanced_pipeline()