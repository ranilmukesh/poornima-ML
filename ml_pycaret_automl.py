#!/usr/bin/env python3
"""
PyCaret AutoML Stack Ridge Model for HbA1c Prediction
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
warnings.filterwarnings('ignore')

# PyCaret AutoML imports
from pycaret.regression import *

def load_and_validate_data(file_path, dataset_name):
    """Load and validate dataset for AutoML processing"""
    print(f"📂 Loading {dataset_name} from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"   ✅ Shape: {df.shape}")
        
        # Check for target column
        if 'PostBLHBA1C' not in df.columns:
            print(f"   ❌ Target column 'PostBLHBA1C' not found")
            return None
        
        # Basic data validation
        print(f"   📊 Target statistics:")
        print(f"      • Mean: {df['PostBLHBA1C'].mean():.2f}")
        print(f"      • Std: {df['PostBLHBA1C'].std():.2f}")
        print(f"      • Range: [{df['PostBLHBA1C'].min():.2f}, {df['PostBLHBA1C'].max():.2f}]")
        print(f"      • Missing values: {df['PostBLHBA1C'].isnull().sum()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
        return None

def run_pycaret_automl_pipeline(df, dataset_name):
    """Run PyCaret AutoML pipeline with Stack Ridge ensemble"""
    
    print(f"\n🤖 Starting PyCaret AutoML for {dataset_name}")
    print("-" * 60)
    
    try:
        # =================================================================
        # 1. SETUP AUTOML ENVIRONMENT
        # =================================================================
        print("🔧 Setting up AutoML environment...")
        
        # Setup PyCaret regression environment
        reg_setup = setup(
            data=df,
            target='PostBLHBA1C',
            session_id=42,
            train_size=0.8,
            silent=True,
            use_gpu=False,  # Set to True if GPU available
            preprocess=True,
            imputation_type='simple',
            numeric_imputation='mean',
            categorical_imputation='mode',
            transformation=True,
            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            remove_outliers=True,
            outliers_threshold=0.05,
            fold_strategy='kfold',
            fold=5
        )
        
        print("   ✅ AutoML environment configured")
        print(f"   📊 Training set size: {int(len(df) * 0.8)}")
        print(f"   📊 Test set size: {int(len(df) * 0.2)}")
        
        # =================================================================
        # 2. COMPARE MODELS - AUTOML MODEL SELECTION
        # =================================================================
        print("\n🏁 Comparing models with AutoML...")
        
        # Compare multiple models automatically
        best_models = compare_models(
            include=['ridge', 'rf', 'et', 'gbr', 'lightgbm', 'xgboost', 'catboost'],
            fold=5,
            sort='MAE',  # Primary metric for clinical accuracy
            n_select=3,  # Select top 3 models for stacking
            silent=True
        )
        
        print("   ✅ Model comparison completed")
        print(f"   🎯 Top models selected based on MAE performance")
        
        # =================================================================
        # 3. HYPERPARAMETER TUNING - AUTOML OPTIMIZATION
        # =================================================================
        print("\n⚡ Hyperparameter tuning with AutoML...")
        
        # Tune the best models automatically
        tuned_models = []
        for i, model in enumerate(best_models):
            print(f"   🔧 Tuning model {i+1}/3...")
            
            tuned_model = tune_model(
                model,
                optimize='MAE',
                search_library='optuna',
                n_iter=20,
                fold=5,
                silent=True
            )
            tuned_models.append(tuned_model)
        
        print("   ✅ Hyperparameter tuning completed")
        
        # =================================================================
        # 4. STACK MODELS - AUTOML ENSEMBLE WITH RIDGE
        # =================================================================
        print("\n🚀 Creating Stack Ridge ensemble with AutoML...")
        
        # Create stacked ensemble with Ridge as meta-learner
        # This is the core AutoML stacking as per PyCaret documentation
        stacked_ridge = stack_models(
            estimator_list=tuned_models,
            meta_model='ridge',  # Ridge as meta-learner for ensemble
            fold=5,
            method='auto',  # Automatic stacking method selection
            restack=True,  # Enable restacking for better performance
            silent=True
        )
        
        print("   ✅ Stack Ridge ensemble created")
        print("   🎯 Meta-learner: Ridge Regression")
        print(f"   📊 Base models: {len(tuned_models)} tuned models")
        
        # =================================================================
        # 5. MODEL EVALUATION - AUTOML PERFORMANCE ANALYSIS
        # =================================================================
        print("\n📊 Evaluating Stack Ridge model...")
        
        # Evaluate the stacked model
        stacked_results = evaluate_model(stacked_ridge, silent=True)
        
        # Get predictions for detailed analysis
        predictions = predict_model(stacked_ridge, silent=True)
        
        # Calculate clinical metrics
        y_true = predictions['PostBLHBA1C']
        y_pred = predictions['prediction_label']
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        print(f"   📈 Performance Metrics:")
        print(f"      • MAE: {mae:.4f}")
        print(f"      • RMSE: {rmse:.4f}")
        print(f"      • R²: {r2:.4f}")
        
        # Clinical assessment
        if mae < 0.5:
            status = "🎯 EXCELLENT - Target achieved!"
            grade = "A"
        elif mae < 0.75:
            status = "✅ GOOD - Clinical acceptable"
            grade = "B"
        elif mae < 1.0:
            status = "⚠️ FAIR - Needs improvement"
            grade = "C"
        else:
            status = "❌ POOR - Significant improvement needed"
            grade = "D"
        
        print(f"   🏥 Clinical Assessment:")
        print(f"      • Status: {status}")
        print(f"      • Grade: {grade}")
        print(f"      • Target: MAE < 0.5")
        
        # =================================================================
        # 6. FEATURE IMPORTANCE - AUTOML INTERPRETABILITY
        # =================================================================
        print("\n🔍 Analyzing feature importance...")
        
        try:
            # Get feature importance from the stacked model
            feature_importance_df = None
            
            # Try to get feature importance using PyCaret's interpret_model
            try:
                feature_importance_plot = interpret_model(stacked_ridge, plot='feature', save=True, silent=True)
                print("   ✅ Feature importance analysis completed")
            except:
                print("   ⚠️ Feature importance plot not available for this model type")
            
            # Alternative: Use permutation importance for any model
            try:
                # Get feature names from setup
                feature_names = get_config('X_train').columns.tolist()
                
                # Calculate permutation importance manually if needed
                from sklearn.inspection import permutation_importance
                from sklearn.metrics import mean_absolute_error
                
                # Get test data
                X_test = get_config('X_test')
                y_test = get_config('y_test')
                
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    stacked_ridge, X_test, y_test,
                    scoring='neg_mean_absolute_error',
                    n_repeats=5,
                    random_state=42
                )
                
                # Create feature importance DataFrame
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': perm_importance.importances_mean,
                    'std': perm_importance.importances_std
                }).sort_values('importance', ascending=False)
                
                print(f"   🔬 Top 10 Most Important Features:")
                for i, row in feature_importance_df.head(10).iterrows():
                    print(f"      {i+1:2d}. {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")
                
            except Exception as e:
                print(f"   ⚠️ Feature importance calculation failed: {str(e)}")
                feature_importance_df = None
        
        except Exception as e:
            print(f"   ⚠️ Feature importance analysis failed: {str(e)}")
            feature_importance_df = None
        
        # =================================================================
        # 7. MODEL SAVING - AUTOML DEPLOYMENT PREPARATION
        # =================================================================
        print("\n💾 Saving model and results...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save the final stacked model
        model_filename = f"models/stack_ridge_automl_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        finalize_model(stacked_ridge)
        
        # Save model using PyCaret
        save_model(stacked_ridge, model_filename.replace('.pkl', ''), silent=True)
        
        print(f"   ✅ Model saved: {model_filename}")
        
        # Save results
        results = {
            'dataset_name': dataset_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'clinical_grade': grade,
            'feature_importance': feature_importance_df.to_dict() if feature_importance_df is not None else None,
            'model_file': model_filename,
            'timestamp': datetime.now().isoformat()
        }
        
        results_filename = f"models/results_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   ✅ Results saved: {results_filename}")
        
        return {
            'model': stacked_ridge,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2},
            'clinical_grade': grade,
            'feature_importance': feature_importance_df,
            'model_file': model_filename,
            'results_file': results_filename
        }
        
    except Exception as e:
        print(f"   ❌ AutoML pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run AutoML Stack Ridge for all datasets"""
    
    print("🚀 PyCaret AutoML Stack Ridge Pipeline")
    print("=" * 80)
    print("🎯 Objective: Predict PostBLHBA1C with MAE < 0.5")
    print("🤖 Method: AutoML with Stack Ridge ensemble")
    print("📅 Started:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)
    
    # Dataset configurations
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
    
    all_results = {}
    
    # Process each dataset with AutoML
    for i, dataset_config in enumerate(datasets, 1):
        print(f"\n🔥 PROCESSING DATASET {i}/3: {dataset_config['name']}")
        print(f"📝 Description: {dataset_config['description']}")
        print("=" * 60)
        
        # Load and validate data
        df = load_and_validate_data(dataset_config['file'], dataset_config['name'])
        
        if df is None:
            print(f"❌ Skipping dataset {i} due to loading issues")
            continue
        
        # Run AutoML pipeline
        result = run_pycaret_automl_pipeline(df, dataset_config['name'])
        
        if result is not None:
            all_results[dataset_config['name']] = result
            print(f"✅ Dataset {i} processed successfully!")
            print(f"   📊 MAE: {result['metrics']['MAE']:.4f}")
            print(f"   🏥 Grade: {result['clinical_grade']}")
        else:
            print(f"❌ Dataset {i} processing failed")
    
    # =================================================================
    # FINAL SUMMARY - AUTOML COMPARISON ACROSS ALL DATASETS
    # =================================================================
    print(f"\n" + "="*100)
    print(f"🏆 FINAL AUTOML SUMMARY - ALL DATASETS COMPARISON")
    print(f"="*100)
    
    if all_results:
        # Find best performing dataset
        best_dataset = min(all_results.items(), key=lambda x: x[1]['metrics']['MAE'])
        
        print(f"📊 AutoML Model Performance Comparison:")
        for name, result in all_results.items():
            mae = result['metrics']['MAE']
            r2 = result['metrics']['R2']
            grade = result['clinical_grade']
            status = "🎯 TARGET MET" if mae < 0.5 else "📈 NEEDS IMPROVEMENT"
            print(f"   • {name}: MAE={mae:.4f}, R²={r2:.4f}, Grade={grade} - {status}")
        
        print(f"\n🥇 Best AutoML Model: {best_dataset[0]}")
        print(f"   • MAE: {best_dataset[1]['metrics']['MAE']:.4f}")
        print(f"   • R²: {best_dataset[1]['metrics']['R2']:.4f}")
        print(f"   • Clinical Grade: {best_dataset[1]['clinical_grade']}")
        
        # Clinical recommendations
        print(f"\n💡 Clinical Recommendations:")
        best_mae = best_dataset[1]['metrics']['MAE']
        if best_mae < 0.5:
            print(f"   ✅ AutoML model ready for clinical validation")
            print(f"   🏥 Consider deployment for diabetes management")
            print(f"   📋 Conduct prospective validation study")
        else:
            print(f"   📈 Model performance needs improvement")
            print(f"   🔬 Consider additional feature engineering")
            print(f"   📊 Collect more high-quality training data")
        
        print(f"\n📁 All AutoML models saved in 'models/' directory")
        print(f"🤖 PyCaret AutoML approach completed successfully")
        
    else:
        print("❌ No datasets processed successfully with AutoML")
    
    print(f"="*100)
    print(f"🎉 PyCaret AutoML Stack Ridge Pipeline Complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    # Run the complete AutoML pipeline
    results = main()