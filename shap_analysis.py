#!/usr/bin/env python3
"""
SHAP Analysis for H2O AutoML Models
====================================
Phase 4: Comprehensive SHAP explainability for all trained models

Features:
- SHAP value extraction from H2O models (pred_contribs=True)
- 5 visualization types (summary, beeswarm, dependence, waterfall, force)
- CSV export for clinical reporting
- Patient-level explanations for BleuLink integration

Reference:
- SHAP Library: https://shap.readthedocs.io/en/latest/
- H2O SHAP: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/explain.html
- TreeSHAP Paper: https://arxiv.org/abs/1905.04610

Author: AI Assistant
Date: 2025-10-28
Version: 1.0 - Production Ready
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import logging

import h2o
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP library
try:
    import shap
    shap.initjs()  # Initialize JavaScript for force plots
except ImportError:
    print("❌ ERROR: SHAP library not installed")
    print("Install with: pip install shap")
    sys.exit(1)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shap_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset configurations
DATASETS = {
    'nmbfinalDiabetes': 'final_imputed_data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed_final_imputed.csv',
    'nmbfinalnewDiabetes': 'final_imputed_data/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed_final_imputed.csv',
    'PrePostFinal': 'final_imputed_data/PrePostFinal (3)_selected_columns_cleaned_processed_final_imputed.csv',
}

# Fallback to temp_processed if final_imputed not found
DATASETS_FALLBACK = {
    'nmbfinalDiabetes': 'temp_processed/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv',
    'nmbfinalnewDiabetes': 'temp_processed/nmbfinalnewDiabetes (3)_selected_columns_cleaned_processed.csv',
    'PrePostFinal': 'temp_processed/PrePostFinal (3)_selected_columns_cleaned_processed.csv',
}

# Model directories (from Phase 3 training)
MODEL_BASE_DIR = "./models"

# Output directories
OUTPUT_DIR = "./shap_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

# Target variable
TARGET = 'PostBLHBA1C'

# Visualization settings
PLOT_DPI = 300
MAX_DISPLAY_FEATURES = 20

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    for directory in [OUTPUT_DIR, PLOTS_DIR, CSV_DIR, HTML_DIR]:
        os.makedirs(directory, exist_ok=True)
    logger.info(f"✓ Output directories created: {OUTPUT_DIR}")

def find_dataset_file(dataset_name):
    """Find dataset file with fallback logic."""
    primary_path = DATASETS.get(dataset_name)
    fallback_path = DATASETS_FALLBACK.get(dataset_name)
    
    if primary_path and os.path.exists(primary_path):
        logger.info(f"  Found: {primary_path}")
        return primary_path
    elif fallback_path and os.path.exists(fallback_path):
        logger.info(f"  Found (fallback): {fallback_path}")
        return fallback_path
    else:
        logger.error(f"  ❌ Dataset not found: {dataset_name}")
        return None

def load_h2o_model(model_path):
    """Load H2O model from disk."""
    try:
        model = h2o.load_model(model_path)
        logger.info(f"  ✓ Loaded model: {model.model_id}")
        return model
    except Exception as e:
        logger.error(f"  ❌ Failed to load model: {e}")
        return None

def extract_shap_values_from_h2o(model, data_h2o, feature_names):
    """
    Extract SHAP values from H2O model using pred_contribs=True.
    
    Reference: H2O-3 documentation - Prediction Contributions
    https://docs.h2o.ai/h2o/latest-stable/h2o-docs/explain.html
    
    Args:
        model: Trained H2O model
        data_h2o: H2OFrame with test data
        feature_names: List of feature column names
        
    Returns:
        tuple: (shap_values, bias_term, test_X_df)
    """
    logger.info("  Extracting SHAP contributions from H2O model...")
    
    try:
        # Get SHAP contributions (H2O native method)
        # Returns: [feature1_shap, feature2_shap, ..., BiasTerm]
        shap_h2o = model.predict(data_h2o, pred_contribs=True)
        
        # Convert to pandas DataFrame
        shap_df = shap_h2o.as_data_frame()
        
        # Separate SHAP values from bias term
        if 'BiasTerm' in shap_df.columns:
            bias_term = shap_df['BiasTerm'].values
            shap_values = shap_df.drop(columns=['BiasTerm']).values
            shap_feature_names = shap_df.drop(columns=['BiasTerm']).columns.tolist()
        else:
            # Some H2O versions use different naming
            logger.warning("  ⚠️ 'BiasTerm' not found, using last column as bias")
            bias_term = shap_df.iloc[:, -1].values
            shap_values = shap_df.iloc[:, :-1].values
            shap_feature_names = shap_df.columns[:-1].tolist()
        
        # Get original feature values (for visualization)
        test_X_h2o = data_h2o.drop(TARGET) if TARGET in data_h2o.columns else data_h2o
        test_X_df = test_X_h2o.as_data_frame()
        
        # Ensure feature alignment
        test_X_df = test_X_df[shap_feature_names]
        
        logger.info(f"  ✓ SHAP values extracted: {shap_values.shape[0]} samples × {shap_values.shape[1]} features")
        logger.info(f"  ✓ Bias term (expected value): {np.mean(bias_term):.4f}")
        
        return shap_values, bias_term, test_X_df, shap_feature_names
        
    except Exception as e:
        logger.error(f"  ❌ SHAP extraction failed: {e}")
        logger.error(f"  Model type: {type(model)}")
        logger.error(f"  Model algorithm: {model.algo if hasattr(model, 'algo') else 'unknown'}")
        raise

def create_shap_explanation_object(shap_values, bias_term, test_X_df):
    """
    Create SHAP Explanation object for visualization.
    
    Reference: SHAP 0.41+ API
    https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html
    """
    try:
        shap_explanation = shap.Explanation(
            values=shap_values,           # SHAP contributions (n_samples × n_features)
            base_values=bias_term,        # Model's expected value
            data=test_X_df.values,        # Original feature values
            feature_names=test_X_df.columns.tolist()
        )
        logger.info(f"  ✓ SHAP Explanation object created")
        return shap_explanation
    except Exception as e:
        logger.error(f"  ❌ Failed to create SHAP Explanation: {e}")
        raise

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_shap_summary(shap_explanation, test_X_df, model_name, dataset_name):
    """
    Generate SHAP Summary Plot (Global Feature Importance).
    
    Shows: Which features matter most + direction of impact
    Reference: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/summary_plot.html
    """
    logger.info(f"  Creating summary plot...")
    
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_explanation, 
            test_X_df, 
            show=False, 
            max_display=MAX_DISPLAY_FEATURES
        )
        plt.title(f'SHAP Feature Importance - {model_name} ({dataset_name})', 
                  fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        
        filename = f'{PLOTS_DIR}/shap_summary_{model_name}_{dataset_name}.png'
        plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Summary plot saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ Summary plot failed: {e}")
        plt.close()
        return None

def plot_shap_beeswarm(shap_explanation, model_name, dataset_name):
    """
    Generate SHAP Beeswarm Plot (Feature Value Distribution + Impact).
    
    Shows: How feature values affect predictions (color = feature value)
    Reference: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
    """
    logger.info(f"  Creating beeswarm plot...")
    
    try:
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
        plt.title(f'SHAP Beeswarm - {model_name} ({dataset_name})', 
                  fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        
        filename = f'{PLOTS_DIR}/shap_beeswarm_{model_name}_{dataset_name}.png'
        plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Beeswarm plot saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ Beeswarm plot failed: {e}")
        plt.close()
        return None

def plot_shap_dependence(shap_values, test_X_df, feature_name, model_name, 
                         dataset_name, interaction_feature=None):
    """
    Generate SHAP Dependence Plot (Single Feature Analysis).
    
    Shows: Relationship between feature value and SHAP value
    Reference: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/dependence_plot.html
    """
    logger.info(f"  Creating dependence plot for: {feature_name}")
    
    try:
        plt.figure(figsize=(10, 6))
        
        if interaction_feature and interaction_feature in test_X_df.columns:
            shap.dependence_plot(
                feature_name,
                shap_values, 
                test_X_df,
                interaction_index=interaction_feature,
                show=False
            )
            title_suffix = f'(colored by {interaction_feature})'
        else:
            shap.dependence_plot(
                feature_name,
                shap_values, 
                test_X_df,
                show=False
            )
            title_suffix = ''
        
        plt.title(f'SHAP Dependence: {feature_name} {title_suffix}\n{model_name} ({dataset_name})',
                  fontsize=12, weight='bold')
        plt.tight_layout()
        
        filename = f'{PLOTS_DIR}/shap_dependence_{feature_name}_{model_name}_{dataset_name}.png'
        plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Dependence plot saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ Dependence plot failed for {feature_name}: {e}")
        plt.close()
        return None

def plot_shap_waterfall(shap_explanation, patient_idx, model_name, dataset_name):
    """
    Generate SHAP Waterfall Plot (Single Patient Explanation).
    
    Shows: How each feature pushed prediction away from base value
    Perfect for BleuLink patient-facing UI
    Reference: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html
    """
    logger.info(f"  Creating waterfall plot for patient {patient_idx}...")
    
    try:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation[patient_idx], show=False)
        plt.title(f'SHAP Explanation - Patient {patient_idx}\n{model_name} ({dataset_name})',
                  fontsize=12, weight='bold', pad=20)
        plt.tight_layout()
        
        filename = f'{PLOTS_DIR}/shap_waterfall_patient_{patient_idx}_{model_name}_{dataset_name}.png'
        plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Waterfall plot saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ Waterfall plot failed for patient {patient_idx}: {e}")
        plt.close()
        return None

def plot_shap_force(shap_explanation, patient_idx, model_name, dataset_name):
    """
    Generate SHAP Force Plot (Interactive HTML for BleuLink).
    
    Shows: Detailed breakdown for single prediction
    Can be embedded in BleuLink dashboard
    Reference: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/force.html
    """
    logger.info(f"  Creating force plot (HTML) for patient {patient_idx}...")
    
    try:
        force_plot = shap.force_plot(
            shap_explanation.base_values[patient_idx],
            shap_explanation.values[patient_idx, :],
            shap_explanation.data[patient_idx, :],
            feature_names=shap_explanation.feature_names,
            matplotlib=False,
            show=False
        )
        
        filename = f'{HTML_DIR}/shap_force_patient_{patient_idx}_{model_name}_{dataset_name}.html'
        shap.save_html(filename, force_plot)
        
        logger.info(f"  ✓ Force plot (HTML) saved: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ Force plot failed for patient {patient_idx}: {e}")
        return None

# ============================================================================
# AGGREGATION & EXPORT FUNCTIONS
# ============================================================================

def calculate_shap_importance(shap_values, feature_names):
    """
    Calculate mean |SHAP| for each feature (global importance).
    
    Reference: https://christophm.github.io/interpretable-ml-book/shap.html#feature-importance
    """
    logger.info("  Calculating SHAP feature importance...")
    
    try:
        # Mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create ranking DataFrame
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap,
            'rank': np.argsort(-mean_abs_shap) + 1
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info(f"  ✓ SHAP importance calculated for {len(feature_names)} features")
        
        return shap_importance_df
        
    except Exception as e:
        logger.error(f"  ❌ SHAP importance calculation failed: {e}")
        raise

def export_shap_values_to_csv(shap_values, feature_names, model_name, dataset_name):
    """Export SHAP values to CSV for clinical reporting."""
    logger.info("  Exporting SHAP values to CSV...")
    
    try:
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        filename = f'{CSV_DIR}/shap_values_{model_name}_{dataset_name}.csv'
        shap_df.to_csv(filename, index=False)
        
        logger.info(f"  ✓ SHAP values exported: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ SHAP export failed: {e}")
        return None

def export_shap_importance_to_csv(shap_importance_df, model_name, dataset_name):
    """Export SHAP importance ranking to CSV."""
    logger.info("  Exporting SHAP importance ranking...")
    
    try:
        filename = f'{CSV_DIR}/shap_feature_importance_{model_name}_{dataset_name}.csv'
        shap_importance_df.to_csv(filename, index=False)
        
        logger.info(f"  ✓ SHAP importance exported: {filename}")
        
        # Print top 10 for logging
        logger.info("\n" + "="*80)
        logger.info(f"TOP 10 FEATURES BY SHAP IMPORTANCE - {model_name} ({dataset_name})")
        logger.info("="*80)
        logger.info("\n" + shap_importance_df.head(10).to_string(index=False))
        
        return filename
        
    except Exception as e:
        logger.error(f"  ❌ SHAP importance export failed: {e}")
        return None

# ============================================================================
# MAIN SHAP ANALYSIS FUNCTION
# ============================================================================

def analyze_model_shap(model_path, dataset_name, seed=42, 
                       analyze_top_features=3, analyze_patients=[0, 50, 100]):
    """
    Comprehensive SHAP analysis for a single model.
    
    Args:
        model_path: Path to trained H2O model directory
        dataset_name: Name of dataset (e.g., 'PrePostFinal')
        seed: Random seed for reproducibility
        analyze_top_features: Number of top features for dependence plots
        analyze_patients: Patient indices for waterfall/force plots
        
    Returns:
        dict: Results summary
    """
    logger.info("\n" + "="*80)
    logger.info(f"SHAP ANALYSIS: {model_path}")
    logger.info("="*80)
    
    results = {
        'model_path': model_path,
        'dataset_name': dataset_name,
        'status': 'failed',
        'plots': {},
        'csv_files': {}
    }
    
    try:
        # Load model
        model = load_h2o_model(model_path)
        if model is None:
            return results
        
        model_name = Path(model_path).stem
        
        # Load dataset
        dataset_path = find_dataset_file(dataset_name)
        if dataset_path is None:
            return results
        
        logger.info(f"Loading dataset: {dataset_name}")
        df = pd.read_csv(dataset_path)
        logger.info(f"  ✓ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
        
        # Convert to H2OFrame
        h2o_frame = h2o.H2OFrame(df)
        
        # Create 80/20 split (same as training)
        splits = h2o_frame.split_frame(ratios=[0.8], seed=seed)
        test_h2o = splits[1]
        logger.info(f"  ✓ Test set: {test_h2o.nrows} rows")
        
        # Get feature names (exclude target)
        feature_names = [col for col in df.columns if col != TARGET]
        
        # Extract SHAP values
        shap_values, bias_term, test_X_df, shap_feature_names = extract_shap_values_from_h2o(
            model, test_h2o, feature_names
        )
        
        # Create SHAP Explanation object
        shap_explanation = create_shap_explanation_object(shap_values, bias_term, test_X_df)
        
        # ====================================================================
        # VISUALIZATION 1: Summary Plot
        # ====================================================================
        plot_file = plot_shap_summary(shap_explanation, test_X_df, model_name, dataset_name)
        if plot_file:
            results['plots']['summary'] = plot_file
        
        # ====================================================================
        # VISUALIZATION 2: Beeswarm Plot
        # ====================================================================
        plot_file = plot_shap_beeswarm(shap_explanation, model_name, dataset_name)
        if plot_file:
            results['plots']['beeswarm'] = plot_file
        
        # ====================================================================
        # VISUALIZATION 3: Dependence Plots (Top Features)
        # ====================================================================
        shap_importance_df = calculate_shap_importance(shap_values, shap_feature_names)
        top_features = shap_importance_df.head(analyze_top_features)['feature'].tolist()
        
        results['plots']['dependence'] = []
        for i, feature in enumerate(top_features):
            if feature in test_X_df.columns:
                # Use second-most important feature for interaction coloring
                interaction_feat = top_features[i+1] if i+1 < len(top_features) else None
                
                plot_file = plot_shap_dependence(
                    shap_values, test_X_df, feature, model_name, dataset_name, interaction_feat
                )
                if plot_file:
                    results['plots']['dependence'].append(plot_file)
        
        # ====================================================================
        # VISUALIZATION 4: Waterfall Plots (Example Patients)
        # ====================================================================
        results['plots']['waterfall'] = []
        for patient_idx in analyze_patients:
            if patient_idx < len(shap_values):
                plot_file = plot_shap_waterfall(shap_explanation, patient_idx, model_name, dataset_name)
                if plot_file:
                    results['plots']['waterfall'].append(plot_file)
        
        # ====================================================================
        # VISUALIZATION 5: Force Plots (Interactive HTML)
        # ====================================================================
        results['plots']['force'] = []
        for patient_idx in analyze_patients:
            if patient_idx < len(shap_values):
                plot_file = plot_shap_force(shap_explanation, patient_idx, model_name, dataset_name)
                if plot_file:
                    results['plots']['force'].append(plot_file)
        
        # ====================================================================
        # EXPORT: SHAP Values & Importance
        # ====================================================================
        csv_file = export_shap_values_to_csv(shap_values, shap_feature_names, model_name, dataset_name)
        if csv_file:
            results['csv_files']['shap_values'] = csv_file
        
        csv_file = export_shap_importance_to_csv(shap_importance_df, model_name, dataset_name)
        if csv_file:
            results['csv_files']['shap_importance'] = csv_file
        
        results['status'] = 'success'
        results['n_samples'] = len(shap_values)
        results['n_features'] = len(shap_feature_names)
        results['expected_value'] = float(np.mean(bias_term))
        
        logger.info(f"\n✅ SHAP analysis completed successfully for {model_name}")
        
    except Exception as e:
        logger.error(f"\n❌ SHAP analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return results

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def analyze_all_models(model_configs, seed=42):
    """
    Run SHAP analysis for all trained models.
    
    Args:
        model_configs: List of dicts with 'model_path' and 'dataset_name'
        seed: Random seed for reproducibility
        
    Returns:
        list: Results for all models
    """
    logger.info("\n" + "="*80)
    logger.info("BATCH SHAP ANALYSIS - ALL MODELS")
    logger.info("="*80)
    logger.info(f"Total models to analyze: {len(model_configs)}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for i, config in enumerate(model_configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL {i}/{len(model_configs)}")
        logger.info(f"{'='*80}")
        
        results = analyze_model_shap(
            model_path=config['model_path'],
            dataset_name=config['dataset_name'],
            seed=seed
        )
        
        all_results.append(results)
    
    # Create summary report
    create_summary_report(all_results)
    
    logger.info("\n" + "="*80)
    logger.info("✅ BATCH SHAP ANALYSIS COMPLETED")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    
    return all_results

def create_summary_report(all_results):
    """Create summary report for all SHAP analyses."""
    logger.info("\nGenerating summary report...")
    
    try:
        summary_data = []
        for result in all_results:
            if result['status'] == 'success':
                summary_data.append({
                    'Model': Path(result['model_path']).stem,
                    'Dataset': result['dataset_name'],
                    'N_Samples': result.get('n_samples', 0),
                    'N_Features': result.get('n_features', 0),
                    'Expected_Value': result.get('expected_value', 0),
                    'Summary_Plot': 'Yes' if 'summary' in result['plots'] else 'No',
                    'Beeswarm_Plot': 'Yes' if 'beeswarm' in result['plots'] else 'No',
                    'Dependence_Plots': len(result['plots'].get('dependence', [])),
                    'Waterfall_Plots': len(result['plots'].get('waterfall', [])),
                    'Force_Plots': len(result['plots'].get('force', [])),
                    'Status': result['status']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(OUTPUT_DIR, 'shap_analysis_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"\n{'='*80}")
            logger.info("SHAP ANALYSIS SUMMARY")
            logger.info(f"{'='*80}")
            logger.info("\n" + summary_df.to_string(index=False))
            logger.info(f"\n✓ Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create summary report: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("🔍 SHAP ANALYSIS FOR H2O AUTOML MODELS")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Goal: Generate SHAP explanations for all trained models")
    print("="*80)
    
    # Setup
    setup_directories()
    
    # Initialize H2O
    try:
        h2o.init()
        logger.info("✓ H2O cluster initialized")
    except Exception as e:
        logger.error(f"❌ H2O initialization failed: {e}")
        sys.exit(1)
    
    # Define models to analyze
    # Adjust paths based on your Phase 3 training output
    model_configs = [
        # Individual datasets (from Phase 3)
        {
            'model_path': './models/h2o_model_PrePostFinal_v2',
            'dataset_name': 'PrePostFinal'
        },
        {
            'model_path': './models/h2o_model_nmbfinalDiabetes_v2',
            'dataset_name': 'nmbfinalDiabetes'
        },
        {
            'model_path': './models/h2o_model_nmbfinalnewDiabetes_v2',
            'dataset_name': 'nmbfinalnewDiabetes'
        },
        
        # Combined datasets (from Phase 7 - if available)
        # Uncomment after Phase 7 is complete
        # {
        #     'model_path': './models/h2o_model_Combo_A_nmb1_nmb2',
        #     'dataset_name': 'Combo_A_nmb1_nmb2'
        # },
        # {
        #     'model_path': './models/h2o_model_Combo_B_nmb1_pp',
        #     'dataset_name': 'Combo_B_nmb1_pp'
        # },
        # {
        #     'model_path': './models/h2o_model_Combo_C_nmb2_pp',
        #     'dataset_name': 'Combo_C_nmb2_pp'
        # },
    ]
    
    # Run batch analysis
    all_results = analyze_all_models(model_configs, seed=42)
    
    # Shutdown H2O
    h2o.cluster().shutdown()
    logger.info("✓ H2O cluster shutdown")
    
    print("\n" + "="*80)
    print("✅ SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"📊 Results directory: {OUTPUT_DIR}")
    print(f"📈 Plots: {PLOTS_DIR}")
    print(f"📄 CSV files: {CSV_DIR}")
    print(f"🌐 HTML files: {HTML_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
