"""
Optimized SDV Imputation Strategy
=================================

The initial SDV test showed that advanced methods didn't outperform baselines.
This is likely because:
1. Limited training epochs (only 10 for speed)
2. Synthetic data matching strategy was too simplistic
3. Need better hyperparameter tuning
4. Should use SDV's built-in missing value handling

Let's create an optimized approach using SDV's native imputation capabilities.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def create_optimized_sdv_imputer():
    """Create an optimized imputation system using SDV."""
    
    print("🎯 OPTIMIZED SDV IMPUTATION SYSTEM")
    print("=" * 60)
    
    # Load test data
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found.")
        return
    
    df = pd.read_csv(test_file)
    
    # Focus on key variables with different strategies
    test_scenarios = {
        'HbA1c_Imputation': {
            'target': 'PostBLHBA1C',
            'description': 'Clinical glucose control measurement',
            'strategy': 'regression',
            'acceptable_rmse': 1.0
        },
        'Age_Imputation': {
            'target': 'PreBLAge', 
            'description': 'Patient demographic data',
            'strategy': 'regression',
            'acceptable_rmse': 5.0
        },
        'Gender_Imputation': {
            'target': 'PreRgender',
            'description': 'Categorical demographic',
            'strategy': 'classification',
            'acceptable_rmse': 0.3
        }
    }
    
    results = {}
    
    for scenario_name, config in test_scenarios.items():
        target_col = config['target']
        
        if target_col not in df.columns:
            continue
            
        print(f"\n📊 {scenario_name}: {config['description']}")
        print(f"Target: {target_col} | Strategy: {config['strategy']}")
        print("-" * 50)
        
        # Get complete data for testing
        complete_mask = df[target_col].notna()
        if complete_mask.sum() < 200:
            print(f"   ⚠️  Insufficient data ({complete_mask.sum()} rows)")
            continue
        
        complete_df = df[complete_mask].copy()
        
        # Select relevant features - more sophisticated selection
        if config['strategy'] == 'regression':
            # For numerical targets, use numerical features
            numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Add some categorical features if they exist
            categorical_cols = ['PreRgender', 'PreRarea', 'PostRgroupname']
            for col in categorical_cols:
                if col in complete_df.columns and col != target_col:
                    feature_cols.append(col)
                    
        else:  # classification
            # For categorical targets, use mix of features
            all_cols = list(complete_df.columns)
            feature_cols = [col for col in all_cols if col != target_col]
        
        # Limit features to prevent overfitting
        feature_cols = feature_cols[:15]
        
        if len(feature_cols) < 5:
            print(f"   ⚠️  Too few features ({len(feature_cols)})")
            continue
        
        # Create test scenario with more realistic missing patterns
        test_df = complete_df[feature_cols + [target_col]].copy()
        n_test = min(800, len(test_df))  # More data for better training
        test_df = test_df.sample(n=n_test, random_state=42)
        
        # Create missing data with different patterns
        missing_patterns = {
            'random_20': 0.20,
            'random_10': 0.10,
            'clustered': 0.15  # Missing in clusters (more realistic)
        }
        
        pattern_results = {}
        
        for pattern_name, missing_rate in missing_patterns.items():
            print(f"\n   🧪 Testing {pattern_name} ({missing_rate*100:.0f}% missing)")
            
            test_df_pattern = test_df.copy()
            n_missing = int(len(test_df_pattern) * missing_rate)
            
            if pattern_name.startswith('random'):
                # Random missing
                np.random.seed(42)
                missing_idx = np.random.choice(test_df_pattern.index, n_missing, replace=False)
            else:
                # Clustered missing (more realistic)
                np.random.seed(42)
                # Create clusters of missing values
                cluster_size = max(1, n_missing // 10)  # 10 clusters
                missing_idx = []
                
                for _ in range(10):
                    if len(missing_idx) >= n_missing:
                        break
                    start_idx = np.random.choice(test_df_pattern.index)
                    cluster_indices = list(range(start_idx, min(start_idx + cluster_size, len(test_df_pattern))))
                    missing_idx.extend([idx for idx in cluster_indices if idx in test_df_pattern.index])
                
                missing_idx = missing_idx[:n_missing]
            
            # Store true values and create missingness
            true_values = test_df_pattern.loc[missing_idx, target_col].copy()
            test_df_pattern.loc[missing_idx, target_col] = np.nan
            
            # Method 1: Improved SDV approach
            try:
                from sdv.single_table import GaussianCopulaSynthesizer
                from sdv.metadata import SingleTableMetadata
                
                # Create metadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(test_df_pattern)
                
                # Optimize metadata for target column
                if config['strategy'] == 'regression':
                    metadata.update_column(column_name=target_col, sdtype='numerical')
                else:
                    metadata.update_column(column_name=target_col, sdtype='categorical')
                
                # Train on complete cases only
                complete_training = test_df_pattern.dropna()
                
                if len(complete_training) < 50:
                    print(f"      ⚠️  Insufficient training data")
                    continue
                
                # Use GaussianCopula with optimized settings
                synthesizer = GaussianCopulaSynthesizer(
                    metadata, 
                    default_distribution='beta'  # Better for bounded data
                )
                
                synthesizer.fit(complete_training)
                
                # Generate larger synthetic dataset for better imputation
                synthetic_size = max(1000, len(complete_training) * 3)
                synthetic_data = synthesizer.sample(num_rows=synthetic_size)
                
                # Improved imputation strategy: Use KNN with synthetic data
                from sklearn.neighbors import NearestNeighbors
                from sklearn.preprocessing import StandardScaler
                
                # Prepare features for matching
                feature_scaler = StandardScaler()
                synthetic_features = feature_scaler.fit_transform(synthetic_data[feature_cols])
                
                # Find matches for missing values
                imputed_values = []
                
                for idx in missing_idx:
                    row_features = test_df_pattern.loc[idx, feature_cols].values
                    if np.isnan(row_features).any():
                        # If features have missing values, use mean of target
                        imputed_value = synthetic_data[target_col].mean()
                    else:
                        row_features_scaled = feature_scaler.transform([row_features])
                        
                        # Find K nearest neighbors in synthetic data
                        nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
                        nbrs.fit(synthetic_features)
                        
                        distances, indices = nbrs.kneighbors(row_features_scaled)
                        
                        # Use weighted average of nearest neighbors
                        weights = 1 / (distances[0] + 1e-6)  # Avoid division by zero
                        neighbor_values = synthetic_data.iloc[indices[0]][target_col].values
                        
                        imputed_value = np.average(neighbor_values, weights=weights)
                    
                    imputed_values.append(imputed_value)
                
                sdv_predictions = pd.Series(imputed_values, index=missing_idx)
                
                # Calculate SDV metrics
                sdv_mse = mean_squared_error(true_values, sdv_predictions)
                sdv_rmse = np.sqrt(sdv_mse)
                sdv_mae = mean_absolute_error(true_values, sdv_predictions)
                
                print(f"      🧠 SDV Enhanced: RMSE={sdv_rmse:.4f} | MAE={sdv_mae:.4f}")
                
            except Exception as e:
                print(f"      ❌ SDV Enhanced failed: {str(e)[:80]}...")
                sdv_rmse, sdv_mae = None, None
            
            # Method 2: Baseline comparison
            if config['strategy'] == 'regression':
                baseline_value = test_df_pattern[target_col].mean()
            else:
                baseline_value = test_df_pattern[target_col].mode().iloc[0] if not test_df_pattern[target_col].mode().empty else 0
            
            baseline_predictions = pd.Series([baseline_value] * len(missing_idx), index=missing_idx)
            baseline_mse = mean_squared_error(true_values, baseline_predictions)
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_mae = mean_absolute_error(true_values, baseline_predictions)
            
            print(f"      📊 Baseline: RMSE={baseline_rmse:.4f} | MAE={baseline_mae:.4f}")
            
            # Calculate improvement
            if sdv_rmse is not None:
                improvement = ((baseline_rmse - sdv_rmse) / baseline_rmse) * 100
                status = "📈 Better" if improvement > 0 else "📉 Worse"
                print(f"      {status}: SDV vs Baseline = {improvement:+.1f}%")
                
                # Check if meets acceptable threshold
                meets_target = sdv_rmse <= config['acceptable_rmse']
                target_status = "✅ Meets target" if meets_target else f"⚠️  Above target ({config['acceptable_rmse']:.1f})"
                print(f"      {target_status}")
                
                pattern_results[pattern_name] = {
                    'sdv_rmse': sdv_rmse,
                    'baseline_rmse': baseline_rmse,
                    'improvement': improvement,
                    'meets_target': meets_target
                }
        
        results[scenario_name] = {
            'config': config,
            'patterns': pattern_results
        }
    
    return results

def create_sdv_recommendations(results):
    """Create recommendations based on SDV testing."""
    
    print(f"\n\n🎯 SDV OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze.")
        return
    
    # Analyze overall performance
    total_improvements = []
    scenarios_meeting_targets = 0
    total_scenarios = 0
    
    for scenario_name, scenario_data in results.items():
        config = scenario_data['config']
        patterns = scenario_data['patterns']
        
        print(f"\n📊 {scenario_name}:")
        print(f"   Target Variable: {config['target']}")
        print(f"   Acceptable RMSE: ≤{config['acceptable_rmse']}")
        
        scenario_improvements = []
        
        for pattern_name, pattern_data in patterns.items():
            improvement = pattern_data['improvement']
            scenario_improvements.append(improvement)
            total_improvements.append(improvement)
            
            meets_target = pattern_data['meets_target']
            total_scenarios += 1
            if meets_target:
                scenarios_meeting_targets += 1
            
            print(f"   {pattern_name}: {improvement:+.1f}% improvement, Target: {'✅' if meets_target else '❌'}")
        
        if scenario_improvements:
            avg_improvement = np.mean(scenario_improvements)
            print(f"   📈 Average improvement: {avg_improvement:+.1f}%")
    
    # Overall summary
    if total_improvements:
        overall_avg = np.mean(total_improvements)
        target_rate = (scenarios_meeting_targets / total_scenarios) * 100
        
        print(f"\n🎯 OVERALL SDV PERFORMANCE:")
        print(f"   Average Improvement: {overall_avg:+.1f}%")
        print(f"   Target Achievement Rate: {target_rate:.1f}% ({scenarios_meeting_targets}/{total_scenarios})")
        
        if overall_avg > 10:
            print(f"   ✅ SDV shows SIGNIFICANT improvement! Recommend integration.")
        elif overall_avg > 0:
            print(f"   ⚖️  SDV shows MODERATE improvement. Consider selective use.")
        else:
            print(f"   ⚠️  SDV needs optimization before integration.")
        
        print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
        print("-" * 30)
        
        if overall_avg > 0:
            print("✅ Integrate GaussianCopulaSynthesizer for key variables")
            print("✅ Use weighted KNN approach with synthetic neighbors")
            print("✅ Focus on variables where SDV shows >10% improvement")
        
        print("🔧 OPTIMIZATION STRATEGIES:")
        print("• Increase synthetic data generation (3x training size)")
        print("• Use domain-specific feature engineering")  
        print("• Consider ensemble methods (SDV + traditional)")
        print("• Tune hyperparameters per variable type")

def main():
    """Run optimized SDV imputation testing."""
    
    print("🚀 OPTIMIZED SDV IMPUTATION EVALUATION")
    print("=" * 70)
    
    try:
        results = create_optimized_sdv_imputer()
        create_sdv_recommendations(results)
        
        print(f"\n🎯 CONCLUSION:")
        print("=" * 20)
        print("""
SDV represents the cutting-edge of imputation technology:
• Learns complex patterns from complete data
• Generates synthetic data preserving relationships  
• Can handle mixed data types naturally
• Provides uncertainty quantification

However, effectiveness depends on:
• Sufficient complete training data
• Appropriate hyperparameter tuning
• Domain-specific optimization
• Computational resources available

Recommendation: Use SDV for high-value variables where 
accuracy improvement justifies computational cost.
""")
        
    except Exception as e:
        print(f"❌ Error in SDV testing: {e}")

if __name__ == "__main__":
    main()