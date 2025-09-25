"""
MissForest Imputation Implementation and Evaluation
==================================================

MissForest is a powerful imputation method that uses Random Forest:
• Handles mixed-type data (numerical + categorical) naturally
• Iteratively improves predictions using previous imputations
• Captures non-linear relationships and interactions
• Often superior to KNN for complex datasets

Let's test MissForest against our current methods and measure RMSE improvements.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os

def install_and_import_missforest():
    """Import MissForest and handle installation if needed."""
    
    try:
        from missingpy import MissForest
        print("✅ MissForest is available")
        return MissForest
    except ImportError:
        print("❌ MissForest not available. Implementing custom version...")
        return None

class CustomMissForest:
    """Custom implementation of MissForest algorithm."""
    
    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.imputers = {}
        self.column_types = {}
        self.label_encoders = {}
        
    def _detect_column_types(self, X):
        """Detect which columns are categorical vs numerical."""
        
        column_types = {}
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                column_types[col] = 'categorical'
            elif X[col].nunique() <= 10 and X[col].dtype != 'float64':
                # Low cardinality numerical might be categorical
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numerical'
        
        return column_types
    
    def _encode_categoricals(self, X):
        """Encode categorical variables for Random Forest."""
        
        X_encoded = X.copy()
        
        for col in X.columns:
            if self.column_types[col] == 'categorical':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values during encoding
                non_null_mask = X[col].notna()
                if non_null_mask.sum() > 0:
                    unique_values = X.loc[non_null_mask, col].astype(str).unique()
                    self.label_encoders[col].fit(unique_values)
                    
                    # Encode non-null values
                    X_encoded.loc[non_null_mask, col] = self.label_encoders[col].transform(
                        X.loc[non_null_mask, col].astype(str)
                    )
        
        return X_encoded
    
    def _decode_categoricals(self, X_encoded, original_X):
        """Decode categorical variables back to original format."""
        
        X_decoded = X_encoded.copy()
        
        for col in X_encoded.columns:
            if self.column_types[col] == 'categorical' and col in self.label_encoders:
                # Round to integers and clip to valid range
                encoded_values = np.round(X_encoded[col]).astype(int)
                n_classes = len(self.label_encoders[col].classes_)
                encoded_values = np.clip(encoded_values, 0, n_classes - 1)
                
                # Decode back to original categories
                try:
                    decoded_values = self.label_encoders[col].inverse_transform(encoded_values)
                    X_decoded[col] = decoded_values
                except:
                    # If decoding fails, use mode of original
                    mode_value = original_X[col].mode().iloc[0] if not original_X[col].mode().empty else 'Unknown'
                    X_decoded[col] = mode_value
        
        return X_decoded
    
    def fit_transform(self, X):
        """Fit MissForest and transform data."""
        
        print(f"🌲 MissForest: Processing {X.shape[0]} rows × {X.shape[1]} columns")
        
        # Detect column types
        self.column_types = self._detect_column_types(X)
        
        categorical_cols = [col for col, dtype in self.column_types.items() if dtype == 'categorical']
        numerical_cols = [col for col, dtype in self.column_types.items() if dtype == 'numerical']
        
        print(f"   📊 Detected: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical columns")
        
        # Initial setup
        X_imputed = X.copy()
        X_encoded = self._encode_categoricals(X_imputed)
        
        # Initialize missing values with simple imputation
        for col in X_encoded.columns:
            missing_mask = X_encoded[col].isna()
            if missing_mask.sum() > 0:
                if self.column_types[col] == 'categorical':
                    # Use mode for categorical
                    fill_value = X_encoded[col].mode().iloc[0] if not X_encoded[col].mode().empty else 0
                else:
                    # Use median for numerical
                    fill_value = X_encoded[col].median()
                
                X_encoded.loc[missing_mask, col] = fill_value
        
        # Iterative imputation
        missing_indicators = {}
        for col in X.columns:
            missing_indicators[col] = X[col].isna()
        
        for iteration in range(self.max_iter):
            print(f"   🔄 Iteration {iteration + 1}/{self.max_iter}")
            
            X_previous = X_encoded.copy()
            
            # Impute each column with missing values
            for col in X.columns:
                if missing_indicators[col].sum() == 0:
                    continue  # No missing values in this column
                
                # Prepare features (all other columns)
                feature_cols = [c for c in X_encoded.columns if c != col]
                
                if len(feature_cols) == 0:
                    continue
                
                # Training data (non-missing values for target column)
                train_mask = ~missing_indicators[col]
                
                if train_mask.sum() < 5:  # Need minimum samples
                    continue
                
                X_train = X_encoded.loc[train_mask, feature_cols]
                y_train = X_encoded.loc[train_mask, col]
                
                # Test data (missing values for target column)
                test_mask = missing_indicators[col]
                X_test = X_encoded.loc[test_mask, feature_cols]
                
                if len(X_test) == 0:
                    continue
                
                # Train Random Forest
                try:
                    if self.column_types[col] == 'categorical':
                        imputer = RandomForestClassifier(
                            n_estimators=min(self.n_estimators, 50),  # Limit for speed
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                    else:
                        imputer = RandomForestRegressor(
                            n_estimators=min(self.n_estimators, 50),  # Limit for speed
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                    
                    imputer.fit(X_train, y_train)
                    
                    # Predict missing values
                    y_pred = imputer.predict(X_test)
                    
                    # Update imputed values
                    X_encoded.loc[test_mask, col] = y_pred
                    
                    # Store imputer for this column
                    self.imputers[col] = imputer
                    
                except Exception as e:
                    print(f"      ⚠️ Error imputing {col}: {str(e)[:50]}...")
                    continue
            
            # Check for convergence (optional - simplified)
            if iteration > 0:
                diff = np.mean(np.abs(X_encoded.values - X_previous.values))
                print(f"      📈 Average change: {diff:.6f}")
                
                if diff < 1e-6:
                    print(f"      ✅ Converged at iteration {iteration + 1}")
                    break
        
        # Decode categorical variables back
        X_final = self._decode_categoricals(X_encoded, X)
        
        print(f"   ✅ MissForest completed in {iteration + 1} iterations")
        
        return X_final

def test_missforest_performance():
    """Test MissForest performance against other methods."""
    
    print("🌲 MISSFOREST IMPUTATION EVALUATION")
    print("=" * 60)
    
    # Load test data
    test_file = "cleaned data/nmbfinalDiabetes (4)_selected_columns_cleaned_processed.csv"
    
    if not os.path.exists(test_file):
        print("❌ Test file not found.")
        return
    
    df = pd.read_csv(test_file)
    
    # Test variables
    test_variables = {
        'PostBLHBA1C': {
            'type': 'numerical',
            'description': 'HbA1c Level (%)',
            'target_rmse': 1.0
        },
        'PreBLAge': {
            'type': 'numerical',
            'description': 'Patient Age (years)',
            'target_rmse': 5.0
        },
        'PreRgender': {
            'type': 'categorical',
            'description': 'Gender',
            'target_rmse': 0.3
        },
        'PreRarea': {
            'type': 'categorical',
            'description': 'Urban/Rural',
            'target_rmse': 0.3
        }
    }
    
    print(f"🎯 Testing MissForest on {len(test_variables)} key variables")
    
    results = {}
    
    for var_name, var_config in test_variables.items():
        if var_name not in df.columns:
            continue
            
        print(f"\n📊 {var_name} ({var_config['description']})")
        print(f"   Type: {var_config['type']} | Target RMSE: ≤{var_config['target_rmse']}")
        print("-" * 50)
        
        # Get complete data
        complete_mask = df[var_name].notna()
        if complete_mask.sum() < 200:
            print(f"   ⚠️ Insufficient data ({complete_mask.sum()} rows)")
            continue
        
        complete_data = df[complete_mask].copy()
        
        # Select features (limit for performance)
        if var_config['type'] == 'numerical':
            feature_cols = df.select_dtypes(include=[np.number]).columns
        else:
            feature_cols = df.columns
        
        feature_cols = [col for col in feature_cols if col != var_name][:15]  # Limit features
        
        if len(feature_cols) < 5:
            print(f"   ⚠️ Too few features ({len(feature_cols)})")
            continue
        
        # Test scenarios
        missing_scenarios = {
            'Low Missing (15%)': 0.15,
            'Moderate Missing (25%)': 0.25
        }
        
        var_results = {}
        
        for scenario_name, missing_rate in missing_scenarios.items():
            print(f"\n   🧪 {scenario_name}")
            
            # Create test dataset
            test_data = complete_data[feature_cols + [var_name]].copy()
            n_test = min(400, len(test_data))  # Limit for speed
            test_data = test_data.sample(n=n_test, random_state=42)
            
            # Create artificial missingness
            n_missing = int(len(test_data) * missing_rate)
            np.random.seed(42)
            missing_indices = np.random.choice(test_data.index, n_missing, replace=False)
            
            true_values = test_data.loc[missing_indices, var_name].copy()
            test_data_missing = test_data.copy()
            test_data_missing.loc[missing_indices, var_name] = np.nan
            
            # Method 1: MissForest
            try:
                print(f"      🌲 Running MissForest...")
                
                # Use custom MissForest implementation
                missforest = CustomMissForest(max_iter=5, n_estimators=30)  # Reduced for speed
                imputed_data = missforest.fit_transform(test_data_missing)
                
                missforest_predictions = imputed_data.loc[missing_indices, var_name]
                
                # Handle categorical encoding for RMSE calculation
                if var_config['type'] == 'categorical':
                    # Encode both true and predicted for comparison
                    encoder = LabelEncoder()
                    combined_values = list(true_values.astype(str)) + list(missforest_predictions.astype(str))
                    encoder.fit(combined_values)
                    
                    true_encoded = encoder.transform(true_values.astype(str))
                    pred_encoded = encoder.transform(missforest_predictions.astype(str))
                    
                    mf_mse = mean_squared_error(true_encoded, pred_encoded)
                    mf_mae = mean_absolute_error(true_encoded, pred_encoded)
                else:
                    mf_mse = mean_squared_error(true_values, missforest_predictions)
                    mf_mae = mean_absolute_error(true_values, missforest_predictions)
                
                mf_rmse = np.sqrt(mf_mse)
                
                print(f"         ✅ MissForest: RMSE={mf_rmse:.4f} | MAE={mf_mae:.4f}")
                
            except Exception as e:
                print(f"         ❌ MissForest failed: {str(e)[:60]}...")
                mf_rmse = None
            
            # Method 2: Baseline (Mean/Mode)
            if var_config['type'] == 'categorical':
                baseline_value = test_data_missing[var_name].mode().iloc[0] if not test_data_missing[var_name].mode().empty else 'Unknown'
            else:
                baseline_value = test_data_missing[var_name].mean()
            
            baseline_predictions = pd.Series([baseline_value] * len(missing_indices), index=missing_indices)
            
            if var_config['type'] == 'categorical':
                # Encode for comparison
                try:
                    baseline_encoded = encoder.transform(baseline_predictions.astype(str))
                    baseline_mse = mean_squared_error(true_encoded, baseline_encoded)
                    baseline_mae = mean_absolute_error(true_encoded, baseline_encoded)
                except:
                    baseline_mse = 1.0  # Default high error
                    baseline_mae = 1.0
            else:
                baseline_mse = mean_squared_error(true_values, baseline_predictions)
                baseline_mae = mean_absolute_error(true_values, baseline_predictions)
            
            baseline_rmse = np.sqrt(baseline_mse)
            
            print(f"         📊 Baseline: RMSE={baseline_rmse:.4f} | MAE={baseline_mae:.4f}")
            
            # Compare performance
            if mf_rmse is not None:
                improvement = ((baseline_rmse - mf_rmse) / baseline_rmse) * 100
                status = "📈 Better" if improvement > 0 else "📉 Worse"
                
                meets_target = mf_rmse <= var_config['target_rmse']
                target_status = "✅ Meets target" if meets_target else "⚠️ Above target"
                
                print(f"         {status}: MissForest vs Baseline = {improvement:+.1f}%")
                print(f"         {target_status} ({var_config['target_rmse']:.1f})")
                
                var_results[scenario_name] = {
                    'missforest_rmse': mf_rmse,
                    'baseline_rmse': baseline_rmse,
                    'improvement': improvement,
                    'meets_target': meets_target
                }
        
        if var_results:
            # Variable summary
            avg_improvement = np.mean([r['improvement'] for r in var_results.values()])
            scenarios_meeting_target = sum([r['meets_target'] for r in var_results.values()])
            
            print(f"\n   📈 {var_name} SUMMARY:")
            print(f"      Average Improvement: {avg_improvement:+.1f}%")
            print(f"      Scenarios Meeting Target: {scenarios_meeting_target}/{len(var_results)}")
            
            results[var_name] = {
                'config': var_config,
                'results': var_results,
                'avg_improvement': avg_improvement,
                'success_rate': scenarios_meeting_target / len(var_results)
            }
    
    return results

def create_missforest_summary(results):
    """Create comprehensive MissForest performance summary."""
    
    print(f"\n\n🌲 MISSFOREST PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No results to summarize")
        return
    
    # Overall statistics
    all_improvements = []
    total_scenarios = 0
    successful_scenarios = 0
    
    for var_name, var_data in results.items():
        improvement = var_data['avg_improvement']
        success_rate = var_data['success_rate']
        
        all_improvements.append(improvement)
        
        scenarios = len(var_data['results'])
        total_scenarios += scenarios
        successful_scenarios += int(success_rate * scenarios)
        
        print(f"\n📊 {var_name}:")
        print(f"   Average Improvement: {improvement:+.1f}%")
        print(f"   Success Rate: {success_rate*100:.0f}%")
        print(f"   Target: ≤{var_data['config']['target_rmse']}")
    
    # Overall performance
    overall_improvement = np.mean(all_improvements)
    overall_success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
    
    print(f"\n🎯 OVERALL MISSFOREST PERFORMANCE:")
    print(f"   Average Improvement over Baselines: {overall_improvement:+.1f}%")
    print(f"   Clinical Target Achievement: {overall_success_rate*100:.1f}%")
    print(f"   Total Scenarios Tested: {total_scenarios}")
    
    # Grade MissForest performance
    if overall_improvement > 20:
        grade = "A+"
        status = "EXCELLENT - Significant improvement!"
    elif overall_improvement > 10:
        grade = "A"
        status = "VERY GOOD - Clear improvement"
    elif overall_improvement > 5:
        grade = "B+"
        status = "GOOD - Moderate improvement"
    elif overall_improvement > 0:
        grade = "B"
        status = "ACCEPTABLE - Some improvement"
    else:
        grade = "C"
        status = "NEEDS OPTIMIZATION - No clear benefit"
    
    print(f"\n🏆 MISSFOREST GRADE: {grade}")
    print(f"   Status: {status}")
    
    print(f"\n💡 COMPARISON WITH PREVIOUS METHODS:")
    print("=" * 45)
    print("   🔸 KNN-only: Mixed performance, categorical > numerical")
    print("   🔸 Smart Hybrid: Context-aware, 2.8x clinical targets")
    print(f"   🔸 MissForest: {overall_improvement:+.1f}% improvement, handles mixed data")
    
    if overall_improvement > 10:
        print(f"\n✅ RECOMMENDATION: Integrate MissForest into pipeline")
        print("   • Use MissForest for complex variables with interactions")
        print("   • Combine with Hybrid System for optimal results")
        print("   • Focus on variables showing >15% improvement")
    else:
        print(f"\n⚖️ RECOMMENDATION: Selective MissForest usage")
        print("   • Use for specific variables where it shows improvement")
        print("   • Maintain Smart Hybrid as primary method")
        print("   • Consider computational cost vs accuracy gain")

def main():
    """Run complete MissForest evaluation."""
    
    print("🌲 MISSFOREST IMPUTATION EVALUATION")
    print("Advanced Random Forest-based Imputation Testing")
    print("=" * 70)
    
    # Check MissForest availability
    MissForest = install_and_import_missforest()
    
    # Run MissForest testing
    results = test_missforest_performance()
    
    # Create summary
    create_missforest_summary(results)
    
    print(f"\n🎯 MISSFOREST EVALUATION COMPLETE!")
    print("=" * 50)
    print("""
🌲 MissForest Advantages:
• Captures non-linear relationships and interactions
• Handles mixed data types naturally
• Iteratively improves predictions
• Provides feature importance insights
• Robust to outliers and skewed distributions

⚡ Computational Considerations:
• More intensive than simple methods
• Scalable with parallel processing
• Tunable complexity vs speed trade-offs
""")

if __name__ == "__main__":
    main()