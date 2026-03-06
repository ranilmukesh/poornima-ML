
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial data cleaning."""
    print("[*] Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"    Original shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    
    # Drop ID (irrelevant for prediction)
    df = df.drop(columns=['id'])
    
    # Remove 'Other' gender (usually only 1-2 records, causes noise)
    other_count = (df['gender'] == 'Other').sum()
    if other_count > 0:
        print(f"    Removing {other_count} record(s) with gender='Other'")
        df = df[df['gender'] != 'Other']
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n[!] Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"    - {col}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n    Cleaned shape: {df.shape}")
    return df


def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    
    # Numeric: Impute missing values (BMI) with mean, then standardize
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def get_feature_names_after_encoding(preprocessor, numeric_features: list, categorical_features: list) -> list:
    """Extract feature names after OneHotEncoding."""
    # Get one-hot encoded feature names
    ohe_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    
    # Combine: numeric features first, then encoded categorical
    all_feature_names = list(numeric_features) + list(ohe_feature_names)
    return all_feature_names


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Apply SMOTE to handle class imbalance."""
    print("\n[*] Applying SMOTE to balance the dataset...")
    print(f"    Before SMOTE - Class distribution:")
    print(f"    - No Stroke (0): {(y_train == 0).sum()}")
    print(f"    - Stroke (1): {(y_train == 1).sum()}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n    After SMOTE - Class distribution:")
    print(f"    - No Stroke (0): {(y_resampled == 0).sum()}")
    print(f"    - Stroke (1): {(y_resampled == 1).sum()}")
    
    return X_resampled, y_resampled


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Train XGBoost classifier with optimized hyperparameters."""
    print("\n[*] Training XGBoost Model...")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print("    [OK] Training complete!")
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Comprehensive model evaluation."""
    print("\n[*] Model Performance Evaluation")
    print("=" * 50)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n    Accuracy: {accuracy:.4f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"    ROC-AUC Score: {roc_auc:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"    F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    Confusion Matrix:")
    print(f"    TN: {cm[0][0]:5d}  |  FP: {cm[0][1]:5d}")
    print(f"    FN: {cm[1][0]:5d}  |  TP: {cm[1][1]:5d}")
    
    # Classification Report
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    
    # Feature Importance
    print("\n    Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def save_artifacts(preprocessor, model, feature_names: list, output_path: str):
    """Save model artifacts for API deployment."""
    print(f"\n[*] Saving artifacts to '{output_path}'...")
    
    artifacts = {
        'preprocessor': preprocessor,
        'model': model,
        'feature_names': feature_names
    }
    
    joblib.dump(artifacts, output_path)
    print(f"    [OK] Successfully saved!")
    print(f"\n    Artifact contents:")
    print(f"    - preprocessor: For data transformation")
    print(f"    - model: Trained XGBoost classifier")
    print(f"    - feature_names: List of {len(feature_names)} features")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("    CardioSense+ Model Training Pipeline")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = "healthcare-dataset-stroke-data.csv"
    OUTPUT_PATH = "cardiosense_artifacts.pkl"
    
    # Define feature columns
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = [
        'gender', 
        'hypertension', 
        'heart_disease', 
        'ever_married', 
        'work_type', 
        'Residence_type', 
        'smoking_status'
    ]
    
    # Step 1: Load and clean data
    df = load_and_clean_data(DATA_PATH)
    
    # Step 2: Separate features and target
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    print(f"\n[*] Target Distribution:")
    print(f"    - No Stroke (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"    - Stroke (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # Step 3: Train-Test Split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n[*] Train-Test Split:")
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples: {len(X_test)}")
    
    # Step 4: Create and fit preprocessor
    print("\n[*] Creating preprocessing pipeline...")
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after encoding
    feature_names = get_feature_names_after_encoding(
        preprocessor, numeric_features, categorical_features
    )
    print(f"    Total features after encoding: {len(feature_names)}")
    
    # Convert to DataFrames with proper column names
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Step 5: Handle class imbalance with SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_df, y_train)
    
    # Step 6: Train model
    model = train_xgboost(X_train_resampled, y_train_resampled)
    
    # Step 7: Evaluate model
    metrics = evaluate_model(model, X_test_df, y_test)
    
    # Step 8: Save artifacts
    save_artifacts(preprocessor, model, feature_names, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("    [OK] Training Pipeline Complete!")
    print("=" * 60)
    print(f"\n    Next steps:")
    print(f"    1. Run: uvicorn main:app --reload")
    print(f"    2. Open: http://127.0.0.1:8000/docs")
    print(f"    3. Test the /predict and /explain endpoints")
    print("=" * 60)


if __name__ == "__main__":
    main()
