"""
DiabeSense+ API
AI-powered diabetes HbA1c prediction with explainable insights
"""

import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import numpy as np

# Graceful LLM import
try:
    from llm import start_chat_session, get_chat_response
    CHAT_AVAILABLE = True
    print("[OK] LLM Chat module loaded (agno + nvidia)")
except Exception as _llm_err:
    CHAT_AVAILABLE = False
    import traceback
    print(f"[!] LLM Chat unavailable: {_llm_err}")
    traceback.print_exc()
    print("    Install with: pip install agno sqlalchemy python-dotenv")

app = FastAPI(
    title="DiabeSense+ API",
    description="AI-powered diabetes HbA1c prediction with explainable insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabesense_artifacts.pkl")

model = None
shap_model = None
preprocessor = None
feature_names = None
base_feature_names = None
poly_transformer = None
poly_indices = None
explainer = None


def load_artifacts():
    """Load saved model artifacts."""
    global model, shap_model, preprocessor, feature_names, base_feature_names
    global poly_transformer, poly_indices, explainer

    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"Artifacts file '{ARTIFACTS_PATH}' not found. "
            "Please run 'python train_model.py' first."
        )

    print("[*] Loading model artifacts...")
    artifacts = joblib.load(ARTIFACTS_PATH)

    model = artifacts['model']
    preprocessor = artifacts['preprocessor']
    feature_names = artifacts['feature_names']

    # Polynomial feature support
    poly_transformer = artifacts.get('poly_transformer', None)
    poly_indices = artifacts.get('poly_indices', None)

    # Compute base feature names (before poly) from preprocessor
    try:
        ohe_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(
            [t[2] for t in preprocessor.transformers_ if t[0] == 'cat'][0]
        ) if hasattr(preprocessor, 'transformers_') else []
        num_names = [t[2] for t in preprocessor.transformers_ if t[0] == 'num'][0] \
            if hasattr(preprocessor, 'transformers_') else []
        base_feature_names = list(num_names) + list(ohe_names)
    except Exception:
        base_feature_names = feature_names  # fallback

    # Use dedicated XGB model for SHAP if available
    shap_model = artifacts.get('shap_model', model)

    print("[*] Initializing SHAP Explainer...")
    explainer = shap.TreeExplainer(shap_model)

    poly_info = f" + poly({len(feature_names) - len(base_feature_names)} interactions)" if poly_transformer else ""
    print(f"[OK] Loaded! {type(model).__name__} | {len(base_feature_names)} base features{poly_info}")
    print(f"     SHAP: {type(shap_model).__name__}")


@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
    except FileNotFoundError as e:
        print(f"[!] Warning: {e}")
        print("    The API will start, but predictions will fail until artifacts are available.")


# =============================================
# SCHEMAS
# =============================================

class PatientData(BaseModel):
    """Input schema for patient diabetes data."""
    PostBLAge: float = Field(..., ge=18, le=90, description="Patient age in years")
    PreBLGender: str = Field(..., description="Patient gender: 'Male', 'Female', or 'Others'")
    PreRarea: int = Field(..., ge=1, le=2, description="1=Urban, 2=Rural")
    PreRmaritalstatus: float = Field(..., description="Marital status code (1-5)")
    PreReducation: float = Field(..., description="Education level code (1-7)")
    PreRpresentoccupation: float = Field(..., description="Occupation code (1-9)")
    PreRdiafather: int = Field(..., ge=0, le=1, description="Father has diabetes: 1=Yes, 0=No")
    PreRdiamother: int = Field(..., ge=0, le=1, description="Mother has diabetes: 1=Yes, 0=No")
    PreRdiabrother: int = Field(..., ge=0, le=1, description="Brother has diabetes: 1=Yes, 0=No")
    PreRdiasister: int = Field(..., ge=0, le=1, description="Sister has diabetes: 1=Yes, 0=No")
    current_smoking: int = Field(..., ge=0, le=1, description="Currently smokes: 1=Yes, 0=No")
    current_alcohol: int = Field(..., ge=0, le=1, description="Currently drinks alcohol: 1=Yes, 0=No")
    PreRsleepquality: float = Field(..., description="Sleep quality (1=Good to 4=Poor)")
    PreRmildactivityduration: float = Field(..., description="Mild activity duration code (1-5)")
    PreRmoderate: float = Field(..., description="Moderate activity frequency (1-6)")
    PreRmoderateduration: float = Field(..., description="Moderate activity duration code (0-5)")
    PreRvigorous: float = Field(..., description="Vigorous activity frequency (1-6)")
    PreRvigorousduration: float = Field(..., description="Vigorous activity duration code (0-5)")
    PreRskipbreakfast: float = Field(..., description="Skip breakfast frequency (1-3)")
    PreRlessfruit: float = Field(..., description="Less fruit intake (1-3)")
    PreRlessvegetable: float = Field(..., description="Less vegetable intake (1-3)")
    PreRmilk: float = Field(..., description="Milk consumption (1-3)")
    PreRmeat: float = Field(..., description="Meat consumption (1-3)")
    PreRfriedfood: float = Field(..., description="Fried food consumption (1-3)")
    PreRsweet: float = Field(..., description="Sweet consumption (1-3)")
    PreRwaist: float = Field(..., ge=0, description="Waist circumference in cm")
    PreRBMI: float = Field(..., ge=0, le=80, description="Body Mass Index")
    PreRsystolicfirst: float = Field(..., ge=0, description="Systolic blood pressure mmHg")
    PreRdiastolicfirst: float = Field(..., ge=0, description="Diastolic blood pressure mmHg")
    PreBLPPBS: float = Field(..., ge=0, description="Post-prandial blood sugar mg/dL")
    PreBLFBS: float = Field(..., ge=0, description="Fasting blood sugar mg/dL")
    PreBLHBA1C: float = Field(..., ge=0, description="Pre-treatment HbA1c %")
    PreBLCHOLESTEROL: float = Field(..., ge=0, description="Cholesterol mg/dL")
    PreBLTRIGLYCERIDES: float = Field(..., ge=0, description="Triglycerides mg/dL")
    Diabetic_Duration: float = Field(..., ge=0, description="Duration of diabetes in years")
    PostRgroupname: int = Field(..., ge=1, le=2, description="1=Standard care + Yoga, 2=Standard care")

    class Config:
        json_schema_extra = {
            "example": {
                "PostBLAge": 55.0,
                "PreBLGender": "Male",
                "PreRarea": 1,
                "PreRmaritalstatus": 1.0,
                "PreReducation": 4.0,
                "PreRpresentoccupation": 3.0,
                "PreRdiafather": 1,
                "PreRdiamother": 0,
                "PreRdiabrother": 0,
                "PreRdiasister": 0,
                "current_smoking": 0,
                "current_alcohol": 0,
                "PreRsleepquality": 2.0,
                "PreRmildactivityduration": 3.0,
                "PreRmoderate": 2.0,
                "PreRmoderateduration": 2.0,
                "PreRvigorous": 1.0,
                "PreRvigorousduration": 1.0,
                "PreRskipbreakfast": 1.0,
                "PreRlessfruit": 2.0,
                "PreRlessvegetable": 2.0,
                "PreRmilk": 2.0,
                "PreRmeat": 2.0,
                "PreRfriedfood": 2.0,
                "PreRsweet": 2.0,
                "PreRwaist": 92.0,
                "PreRBMI": 27.5,
                "PreRsystolicfirst": 130.0,
                "PreRdiastolicfirst": 84.0,
                "PreBLPPBS": 220.0,
                "PreBLFBS": 140.0,
                "PreBLHBA1C": 8.2,
                "PreBLCHOLESTEROL": 210.0,
                "PreBLTRIGLYCERIDES": 180.0,
                "Diabetic_Duration": 5.0,
                "PostRgroupname": 1
            }
        }


class PredictionResponse(BaseModel):
    predicted_hba1c: float = Field(..., description="Predicted post-treatment HbA1c %")
    risk_level: str = Field(..., description="NORMAL, PRE_DIABETIC, DIABETIC, or HIGH_RISK")
    confidence: str = Field(..., description="Model confidence descriptor")
    delta_hba1c: Optional[float] = Field(None, description="Change from pre-treatment HbA1c")
    outcome_line: Optional[str] = Field(None, description="Predicted trajectory")
    response_line: Optional[str] = Field(None, description="Response classification")
    target_line: Optional[str] = Field(None, description="Glycemic target achievement")


class FactorImpact(BaseModel):
    feature: str
    impact: float
    direction: str
    interpretation: str


class ExplainResponse(BaseModel):
    top_contributing_factors: List[FactorImpact]
    base_value: float
    prediction_value: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# =============================================
# HELPERS
# =============================================

def prepare_input_dataframe(data: PatientData) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame."""
    input_dict = data.model_dump()
    return pd.DataFrame([input_dict])


def get_risk_level(hba1c: float) -> tuple:
    """Determine risk level from predicted HbA1c."""
    if hba1c < 5.7:
        return "NORMAL", "High Confidence"
    elif hba1c < 6.5:
        return "PRE_DIABETIC", "Moderate Confidence"
    elif hba1c < 8.0:
        return "DIABETIC", "Moderate Confidence"
    else:
        return "HIGH_RISK", "High Confidence"


def interpret_feature(feature_name: str, impact: float) -> str:
    """Generate human-readable interpretation for a feature's SHAP impact."""
    clean_name = feature_name.replace('_', ' ').title()

    # Handle one-hot encoded features
    if '_' in feature_name and any(x in feature_name.lower() for x in
                                    ['gender', 'area', 'marital', 'education',
                                     'occupation', 'smoking', 'alcohol',
                                     'sleep', 'groupname', 'breakfast',
                                     'fruit', 'vegetable', 'milk', 'meat',
                                     'fried', 'sweet', 'moderate', 'vigorous',
                                     'diafather', 'diamother', 'diabrother', 'diasister']):
        parts = feature_name.rsplit('_', 1)
        if len(parts) == 2:
            if impact > 0:
                return f"This category increases predicted HbA1c"
            else:
                return f"This category decreases predicted HbA1c"

    abs_impact = abs(impact)
    intensity = "significantly" if abs_impact > 0.3 else ("moderately" if abs_impact > 0.1 else "slightly")

    if impact > 0:
        return f"{clean_name} {intensity} increases predicted HbA1c"
    else:
        return f"{clean_name} {intensity} decreases predicted HbA1c"


def _apply_poly_features(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply polynomial interaction features to match training pipeline."""
    if poly_transformer is None or poly_indices is None:
        return processed_df

    X_poly_src = processed_df.iloc[:, poly_indices].values
    X_poly = poly_transformer.transform(X_poly_src)

    poly_names = poly_transformer.get_feature_names_out(
        [base_feature_names[i] for i in poly_indices]
    )
    new_poly_names = [n for n in poly_names if ' ' in n]
    new_poly_data = X_poly[:, len(poly_indices):]

    poly_df = pd.DataFrame(new_poly_data, columns=new_poly_names, index=processed_df.index)
    return pd.concat([processed_df, poly_df], axis=1)


def _predict_hba1c(patient_data: dict) -> float:
    """Internal helper to predict HbA1c from a patient dict."""
    input_df = pd.DataFrame([patient_data])
    processed_array = preprocessor.transform(input_df)
    processed_df = pd.DataFrame(processed_array, columns=base_feature_names)
    processed_df = _apply_poly_features(processed_df)
    prediction = float(model.predict(processed_df)[0])
    return round(prediction, 2)


# =============================================
# ENDPOINTS
# =============================================

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to DiabeSense+ API",
        "description": "AI-powered diabetes HbA1c prediction with explainable insights",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict - Get HbA1c prediction",
            "explain": "POST /explain - Get SHAP-based explanation",
            "whatif": "POST /whatif - What-If scenario analysis",
            "health": "GET /health - Check API health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_hba1c(data: PatientData):
    """Predict post-treatment HbA1c for a patient."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        input_df = prepare_input_dataframe(data)
        processed_array = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_array, columns=base_feature_names)
        processed_df = _apply_poly_features(processed_df)

        predicted_hba1c = float(model.predict(processed_df)[0])
        risk_level, confidence = get_risk_level(predicted_hba1c)

        # Clinical interpretation from reference sheet
        pre_hba1c = data.PreBLHBA1C
        age = data.PostBLAge
        delta = round(pre_hba1c - predicted_hba1c, 2)

        def _cat(v):
            if v < 5.7: return "Normoglycemia"
            elif v <= 6.4: return "Prediabetes"
            else: return "Diabetes"

        pre_cat = _cat(pre_hba1c)
        post_cat = _cat(predicted_hba1c)
        order = {"Normoglycemia": 0, "Prediabetes": 1, "Diabetes": 2}

        if order[post_cat] < order[pre_cat]: traj = "Regression"
        elif order[post_cat] == order[pre_cat]: traj = "Persistence"
        else: traj = "Progression"

        outcome_line = f"Predicted outcome: {pre_cat} → {post_cat} ({traj})"

        response_line = None
        if pre_cat == "Diabetes":
            if delta >= 1.0: response_line = f"Predicted response: Major improvement – Risk reduction achieved (ΔHbA1c {delta:+.2f}%)"
            elif delta >= 0.5: response_line = f"Predicted response: Clinically meaningful improvement (ΔHbA1c {delta:+.2f}%)"
            elif delta >= 0: response_line = f"Predicted response: Stabilization / modest improvement (ΔHbA1c {delta:+.2f}%)"
            else: response_line = f"Predicted response: Non-response (ΔHbA1c {delta:+.2f}%)"

        target_line = None
        if pre_cat == "Diabetes" and pre_hba1c > 7.0:
            target = 7.0 if age < 65 else 7.5
            achieved = "Achieved" if predicted_hba1c <= target else "Not achieved"
            target_line = f"Glycemic control target: ≤{target}% (age {int(age)}) | {achieved}"

        return PredictionResponse(
            predicted_hba1c=round(predicted_hba1c, 2),
            risk_level=risk_level,
            confidence=confidence,
            delta_hba1c=delta,
            outcome_line=outcome_line,
            response_line=response_line,
            target_line=target_line,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_prediction(data: PatientData):
    """Explain factors contributing to the HbA1c prediction using SHAP."""
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        input_df = prepare_input_dataframe(data)
        processed_array = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_array, columns=base_feature_names)
        processed_df = _apply_poly_features(processed_df)

        shap_values = explainer.shap_values(processed_df)

        # For regression, shap_values is a 2D array directly
        if isinstance(shap_values, list):
            vals = shap_values[0]
        else:
            vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        base_value = float(explainer.expected_value)
        if hasattr(explainer.expected_value, '__iter__'):
            base_value = float(explainer.expected_value[0])

        feature_impact = list(zip(feature_names, vals.tolist()))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = []
        for feat, impact in feature_impact[:5]:
            top_factors.append(FactorImpact(
                feature=feat,
                impact=round(float(impact), 4),
                direction="Increases HbA1c" if impact > 0 else "Reduces HbA1c",
                interpretation=interpret_feature(feat, impact)
            ))

        prediction_value = base_value + sum(vals)

        return ExplainResponse(
            top_contributing_factors=top_factors,
            base_value=round(base_value, 4),
            prediction_value=round(float(prediction_value), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# =============================================
# WHAT-IF ANALYSIS
# =============================================

class WhatIfScenario(BaseModel):
    scenario_id: int
    title: str
    description: str
    change_summary: str
    original_hba1c: float
    modified_hba1c: float
    hba1c_delta: float
    improvement_percent: float
    icon: str
    factor_changed: str
    original_value: str
    suggested_value: str


class WhatIfResponse(BaseModel):
    original_hba1c: float
    original_risk_level: str
    scenarios: List[WhatIfScenario]
    best_scenario: Optional[WhatIfScenario] = None
    combined_hba1c: Optional[float] = None
    combined_risk_level: Optional[str] = None


def _generate_scenarios(data: PatientData, shap_factors: list) -> list:
    """Generate what-if scenarios based on patient data."""
    original_dict = data.model_dump()
    scenarios = []

    scenario_definitions = [
        {
            "field": "PreRBMI",
            "condition": lambda d: d["PreRBMI"] > 25,
            "modify": lambda d: {**d, "PreRBMI": 22.0},
            "title": "Healthy BMI",
            "desc": lambda d: f"What if your BMI was healthy (22.0) instead of {d['PreRBMI']:.1f}?",
            "change": lambda d: f"BMI: {d['PreRBMI']:.1f} → 22.0",
            "icon": "⚖️",
            "orig_val": lambda d: f"{d['PreRBMI']:.1f}",
            "new_val": lambda d: "22.0",
        },
        {
            "field": "PreBLFBS",
            "condition": lambda d: d["PreBLFBS"] > 100,
            "modify": lambda d: {**d, "PreBLFBS": 90.0},
            "title": "Normal Fasting Sugar",
            "desc": lambda d: f"What if your fasting sugar was normal (90) instead of {d['PreBLFBS']:.0f} mg/dL?",
            "change": lambda d: f"FBS: {d['PreBLFBS']:.0f} → 90 mg/dL",
            "icon": "🩸",
            "orig_val": lambda d: f"{d['PreBLFBS']:.0f} mg/dL",
            "new_val": lambda d: "90 mg/dL",
        },
        {
            "field": "PreBLPPBS",
            "condition": lambda d: d["PreBLPPBS"] > 140,
            "modify": lambda d: {**d, "PreBLPPBS": 130.0},
            "title": "Normal Post-Prandial Sugar",
            "desc": lambda d: f"What if your PPBS was normal (130) instead of {d['PreBLPPBS']:.0f} mg/dL?",
            "change": lambda d: f"PPBS: {d['PreBLPPBS']:.0f} → 130 mg/dL",
            "icon": "🍽️",
            "orig_val": lambda d: f"{d['PreBLPPBS']:.0f} mg/dL",
            "new_val": lambda d: "130 mg/dL",
        },
        {
            "field": "PreBLHBA1C",
            "condition": lambda d: d["PreBLHBA1C"] > 6.5,
            "modify": lambda d: {**d, "PreBLHBA1C": 5.7},
            "title": "Normal Pre-Treatment HbA1c",
            "desc": lambda d: f"What if your starting HbA1c was near-normal (5.7%) instead of {d['PreBLHBA1C']:.1f}%?",
            "change": lambda d: f"HbA1c: {d['PreBLHBA1C']:.1f}% → 5.7%",
            "icon": "📉",
            "orig_val": lambda d: f"{d['PreBLHBA1C']:.1f}%",
            "new_val": lambda d: "5.7%",
        },
        {
            "field": "current_smoking",
            "condition": lambda d: d["current_smoking"] == 1,
            "modify": lambda d: {**d, "current_smoking": 0},
            "title": "Quit Smoking",
            "desc": lambda d: "What if you quit smoking?",
            "change": lambda d: "Smoking: Yes → No",
            "icon": "🚭",
            "orig_val": lambda d: "Yes",
            "new_val": lambda d: "No",
        },
        {
            "field": "current_alcohol",
            "condition": lambda d: d["current_alcohol"] == 1,
            "modify": lambda d: {**d, "current_alcohol": 0},
            "title": "Quit Alcohol",
            "desc": lambda d: "What if you stopped alcohol consumption?",
            "change": lambda d: "Alcohol: Yes → No",
            "icon": "🚫",
            "orig_val": lambda d: "Yes",
            "new_val": lambda d: "No",
        },
        {
            "field": "PreRsystolicfirst",
            "condition": lambda d: d["PreRsystolicfirst"] > 130,
            "modify": lambda d: {**d, "PreRsystolicfirst": 120.0, "PreRdiastolicfirst": 80.0},
            "title": "Normal Blood Pressure",
            "desc": lambda d: f"What if your BP was normal (120/80) instead of {d['PreRsystolicfirst']:.0f}/{d['PreRdiastolicfirst']:.0f}?",
            "change": lambda d: f"BP: {d['PreRsystolicfirst']:.0f}/{d['PreRdiastolicfirst']:.0f} → 120/80",
            "icon": "💊",
            "orig_val": lambda d: f"{d['PreRsystolicfirst']:.0f}/{d['PreRdiastolicfirst']:.0f}",
            "new_val": lambda d: "120/80 mmHg",
        },
        {
            "field": "PreBLCHOLESTEROL",
            "condition": lambda d: d["PreBLCHOLESTEROL"] > 200,
            "modify": lambda d: {**d, "PreBLCHOLESTEROL": 180.0},
            "title": "Normal Cholesterol",
            "desc": lambda d: f"What if your cholesterol was under control (180) instead of {d['PreBLCHOLESTEROL']:.0f}?",
            "change": lambda d: f"Cholesterol: {d['PreBLCHOLESTEROL']:.0f} → 180 mg/dL",
            "icon": "❤️",
            "orig_val": lambda d: f"{d['PreBLCHOLESTEROL']:.0f} mg/dL",
            "new_val": lambda d: "180 mg/dL",
        },
        {
            "field": "PreBLTRIGLYCERIDES",
            "condition": lambda d: d["PreBLTRIGLYCERIDES"] > 150,
            "modify": lambda d: {**d, "PreBLTRIGLYCERIDES": 130.0},
            "title": "Normal Triglycerides",
            "desc": lambda d: f"What if your triglycerides were normal (130) instead of {d['PreBLTRIGLYCERIDES']:.0f}?",
            "change": lambda d: f"Triglycerides: {d['PreBLTRIGLYCERIDES']:.0f} → 130 mg/dL",
            "icon": "🧪",
            "orig_val": lambda d: f"{d['PreBLTRIGLYCERIDES']:.0f} mg/dL",
            "new_val": lambda d: "130 mg/dL",
        },
        {
            "field": "PostRgroupname",
            "condition": lambda d: d["PostRgroupname"] == 2,
            "modify": lambda d: {**d, "PostRgroupname": 1},
            "title": "Join Yoga Intervention",
            "desc": lambda d: "What if you joined the yoga intervention group instead of control?",
            "change": lambda d: "Group: Control → Yoga",
            "icon": "🧘",
            "orig_val": lambda d: "Control",
            "new_val": lambda d: "Yoga",
        },
        {
            "field": "PreRsleepquality",
            "condition": lambda d: d["PreRsleepquality"] > 2,
            "modify": lambda d: {**d, "PreRsleepquality": 1.0},
            "title": "Improve Sleep Quality",
            "desc": lambda d: "What if your sleep quality improved to good?",
            "change": lambda d: f"Sleep: {d['PreRsleepquality']:.0f} → 1 (Good)",
            "icon": "😴",
            "orig_val": lambda d: f"{d['PreRsleepquality']:.0f} (Poor)",
            "new_val": lambda d: "1 (Good)",
        },
    ]

    scenario_id = 1
    for defn in scenario_definitions:
        if defn["condition"](original_dict):
            modified_dict = defn["modify"](original_dict)
            modified_hba1c = _predict_hba1c(modified_dict)

            scenarios.append({
                "scenario_id": scenario_id,
                "title": defn["title"],
                "description": defn["desc"](original_dict),
                "change_summary": defn["change"](original_dict),
                "icon": defn["icon"],
                "factor_changed": defn["field"],
                "original_value": defn["orig_val"](original_dict),
                "suggested_value": defn["new_val"](original_dict),
                "modified_hba1c": modified_hba1c,
            })
            scenario_id += 1

    return scenarios


@app.post("/whatif", response_model=WhatIfResponse, tags=["What-If Analysis"])
async def whatif_analysis(data: PatientData):
    """Generate What-If counterfactual scenarios for diabetes management."""
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        original_hba1c = _predict_hba1c(data.model_dump())
        original_risk_level, _ = get_risk_level(original_hba1c)

        # Get SHAP factors
        input_df = prepare_input_dataframe(data)
        processed_array = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_array, columns=feature_names)
        shap_values = explainer.shap_values(processed_df)

        if isinstance(shap_values, list):
            vals = shap_values[0]
        else:
            vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        shap_factors = [
            {"feature": fname, "impact": float(val)}
            for fname, val in zip(feature_names, vals.tolist())
        ]

        raw_scenarios = _generate_scenarios(data, shap_factors)

        scenarios = []
        for s in raw_scenarios:
            delta = original_hba1c - s["modified_hba1c"]
            improvement = (delta / original_hba1c * 100) if original_hba1c > 0 else 0

            scenarios.append(WhatIfScenario(
                scenario_id=s["scenario_id"],
                title=s["title"],
                description=s["description"],
                change_summary=s["change_summary"],
                original_hba1c=original_hba1c,
                modified_hba1c=s["modified_hba1c"],
                hba1c_delta=round(delta, 2),
                improvement_percent=round(improvement, 2),
                icon=s["icon"],
                factor_changed=s["factor_changed"],
                original_value=s["original_value"],
                suggested_value=s["suggested_value"],
            ))

        scenarios.sort(key=lambda x: x.hba1c_delta, reverse=True)
        best = scenarios[0] if scenarios else None

        # Combined scenario
        combined_hba1c = None
        combined_risk_level = None
        if len(scenarios) > 1:
            combined_dict = data.model_dump()
            for s in raw_scenarios:
                field = s["factor_changed"]
                if field == "PreRBMI":
                    combined_dict["PreRBMI"] = 22.0
                elif field == "PreBLFBS":
                    combined_dict["PreBLFBS"] = 90.0
                elif field == "PreBLPPBS":
                    combined_dict["PreBLPPBS"] = 130.0
                elif field == "PreBLHBA1C":
                    combined_dict["PreBLHBA1C"] = 5.7
                elif field == "current_smoking":
                    combined_dict["current_smoking"] = 0
                elif field == "current_alcohol":
                    combined_dict["current_alcohol"] = 0
                elif field == "PreRsystolicfirst":
                    combined_dict["PreRsystolicfirst"] = 120.0
                    combined_dict["PreRdiastolicfirst"] = 80.0
                elif field == "PreBLCHOLESTEROL":
                    combined_dict["PreBLCHOLESTEROL"] = 180.0
                elif field == "PreBLTRIGLYCERIDES":
                    combined_dict["PreBLTRIGLYCERIDES"] = 130.0
                elif field == "PostRgroupname":
                    combined_dict["PostRgroupname"] = 1
                elif field == "PreRsleepquality":
                    combined_dict["PreRsleepquality"] = 1.0

            combined_hba1c = _predict_hba1c(combined_dict)
            combined_risk_level, _ = get_risk_level(combined_hba1c)

        return WhatIfResponse(
            original_hba1c=original_hba1c,
            original_risk_level=original_risk_level,
            scenarios=scenarios,
            best_scenario=best,
            combined_hba1c=combined_hba1c,
            combined_risk_level=combined_risk_level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If analysis error: {str(e)}")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(patients: List[PatientData]):
    """Predict HbA1c for multiple patients."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        results = []
        for patient in patients:
            input_df = prepare_input_dataframe(patient)
            processed_array = preprocessor.transform(input_df)
            processed_df = pd.DataFrame(processed_array, columns=base_feature_names)
            processed_df = _apply_poly_features(processed_df)

            predicted = float(model.predict(processed_df)[0])
            risk_level, confidence = get_risk_level(predicted)

            results.append({
                "patient_data": patient.model_dump(),
                "predicted_hba1c": round(predicted, 2),
                "risk_level": risk_level,
                "confidence": confidence
            })

        return {"results": results, "total_patients": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# =============================================
# AI CHAT ENDPOINTS
# =============================================

class ChatStartRequest(BaseModel):
    patient_data: dict
    prediction: dict
    explanation: dict
    whatif: dict


class ChatStartResponse(BaseModel):
    session_id: str
    message: str


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    response: str


@app.post("/chat/start", response_model=ChatStartResponse, tags=["Chat"])
async def api_chat_start(req: ChatStartRequest):
    if not CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chat not available. Install: pip install agno sqlalchemy")
    try:
        session_id, greeting = start_chat_session(
            req.patient_data, req.prediction, req.explanation, req.whatif
        )
        return ChatStartResponse(session_id=session_id, message=greeting)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat start error: {str(e)}")


@app.post("/chat/message", response_model=ChatMessageResponse, tags=["Chat"])
async def api_chat_message(req: ChatMessageRequest):
    if not CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chat not available. Install: pip install agno sqlalchemy")
    try:
        response_text = get_chat_response(req.session_id, req.message)
        return ChatMessageResponse(response=response_text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
