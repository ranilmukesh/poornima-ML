import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os

# Graceful LLM import — chat feature is optional
# Predictions work fine even without agno installed
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
    title="CardioSense+ API",
    description="AI-powered stroke risk prediction with explainable insights",
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

ARTIFACTS_PATH = "cardiosense_artifacts.pkl"

model = None
preprocessor = None
feature_names = None
explainer = None


def load_artifacts():
    """Load saved model artifacts."""
    global model, preprocessor, feature_names, explainer
    
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
    
    # Initialize SHAP TreeExplainer (optimized for tree-based models)
    print("[*] Initializing SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    
    print("[OK] All artifacts loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load model artifacts when the API starts."""
    try:
        load_artifacts()
    except FileNotFoundError as e:
        print(f"[!] Warning: {e}")
        print("    The API will start, but predictions will fail until artifacts are available.")


class PatientData(BaseModel):
    """Input schema for patient health data."""
    gender: str = Field(..., description="Patient gender: 'Male' or 'Female'")
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    hypertension: int = Field(..., ge=0, le=1, description="1 if patient has hypertension, 0 otherwise")
    heart_disease: int = Field(..., ge=0, le=1, description="1 if patient has heart disease, 0 otherwise")
    ever_married: str = Field(..., description="'Yes' or 'No'")
    work_type: str = Field(..., description="Type of work: 'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'")
    Residence_type: str = Field(..., description="Residence type: 'Urban' or 'Rural'")
    avg_glucose_level: float = Field(..., ge=0, description="Average glucose level in blood")
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status: 'formerly smoked', 'never smoked', 'smokes', 'Unknown'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 67.0,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: int = Field(..., description="0 = No Stroke, 1 = Stroke")
    probability_percentage: float = Field(..., description="Probability of stroke as percentage")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    confidence: str = Field(..., description="Model confidence level")


class FactorImpact(BaseModel):
    """Schema for individual factor impact."""
    feature: str
    impact: float
    direction: str
    interpretation: str


class ExplainResponse(BaseModel):
    """Output schema for explainability results."""
    top_contributing_factors: List[FactorImpact]
    base_value: float
    prediction_value: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str



def prepare_input_dataframe(data: PatientData) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame for preprocessing."""
    input_dict = data.model_dump()
    return pd.DataFrame([input_dict])


def get_risk_level(probability: float) -> tuple:
    """Determine risk level and confidence based on probability."""
    if probability >= 0.7:
        return "HIGH", "Very High Confidence"
    elif probability >= 0.5:
        return "HIGH", "High Confidence"
    elif probability >= 0.3:
        return "MEDIUM", "Moderate Confidence"
    elif probability >= 0.15:
        return "LOW", "Moderate Confidence"
    else:
        return "LOW", "High Confidence"


def interpret_feature(feature_name: str, impact: float) -> str:
    """Generate human-readable interpretation for a feature's impact."""
    clean_name = feature_name.replace('_', ' ').title()
    
    if '_' in feature_name and any(x in feature_name.lower() for x in 
                                    ['gender', 'work_type', 'residence', 'married', 'smoking']):
        parts = feature_name.split('_', 1)
        if len(parts) == 2:
            category, value = parts[0], parts[1]
            if impact > 0:
                return f"Being '{value}' for {category} increases stroke risk"
            else:
                return f"Being '{value}' for {category} decreases stroke risk"
    
    abs_impact = abs(impact)
    intensity = "significantly" if abs_impact > 0.3 else ("moderately" if abs_impact > 0.1 else "slightly")
    
    if impact > 0:
        return f"{clean_name} {intensity} increases stroke risk"
    else:
        return f"{clean_name} {intensity} decreases stroke risk"


@app.get("/", tags=["General"])
async def root():
    """API welcome endpoint."""
    return {
        "message": "Welcome to CardioSense+ API",
        "description": "AI-powered stroke risk prediction with explainable insights",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict - Get stroke risk prediction",
            "explain": "POST /explain - Get SHAP-based explanation",
            "health": "GET /health - Check API health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_stroke_risk(data: PatientData):
    """
    Predict stroke risk for a patient.
    
    Returns:
    - prediction: Binary classification (0 or 1)
    - probability_percentage: Probability of stroke (0-100%)
    - risk_level: Categorical risk level (LOW/MEDIUM/HIGH)
    - confidence: Model confidence in the prediction
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train_model.py' first."
        )
    
    try:
        input_df = prepare_input_dataframe(data)
        
        processed_array = preprocessor.transform(input_df)
        
        processed_df = pd.DataFrame(processed_array, columns=feature_names)
        
        prediction = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])
        
        risk_level, confidence = get_risk_level(probability)
        
        return PredictionResponse(
            prediction=prediction,
            probability_percentage=round(probability * 100, 2),
            risk_level=risk_level,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_risk(data: PatientData):
    """
    Explain the factors contributing to the stroke risk prediction.
    
    Uses SHAP (SHapley Additive exPlanations) to provide:
    - Top contributing factors for this specific patient
    - Direction of impact (increases/decreases risk)
    - Human-readable interpretations
    """
    # Check if model and explainer are loaded
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train_model.py' first."
        )
    
    try:
        input_df = prepare_input_dataframe(data)
        
        processed_array = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_array, columns=feature_names)
        
        shap_values = explainer.shap_values(processed_df)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            vals = shap_values[0]
        
        if hasattr(explainer.expected_value, '__iter__'):
            base_value = float(explainer.expected_value[1]) if len(explainer.expected_value) > 1 else float(explainer.expected_value[0])
        else:
            base_value = float(explainer.expected_value)
        
        feature_impact = list(zip(feature_names, vals.tolist()))
        
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for feature, impact in feature_impact[:5]:
            factor = FactorImpact(
                feature=feature,
                impact=round(float(impact), 4),
                direction="Increases Risk" if impact > 0 else "Reduces Risk",
                interpretation=interpret_feature(feature, impact)
            )
            top_factors.append(factor)
        
        # Calculate prediction value from SHAP
        prediction_value = base_value + sum(vals)
        
        return ExplainResponse(
            top_contributing_factors=top_factors,
            base_value=round(base_value, 4),
            prediction_value=round(float(prediction_value), 4)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation error: {str(e)}"
        )


class WhatIfScenario(BaseModel):
    """Schema for a single what-if scenario."""
    scenario_id: int
    title: str
    description: str
    change_summary: str
    original_risk: float
    modified_risk: float
    risk_delta: float
    risk_reduction_percent: float
    icon: str
    factor_changed: str
    original_value: str
    suggested_value: str


class WhatIfResponse(BaseModel):
    """Schema for what-if analysis results."""
    original_risk: float
    original_risk_level: str
    scenarios: List[WhatIfScenario]
    best_scenario: Optional[WhatIfScenario] = None
    combined_risk: Optional[float] = None
    combined_risk_level: Optional[str] = None


def _predict_risk(patient_data: dict) -> float:
    """Internal helper to predict risk probability from a patient dict."""
    input_df = pd.DataFrame([patient_data])
    processed_array = preprocessor.transform(input_df)
    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    probability = float(model.predict_proba(processed_df)[0][1])
    return round(probability * 100, 2)


def _generate_scenarios(data: PatientData, shap_factors: list) -> list:
    """
    Generate what-if scenarios based on SHAP factors and patient data.
    Returns a list of scenario dicts with modifications.
    """
    original_dict = data.model_dump()
    scenarios = []
    
    # Map from encoded feature names back to original fields + suggested changes
    # Each entry: (encoded_feature_prefix, original_field, condition_fn, modification, title, description, icon)
    scenario_definitions = [
        {
            "features": ["age"],
            "field": "age",
            "condition": lambda d: d["age"] > 50,
            "modify": lambda d: {**d, "age": max(d["age"] - 10, 30)},
            "title": "10 Years Younger",
            "desc": lambda d: f"What if you were {max(d['age'] - 10, 30):.0f} years old instead of {d['age']:.0f}?",
            "change": lambda d: f"Age: {d['age']:.0f} → {max(d['age'] - 10, 30):.0f}",
            "icon": "⏪",
            "orig_val": lambda d: f"{d['age']:.0f} years",
            "new_val": lambda d: f"{max(d['age'] - 10, 30):.0f} years",
        },
        {
            "features": ["avg_glucose_level"],
            "field": "avg_glucose_level",
            "condition": lambda d: d["avg_glucose_level"] > 100,
            "modify": lambda d: {**d, "avg_glucose_level": 90.0},
            "title": "Normal Glucose Level",
            "desc": lambda d: f"What if your glucose was normal (90 mg/dL) instead of {d['avg_glucose_level']:.1f} mg/dL?",
            "change": lambda d: f"Glucose: {d['avg_glucose_level']:.1f} → 90.0 mg/dL",
            "icon": "🩸",
            "orig_val": lambda d: f"{d['avg_glucose_level']:.1f} mg/dL",
            "new_val": lambda d: "90.0 mg/dL",
        },
        {
            "features": ["bmi"],
            "field": "bmi",
            "condition": lambda d: d["bmi"] > 25,
            "modify": lambda d: {**d, "bmi": 22.0},
            "title": "Healthy BMI",
            "desc": lambda d: f"What if your BMI was healthy (22.0) instead of {d['bmi']:.1f}?",
            "change": lambda d: f"BMI: {d['bmi']:.1f} → 22.0",
            "icon": "⚖️",
            "orig_val": lambda d: f"{d['bmi']:.1f}",
            "new_val": lambda d: "22.0",
        },
        {
            "features": ["hypertension"],
            "field": "hypertension",
            "condition": lambda d: d["hypertension"] == 1,
            "modify": lambda d: {**d, "hypertension": 0},
            "title": "No Hypertension",
            "desc": lambda d: "What if you didn't have hypertension?",
            "change": lambda d: "Hypertension: Yes → No",
            "icon": "💊",
            "orig_val": lambda d: "Yes",
            "new_val": lambda d: "No",
        },
        {
            "features": ["heart_disease"],
            "field": "heart_disease",
            "condition": lambda d: d["heart_disease"] == 1,
            "modify": lambda d: {**d, "heart_disease": 0},
            "title": "No Heart Disease",
            "desc": lambda d: "What if you didn't have heart disease?",
            "change": lambda d: "Heart Disease: Yes → No",
            "icon": "❤️",
            "orig_val": lambda d: "Yes",
            "new_val": lambda d: "No",
        },
        {
            "features": ["smoking_status_smokes", "smoking_status_formerly smoked"],
            "field": "smoking_status",
            "condition": lambda d: d["smoking_status"] in ["smokes", "formerly smoked"],
            "modify": lambda d: {**d, "smoking_status": "never smoked"},
            "title": "Never Smoked",
            "desc": lambda d: f"What if you never smoked (currently: {d['smoking_status']})?",
            "change": lambda d: f"Smoking: {d['smoking_status']} → never smoked",
            "icon": "🚭",
            "orig_val": lambda d: d["smoking_status"],
            "new_val": lambda d: "never smoked",
        },
    ]
    
    # Get the SHAP feature names that are risk-increasing (positive SHAP)
    risk_features = set()
    for factor in shap_factors:
        if factor["impact"] > 0:
            risk_features.add(factor["feature"])
    
    scenario_id = 1
    for defn in scenario_definitions:
        # Check if any of the encoded features for this scenario are in the risk-increasing set
        # OR if the condition is met (relevant for the patient)
        feature_relevant = any(f in risk_features for f in defn["features"])
        condition_met = defn["condition"](original_dict)
        
        if condition_met:  # Only show scenarios where the change is applicable
            modified_dict = defn["modify"](original_dict)
            modified_risk = _predict_risk(modified_dict)
            
            scenarios.append({
                "scenario_id": scenario_id,
                "title": defn["title"],
                "description": defn["desc"](original_dict),
                "change_summary": defn["change"](original_dict),
                "icon": defn["icon"],
                "factor_changed": defn["field"],
                "original_value": defn["orig_val"](original_dict),
                "suggested_value": defn["new_val"](original_dict),
                "modified_risk": modified_risk,
                "shap_relevant": feature_relevant,
            })
            scenario_id += 1
    
    return scenarios


@app.post("/whatif", response_model=WhatIfResponse, tags=["What-If Analysis"])
async def whatif_analysis(data: PatientData):
    """
    Generate What-If counterfactual scenarios for a patient.
    
    Automatically identifies modifiable risk factors and shows how
    changing each factor would impact the stroke risk prediction.
    This provides actionable insights for risk reduction.
    """
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train_model.py' first."
        )
    
    try:
        # Step 1: Get original prediction
        original_risk = _predict_risk(data.model_dump())
        original_risk_level, _ = get_risk_level(original_risk / 100)
        
        # Step 2: Get SHAP factors for this patient
        input_df = prepare_input_dataframe(data)
        processed_array = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed_array, columns=feature_names)
        shap_values = explainer.shap_values(processed_df)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            vals = shap_values[0]
        
        # Build list of factor dicts
        shap_factors = [
            {"feature": fname, "impact": float(val)}
            for fname, val in zip(feature_names, vals.tolist())
        ]
        shap_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        # Step 3: Generate scenarios
        raw_scenarios = _generate_scenarios(data, shap_factors)
        
        # Step 4: Build response scenarios
        scenarios = []
        for s in raw_scenarios:
            delta = original_risk - s["modified_risk"]
            reduction_pct = (delta / original_risk * 100) if original_risk > 0 else 0
            
            scenarios.append(WhatIfScenario(
                scenario_id=s["scenario_id"],
                title=s["title"],
                description=s["description"],
                change_summary=s["change_summary"],
                original_risk=original_risk,
                modified_risk=s["modified_risk"],
                risk_delta=round(delta, 2),
                risk_reduction_percent=round(reduction_pct, 2),
                icon=s["icon"],
                factor_changed=s["factor_changed"],
                original_value=s["original_value"],
                suggested_value=s["suggested_value"],
            ))
        
        # Sort by biggest risk reduction
        scenarios.sort(key=lambda x: x.risk_delta, reverse=True)
        
        # Step 5: Best scenario
        best = scenarios[0] if scenarios else None
        
        # Step 6: Combined scenario (apply ALL modifications at once)
        combined_risk = None
        combined_risk_level = None
        if len(scenarios) > 1:
            combined_dict = data.model_dump()
            for s in raw_scenarios:
                # Apply each modification
                field = s["factor_changed"]
                if field == "age":
                    combined_dict["age"] = max(combined_dict["age"] - 10, 30)
                elif field == "avg_glucose_level":
                    combined_dict["avg_glucose_level"] = 90.0
                elif field == "bmi":
                    combined_dict["bmi"] = 22.0
                elif field == "hypertension":
                    combined_dict["hypertension"] = 0
                elif field == "heart_disease":
                    combined_dict["heart_disease"] = 0
                elif field == "smoking_status":
                    combined_dict["smoking_status"] = "never smoked"
            
            combined_risk = _predict_risk(combined_dict)
            combined_risk_level, _ = get_risk_level(combined_risk / 100)
        
        return WhatIfResponse(
            original_risk=original_risk,
            original_risk_level=original_risk_level,
            scenarios=scenarios,
            best_scenario=best,
            combined_risk=combined_risk,
            combined_risk_level=combined_risk_level,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"What-If analysis error: {str(e)}"
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(patients: List[PatientData]):
    """
    Predict stroke risk for multiple patients at once.
    
    Useful for batch processing or screening programs.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train_model.py' first."
        )
    
    try:
        results = []
        for patient in patients:
            # Reuse single prediction logic
            input_df = prepare_input_dataframe(patient)
            processed_array = preprocessor.transform(input_df)
            processed_df = pd.DataFrame(processed_array, columns=feature_names)
            
            prediction = int(model.predict(processed_df)[0])
            probability = float(model.predict_proba(processed_df)[0][1])
            risk_level, confidence = get_risk_level(probability)
            
            results.append({
                "patient_data": patient.model_dump(),
                "prediction": prediction,
                "probability_percentage": round(probability * 100, 2),
                "risk_level": risk_level,
                "confidence": confidence
            })
        
        return {"results": results, "total_patients": len(results)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


# =============================================
# AI CHAT ENDPOINTS
# =============================================

class ChatStartRequest(BaseModel):
    """Request to start a new chat session with patient context."""
    patient_data: dict
    prediction: dict
    explanation: dict
    whatif: dict


class ChatStartResponse(BaseModel):
    session_id: str
    message: str


class ChatMessageRequest(BaseModel):
    """Request to send a message in an existing chat session."""
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    response: str


@app.post("/chat/start", response_model=ChatStartResponse, tags=["Chat"])
async def api_chat_start(req: ChatStartRequest):
    """
    Start a new AI chat session with patient context injected.
    Returns a session ID and an auto-generated greeting.
    """
    if not CHAT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Chat not available. Install: pip install agno sqlalchemy"
        )

    try:
        session_id, greeting = start_chat_session(
            req.patient_data, req.prediction, req.explanation, req.whatif
        )
        return ChatStartResponse(session_id=session_id, message=greeting)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat start error: {str(e)}")


@app.post("/chat/message", response_model=ChatMessageResponse, tags=["Chat"])
async def api_chat_message(req: ChatMessageRequest):
    """
    Send a message in an existing chat session.
    The agent has full context of the patient's assessment.
    """
    if not CHAT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Chat not available. Install: pip install agno sqlalchemy"
        )

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
