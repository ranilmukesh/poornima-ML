"""
DiabeSense+ LLM Chat Module
AI-powered health chat using Agno SDK + Nvidia LLM
Context Injection pattern: patient data is injected into the system prompt.
"""

import os
import uuid

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"[LLM] Loaded .env from: {env_path}")
except ImportError:
    print("[LLM] python-dotenv not installed, relying on env var directly")

_key = os.environ.get("NVIDIA_API_KEY", "")
if _key:
    print(f"[LLM] NVIDIA_API_KEY found: {_key[:12]}...{_key[-4:]}")
else:
    print("[LLM] WARNING: NVIDIA_API_KEY is NOT set!")

from agno.agent import Agent
from agno.models.nvidia import Nvidia
from agno.db.sqlite import SqliteDb

os.makedirs("tmp", exist_ok=True)
_chat_db = SqliteDb(db_file="tmp/diabesense_chat.db")
_sessions: dict = {}


def build_system_context(
    patient_data: dict,
    prediction: dict,
    explanation: dict,
    whatif: dict,
) -> str:
    """Build structured context string from patient analysis data."""

    # ── Patient Profile ──
    patient_lines = [
        f"Age: {patient_data.get('PostBLAge', 'N/A')} years",
        f"Gender: {patient_data.get('PreBLGender', 'N/A')}",
        f"Area: {'Urban' if patient_data.get('PreRarea') == 1 else 'Rural' if patient_data.get('PreRarea') == 2 else 'N/A'}",
        f"Marital Status: {patient_data.get('PreRmaritalstatus', 'N/A')}",
        f"Education Level: {patient_data.get('PreReducation', 'N/A')}",
        f"Occupation: {patient_data.get('PreRpresentoccupation', 'N/A')}",
    ]

    family_lines = [
        f"Father Diabetic: {'Yes' if patient_data.get('PreRdiafather') else 'No'}",
        f"Mother Diabetic: {'Yes' if patient_data.get('PreRdiamother') else 'No'}",
        f"Brother Diabetic: {'Yes' if patient_data.get('PreRdiabrother') else 'No'}",
        f"Sister Diabetic: {'Yes' if patient_data.get('PreRdiasister') else 'No'}",
    ]

    lifestyle_lines = [
        f"Current Smoking: {'Yes' if patient_data.get('current_smoking') else 'No'}",
        f"Current Alcohol: {'Yes' if patient_data.get('current_alcohol') else 'No'}",
        f"Sleep Quality: {patient_data.get('PreRsleepquality', 'N/A')} (1=Good, 4=Poor)",
    ]

    health_lines = [
        f"Waist: {patient_data.get('PreRwaist', 'N/A')} cm",
        f"BMI: {patient_data.get('PreRBMI', 'N/A')}",
        f"Systolic BP: {patient_data.get('PreRsystolicfirst', 'N/A')} mmHg",
        f"Diastolic BP: {patient_data.get('PreRdiastolicfirst', 'N/A')} mmHg",
    ]

    blood_lines = [
        f"Fasting Blood Sugar (FBS): {patient_data.get('PreBLFBS', 'N/A')} mg/dL",
        f"Post-Prandial Blood Sugar (PPBS): {patient_data.get('PreBLPPBS', 'N/A')} mg/dL",
        f"Pre-Treatment HbA1c: {patient_data.get('PreBLHBA1C', 'N/A')}%",
        f"Cholesterol: {patient_data.get('PreBLCHOLESTEROL', 'N/A')} mg/dL",
        f"Triglycerides: {patient_data.get('PreBLTRIGLYCERIDES', 'N/A')} mg/dL",
        f"Diabetic Duration: {patient_data.get('Diabetic_Duration', 'N/A')} months",
        f"Intervention Group: {'Yoga' if patient_data.get('PostRgroupname') == 1 else 'Control' if patient_data.get('PostRgroupname') == 2 else 'N/A'}",
    ]

    patient_block = "\n".join(f"  - {l}" for l in patient_lines)
    family_block = "\n".join(f"  - {l}" for l in family_lines)
    lifestyle_block = "\n".join(f"  - {l}" for l in lifestyle_lines)
    health_block = "\n".join(f"  - {l}" for l in health_lines)
    blood_block = "\n".join(f"  - {l}" for l in blood_lines)

    # ── Prediction ──
    pred_block = (
        f"  - Predicted Post-Treatment HbA1c: {prediction.get('predicted_hba1c', '?')}%\n"
        f"  - Risk Level: {prediction.get('risk_level', '?')}\n"
        f"  - Confidence: {prediction.get('confidence', '?')}"
    )

    # ── SHAP Factors ──
    factors = explanation.get("top_contributing_factors", [])
    factor_lines = []
    for f in factors:
        factor_lines.append(
            f"  - {f.get('feature', '?')} | {f.get('direction', '?')} | "
            f"{f.get('interpretation', '')}"
        )
    shap_block = "\n".join(factor_lines) if factor_lines else "  (None available)"

    # ── What-If Scenarios ──
    scenarios = whatif.get("scenarios", [])
    scenario_lines = []
    for s in scenarios:
        scenario_lines.append(
            f"  - {s.get('title', '?')}: "
            f"{s.get('original_hba1c', '?')} -> {s.get('modified_hba1c', '?')} "
            f"(delta: {s.get('hba1c_delta', 0):+.2f}) | {s.get('description', '')}"
        )
    whatif_block = "\n".join(scenario_lines) if scenario_lines else "  (None generated)"

    combined = whatif.get("combined_hba1c")
    combined_line = ""
    if combined is not None:
        combined_line = (
            f"\n  BEST COMBINED OUTCOME (all changes): "
            f"HbA1c {combined}% ({whatif.get('combined_risk_level', '?')})"
        )

    return (
        "=== PATIENT DIABETES ASSESSMENT DATA ===\n\n"
        f"PATIENT PROFILE:\n{patient_block}\n\n"
        f"FAMILY DIABETES HISTORY:\n{family_block}\n\n"
        f"LIFESTYLE:\n{lifestyle_block}\n\n"
        f"HEALTH METRICS:\n{health_block}\n\n"
        f"BLOOD WORK:\n{blood_block}\n\n"
        f"PREDICTED OUTCOME:\n{pred_block}\n\n"
        f"TOP CONTRIBUTING FACTORS (SHAP):\n{shap_block}\n\n"
        f"WHAT-IF SCENARIOS:\n{whatif_block}{combined_line}\n\n"
        "========================================"
    )


def start_chat_session(
    patient_data: dict,
    prediction: dict,
    explanation: dict,
    whatif: dict,
) -> tuple:
    """Start a new chat session with patient context injected."""
    session_id = f"ds-{uuid.uuid4().hex[:8]}"
    system_context = build_system_context(patient_data, prediction, explanation, whatif)

    agent = Agent(
        model=Nvidia(
            max_tokens=16384,
            temperature=0.1,
            top_p=0.95,
            id="nvidia/nemotron-3-super-120b-a12b"
        ),
        description=(
            "You are DiabeSense AI, a warm and empathetic diabetes health assistant "
            "built into the DiabeSense+ HbA1c prediction platform. "
            "A patient has just completed their diabetes assessment. "
            "Their full data and results are below.\n\n"
            + system_context
        ),
        instructions=[
            "Speak in a warm, empathetic, and professional tone.",
            "Reference the patient's specific data when answering.",
            "Explain medical terms (HbA1c, FBS, PPBS, BMI, triglycerides) in plain language.",
            "When discussing What-If scenarios, give practical lifestyle tips for diabetes management.",
            "If predicted HbA1c is HIGH, be reassuring but honest about the importance of management.",
            "Always remind them you are an AI, not a doctor.",
            "Recommend consulting an endocrinologist or healthcare professional for decisions.",
            "Keep responses concise (2-3 paragraphs) unless asked for more.",
            "Never fabricate data — only reference the assessment data above.",
            "If the user seems anxious, de-escalate with empathy first.",
            "Discuss the role of yoga, exercise, and diet in diabetes management when relevant.",
        ],
        expected_output=(
            "Clear, empathetic, personalized diabetes health guidance based on "
            "the patient's specific assessment data, with practical next steps."
        ),
        db=_chat_db,
        add_history_to_context=True,
        num_history_runs=5,
        add_datetime_to_context=True,
    )

    _sessions[session_id] = agent

    greeting_prompt = (
        "The patient has just seen their predicted post-treatment HbA1c results. "
        "Introduce yourself in 1 sentence, then give a brief empathetic "
        "summary of their key findings (predicted HbA1c, top 2 factors). "
        "End by asking what they'd like to discuss. Keep it concise."
    )
    response = agent.run(greeting_prompt, session_id=session_id)

    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content:
        raise ConnectionError(f"LLM API Error: {response.content}")

    return session_id, response.content


def get_chat_response(session_id: str, user_message: str) -> str:
    """Get a response in an existing session."""
    agent = _sessions.get(session_id)
    if agent is None:
        raise ValueError(f"Session '{session_id}' not found or expired.")

    response = agent.run(user_message, session_id=session_id)

    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content:
        raise ConnectionError(f"LLM API Error: {response.content}")

    return response.content