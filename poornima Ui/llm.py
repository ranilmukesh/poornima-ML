"""
CardioSense+ LLM Chat Module
AI-powered health chat using Agno SDK + Nvidia LLM
Context Injection pattern: patient data is injected into the system prompt.
"""

import os
import uuid

# Load .env file written by start_cardiosense.bat
# override=True ensures .env always wins (fixes Windows subprocess env inheritance issues)
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"[LLM] Loaded .env from: {env_path}")
except ImportError:
    print("[LLM] python-dotenv not installed, relying on env var directly")

# Debug: check if key is present
_key = os.environ.get("NVIDIA_API_KEY", "")
if _key:
    print(f"[LLM] NVIDIA_API_KEY found: {_key[:12]}...{_key[-4:]}")
else:
    print("[LLM] WARNING: NVIDIA_API_KEY is NOT set!")

from agno.agent import Agent
from agno.models.nvidia import Nvidia
from agno.db.sqlite import SqliteDb

# Ensure tmp directory exists for chat DB
os.makedirs("tmp", exist_ok=True)

# SQLite for multi-turn chat persistence (exactly as shown in Agno docs)
_chat_db = SqliteDb(db_file="tmp/cardiosense_chat.db")

# In-memory registry: session_id -> Agent instance
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
        f"Gender: {patient_data.get('gender', 'N/A')}",
        f"Age: {patient_data.get('age', 'N/A')} years",
        f"Hypertension: {'Yes' if patient_data.get('hypertension') else 'No'}",
        f"Heart Disease: {'Yes' if patient_data.get('heart_disease') else 'No'}",
        f"Ever Married: {patient_data.get('ever_married', 'N/A')}",
        f"Work Type: {patient_data.get('work_type', 'N/A')}",
        f"Residence: {patient_data.get('Residence_type', 'N/A')}",
        f"Avg Glucose Level: {patient_data.get('avg_glucose_level', 'N/A')} mg/dL",
        f"BMI: {patient_data.get('bmi', 'N/A')}",
        f"Smoking Status: {patient_data.get('smoking_status', 'N/A')}",
    ]
    patient_block = "\n".join(f"  - {l}" for l in patient_lines)

    # ── Prediction ──
    pred_block = (
        f"  - Stroke Risk: {prediction.get('probability_percentage', '?')}%\n"
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
            f"{s.get('original_risk', '?')}% -> {s.get('modified_risk', '?')}% "
            f"(delta: {s.get('risk_delta', 0):+.1f}%) | {s.get('description', '')}"
        )
    whatif_block = "\n".join(scenario_lines) if scenario_lines else "  (None generated)"

    combined = whatif.get("combined_risk")
    combined_line = ""
    if combined is not None:
        combined_line = (
            f"\n  BEST COMBINED OUTCOME (all changes): "
            f"{combined}% ({whatif.get('combined_risk_level', '?')})"
        )

    return (
        "=== PATIENT HEALTH ASSESSMENT DATA ===\n\n"
        f"PATIENT PROFILE:\n{patient_block}\n\n"
        f"STROKE RISK PREDICTION:\n{pred_block}\n\n"
        f"TOP RISK FACTORS (SHAP):\n{shap_block}\n\n"
        f"WHAT-IF SCENARIOS:\n{whatif_block}{combined_line}\n\n"
        "======================================"
    )


def start_chat_session(
    patient_data: dict,
    prediction: dict,
    explanation: dict,
    whatif: dict,
) -> tuple:
    """
    Start a new chat session with patient context injected.
    Returns (session_id, greeting_message_text).
    """
    session_id = f"cs-{uuid.uuid4().hex[:8]}"
    system_context = build_system_context(patient_data, prediction, explanation, whatif)

    # Create agent following exact Agno SDK pattern from the docs
    agent = Agent(
        model=Nvidia(
                max_tokens=16384,
                temperature=0.1,  # Slightly higher for better reasoning
                top_p=0.95,
                id="minimaxai/minimax-m2.1"
            ),
        # description is added to the START of system message
        description=(
            "You are CardioSense AI, a warm and empathetic health assistant "
            "built into the CardioSense+ stroke risk prediction platform. "
            "A patient has just completed their assessment. "
            "Their full data and results are below.\n\n"
            + system_context
        ),
        # instructions are wrapped in <instructions> tags
        instructions=[
            "Speak in a warm, empathetic, and professional tone.",
            "Reference the patient's specific data when answering.",
            "Explain medical terms (hypertension, BMI, glucose) in plain language.",
            "When discussing What-If scenarios, give practical lifestyle tips.",
            "If risk is HIGH, be reassuring but honest.",
            "Always remind them you are an AI, not a doctor.",
            "Recommend consulting a healthcare professional for decisions.",
            "Keep responses concise (2-3 paragraphs) unless asked for more.",
            "Never fabricate data — only reference the assessment data above.",
            "If the user seems anxious, de-escalate with empathy first.",
        ],
        # expected_output is appended to END of system message
        expected_output=(
            "Clear, empathetic, personalized health guidance based on "
            "the patient's specific assessment data, with practical next steps."
        ),
        # Persistence (exactly as shown in Agno docs)
        db=_chat_db,
        add_history_to_context=True,
        num_history_runs=5,
        add_datetime_to_context=True,
    )

    _sessions[session_id] = agent

    # Auto-generate greeting using agent.run() — the programmatic version
    # of print_response() (which prints to terminal and can't be used in APIs)
    greeting_prompt = (
        "The patient has just seen their stroke risk results. "
        "Introduce yourself in 1 sentence, then give a brief empathetic "
        "summary of their key findings (risk %, top 2 factors). "
        "End by asking what they'd like to discuss. Keep it concise."
    )
    response = agent.run(greeting_prompt, session_id=session_id)
    
    # Agno catches API errors internally and returns them as strings like
    # "Connection error." or "404 page not found", so we must catch those specifically
    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content or str(getattr(response, "status", "")).lower() == "error":
        raise ConnectionError(f"LLM API Error: {response.content}")
        
    return session_id, response.content


def get_chat_response(session_id: str, user_message: str) -> str:
    """Get a response in an existing session. Returns response text."""
    agent = _sessions.get(session_id)
    if agent is None:
        raise ValueError(f"Session '{session_id}' not found or expired.")

    response = agent.run(user_message, session_id=session_id)
    
    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content or str(getattr(response, "status", "")).lower() == "error":
        raise ConnectionError(f"LLM API Error: {response.content}")
        
    return response.content