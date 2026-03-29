# DiabSense+ 🩺

**DiabSense+** is an advanced AI-powered web platform designed to predict a patient's post-treatment HbA1c levels based on a comprehensive set of clinical, demographic, and lifestyle factors. By leveraging robust ensemble machine learning and Explainable AI (SHAP), the application provides personalized, actionable insights to aid in diabetes management and lifestyle intervention planning.

---

## ✨ Key Features

* **Highly Accurate HbA1c Prediction** 🎯
  Utilizes a state-of-the-art Stacking Ensemble model (combining 9 diverse algorithms including XGBoost, Random Forest, and SVR) to accurately project HbA1c levels 3 months post-intervention.
* **Explainable AI (XAI) with SHAP** 📊
  Demystifies the "black box" of machine learning by providing a dynamic waterfall chart of the top contributing factors. It visually explains exactly *why* a specific prediction was made for a patient.
* **Interactive Risk Simulator** 🎛️
  Allows users to dynamically adjust key health metrics (BMI, Fasting Blood Sugar, PPBS, Cholesterol, Triglycerides, Waist Circumference, and Blood Pressure) via intuitive sliders to instantly see real-time updates to their predicted HbA1c trajectory.
* **Intelligent "What-If" Analysis** 💡
  Automatically generates actionable, personalized lifestyle modification scenarios (e.g., "Join Yoga Intervention", "Improve Sleep Quality"). It quantifies the exact theoretical reduction in HbA1c if the patient adopts these specific changes.
* **LLM-Powered Chat Assistant** 💬
  Integrates a context-aware AI chatbot (powered by Agno/NVIDIA) to answer patient queries directly related to their specific prediction results and general diabetes management.

---

## 🛠️ Technology Stack

**Frontend**
* HTML5 / CSS3 / Vanilla JavaScript
* Chart.js (for SHAP visualizations)
* Fully responsive, glassmorphism UI design

**Backend API**
* Python
* FastAPI & Uvicorn (High-performance ASGI server)
* Pydantic (Strongly typed data validation)

**Machine Learning & Data Science**
* Scikit-Learn (Pipelines, Imputation, Linear/Tree models)
* XGBoost
* SHAP (TreeExplainer for interpretability)
* Pandas & NumPy

---

## 🧠 Machine Learning Pipeline

The prediction engine is built for maximum robustness and handle complex real-world clinical data:

1. **Automated Imputation Tournament:** Before training, the pipeline evaluates multiple missing-data imputation strategies (Mean, Median, KNN, MICE) against a masked validation set, automatically selecting the approach with the lowest RMSE.
2. **Interaction Feature Engineering:** Explicitly creates interaction features between the assigned Care Plan (e.g., Yoga vs Standard Care) and baseline clinical markers (HbA1c, FBS, PPBS) to force the model to learn the modulating effect of interventions.
3. **Stacking Regressor Ensemble:** Fits 9 diverse base estimators (Ridge, Lasso, ElasticNet, BayesianRidge, SVM, Random Forest, Gradient Boosting, KNN, XGBoost). Their predictions are fed into a Ridge Meta-Learner to produce the final, highly accurate HbA1c prediction.
4. **Standalone Explainer Model:** Runs a parallel XGBoost model purely dedicated to generating stable SHAP values for the frontend interface.

---

## 🚀 Setup & Installation

Follow these steps to run the application locally:

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn pydantic joblib
# Optional: For the LLM Chat functionality
pip install agno sqlalchemy python-dotenv
```

### 3. Train the Model (Required First Step)
Before starting the API, you must train the model using your CSV datasets. This creates the `diabsense_artifacts.pkl` file required by the backend.
```bash
python train_model.py
```

### 4. Start the Backend API
Launch the FastAPI server. It will automatically load the compiled model artifacts.
```bash
uvicorn main:app --reload
# The API will be available at http://127.0.0.1:8000
# Interactive API docs available at http://127.0.0.1:8000/docs
```

### 5. Launch the Frontend
Simply open the `index.html` file in any modern web browser.
*(Alternatively, serve the directory using a simple HTTP server: `python -m http.server 8080`)*

---

## 📂 Project Structure

* `index.html` - The main frontend user interface.
* `styles.css` - Custom styling, CSS variables, and glassmorphism effects.
* `script.js` - Frontend logic, API communication, SHAP charting, and Simulator controls.
* `main.py` - FastAPI backend application, containing endpoints for `/predict`, `/explain`, and `/whatif`.
* `train_model.py` - The automated machine learning pipeline script.
* `llm.py` *(if applicable)* - Configuration for the integrated AI chatbot.

---

## 🤝 Clinical Context & Disclaimer
*This application is a demonstrative tool designed to showcase AI capabilities in health data modeling. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.*
