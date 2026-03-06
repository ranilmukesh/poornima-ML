@echo off
title DiabeSense+ Launcher
color 0A

echo ============================================================
echo          DiabeSense+ - AI Diabetes HbA1c Predictor
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo [1/6] Installing/Updating dependencies...
pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn joblib imbalanced-learn pydantic agno sqlalchemy python-dotenv --quiet
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install.
) else (
    echo       Done!
)
echo.

:: ALWAYS delete existing model artifacts to prevent version mismatch issues
echo [2/6] Cleaning old model artifacts...
if exist "diabesense_artifacts.pkl" (
    del /f /q "diabesense_artifacts.pkl"
    echo       Old artifacts deleted!
) else (
    echo       No old artifacts found.
)
:: Also clean legacy CardioSense artifacts if present
if exist "cardiosense_artifacts.pkl" (
    del /f /q "cardiosense_artifacts.pkl"
    echo       Legacy CardioSense artifacts removed.
)
echo.

:: Set Nvidia API Key for AI Chat feature (permanent + session)
echo [3/6] Setting up AI Chat credentials...
setx NVIDIA_API_KEY "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" >nul 2>&1
set NVIDIA_API_KEY=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
echo NVIDIA_API_KEY=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> .env
echo       NVIDIA_API_KEY configured (permanent + session + .env)!
echo.

:: ALWAYS train the model fresh to ensure compatibility with current Python packages
echo [4/6] Training DiabeSense+ model with current environment...
echo       This merges all 4 diabetes CSV datasets and trains XGBRegressor.
echo.
python train_model.py
if errorlevel 1 (
    echo [ERROR] Model training failed!
    echo Please check if all 4 diabetes CSV dataset files exist in this folder.
    pause
    exit /b 1
)
echo       Model trained successfully!
echo.

echo [5/6] Starting API Server on http://127.0.0.1:8000 ...
start "DiabeSense+ API" cmd /k "title DiabeSense+ API Server && color 0B && set NVIDIA_API_KEY=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 && python -m uvicorn main:app --host 127.0.0.1 --port 8000"
timeout /t 5 /nobreak >nul
echo       API Server started!
echo.

echo [6/6] Starting Frontend Server on http://127.0.0.1:3000 ...
start "DiabeSense+ Frontend" cmd /k "title DiabeSense+ Frontend && color 0E && python -m http.server 3000"
timeout /t 2 /nobreak >nul
echo       Frontend Server started!
echo.

echo ============================================================
echo                    SERVERS ARE RUNNING!
echo ============================================================
echo.
echo   API Documentation:  http://127.0.0.1:8000/docs
echo   Frontend UI:        http://127.0.0.1:3000
echo.
echo   TIP: Type fillDemoData() in browser console for demo data
echo   TIP: Click the chat bubble after results for AI assistant
echo.
echo ============================================================
echo.

:: Open the frontend in default browser
echo Opening DiabeSense+ in your browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:3000

echo.
echo Press any key to close this launcher (servers will keep running)...
pause >nul
