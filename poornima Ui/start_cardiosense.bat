@echo off
title CardioSense+ Launcher
color 0A

echo ============================================================
echo             CardioSense+ - AI Stroke Risk Predictor
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
pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn joblib imbalanced-learn pydantic agno sqlalchemy --quiet
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install.
) else (
    echo       Done!
)
echo.

:: ALWAYS delete existing model artifacts to prevent version mismatch issues
echo [2/6] Cleaning old model artifacts...
if exist "cardiosense_artifacts.pkl" (
    del /f /q "cardiosense_artifacts.pkl"
    echo       Old artifacts deleted!
) else (
    echo       No old artifacts found.
)
echo.

:: Set Nvidia API Key for AI Chat feature (permanent + session)
echo [3/6] Setting up AI Chat credentials...
:: setx writes to Windows registry - all new processes (incl uvicorn subprocesses) inherit it
setx NVIDIA_API_KEY "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" >nul 2>&1
:: set covers the current session immediately (setx only takes effect on NEW windows)
set NVIDIA_API_KEY=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
:: Also write a .env file so python-dotenv can load it as a fallback
echo NVIDIA_API_KEY=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> .env
echo       NVIDIA_API_KEY configured (permanent + session + .env)!
echo.

:: ALWAYS train the model fresh to ensure compatibility with current Python packages
echo [4/6] Training model with current environment...
echo       This ensures compatibility with your Python package versions.
echo.
python train_model.py
if errorlevel 1 (
    echo [ERROR] Model training failed!
    echo Please check if the dataset file exists: healthcare-dataset-stroke-data.csv
    pause
    exit /b 1
)
echo       Model trained successfully!
echo.

echo [5/6] Starting API Server on http://127.0.0.1:8000 ...
:: No --reload: avoids uvicorn spawning a watchfiles subprocess that loses env vars on Windows
start "CardioSense+ API" cmd /k "title CardioSense+ API Server && color 0B && set NVIDIA_API_KEY=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 && python -m uvicorn main:app --host 127.0.0.1 --port 8000"
timeout /t 5 /nobreak >nul
echo       API Server started!
echo.

echo [6/6] Starting Frontend Server on http://127.0.0.1:3000 ...
start "CardioSense+ Frontend" cmd /k "title CardioSense+ Frontend && color 0E && python -m http.server 3000"
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
echo Opening CardioSense+ in your browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:3000

echo.
echo Press any key to close this launcher (servers will keep running)...
pause >nul
