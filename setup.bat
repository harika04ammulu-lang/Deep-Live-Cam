@echo off
setlocal
cd /d "%~dp0"

echo === Deep-Live-Cam setup ===

python --version 2>nul | findstr /b /c:"Python 3.10" >nul
if errorlevel 1 (
    echo ERROR: Python 3.10 is required. See SETUP.md section 2.1.
    python --version
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv || exit /b 1
)

call venv\Scripts\activate.bat || exit /b 1

python -m pip install --upgrade pip || exit /b 1

echo Installing insightface wheel first...
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl || exit /b 1

echo Installing requirements - this takes a while...
pip install -r requirements.txt || exit /b 1

if not exist models mkdir models
if not exist models\inswapper_128_fp16.onnx (
    echo Downloading fast swap model...
    curl -L -o models\inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
)

echo.
echo Setup finished. Start the app with run_app.bat
endlocal
