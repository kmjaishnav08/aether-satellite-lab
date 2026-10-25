@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment...
  py -m venv .venv
  if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"
echo Checking required libraries...
pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Running physics validation tests...
python -m unittest discover -s tests -v
if errorlevel 1 goto :error
echo.
echo Starting AETHER Satellite Capture and Tether Dynamics Lab...
streamlit run app.py
goto :eof
:error
echo.
echo Setup or launch failed. Confirm that Python is installed and internet is available for first-time package installation.
pause
exit /b 1
