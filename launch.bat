@echo off

set PYTHON=python
set VENV_DIR=.venv

:start_venv
if [%VENV_DIR%] == [-] goto  :skip_venv

dir %VENV_DIR%\Scripts\Python.exe
if %ERRORLEVEL% == 0 goto :activate_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv %VENV_DIR%
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory %VENV_DIR%

:activate_venv
set PYTHON="%~dp0%VENV_DIR%\Scripts\python.exe"
echo venv %PYTHON%
goto :launch

:skip_venv

:launch
call %VENV_DIR%\Scripts\activate.bat
%PYTHON% -m pip install -r requirements.txt | findstr /V /C:"Requirement already satisfied"
%PYTHON% MagicPrompt.py
exit

:endofscript
echo.
echo Launch unsuccessful. Exiting.

