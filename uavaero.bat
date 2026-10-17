@echo off
REM ============================================================================
REM  uavaero - double-click to run.
REM
REM  Uses the SYSTEM Python deliberately. There is no virtualenv in this tree.
REM
REM  The window stays open if the app exits badly, because the whole reason to
REM  launch from a .bat rather than a shortcut is to keep the traceback where
REM  you can read it.
REM ============================================================================

setlocal
cd /d "%~dp0"
title uavaero

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Python was not found on PATH.
    echo   Install Python 3.13 and tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

REM Which interpreter, in case there is more than one on this machine and the
REM wrong one is first on PATH - the commonest cause of "but I installed that".
for /f "delims=" %%v in ('python -c "import sys;print(sys.version.split()[0])"') do set PYVER=%%v
for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)"') do set PYEXE=%%p
echo   Python %PYVER%
echo   %PYEXE%
echo.

REM Fail early with a readable message rather than a Qt traceback.
python -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   PySide6 is not installed for the Python above.
    echo   Install the dependencies with:
    echo.
    echo       python -m pip install PySide6 numpy scipy matplotlib PyYAML
    echo       python -m pip install pyvista pyvistaqt neuralfoil aerosandbox build123d
    echo.
    echo   The pinned versions are in pyproject.toml.
    echo.
    pause
    exit /b 1
)

REM OpenVSP is a warning rather than an error: you can author and save an
REM airframe without it, you just cannot solve one.
python -c "import openvsp" >nul 2>nul
if errorlevel 1 (
    echo   [warning] OpenVSP is not importable for this Python.
    echo             Geometry and saving will work; solving will not.
    echo.
)

echo   Starting uavaero...
echo   Leave this window open - solver output appears here as well as in the
echo   app's console (Ctrl+`^).
echo.

python main.py
set EXITCODE=%ERRORLEVEL%

REM A clean exit closes the window. A crash keeps it open.
if not "%EXITCODE%"=="0" (
    echo.
    echo ----------------------------------------------------------------
    echo   uavaero exited with code %EXITCODE%.
    echo   The traceback above is the useful part - copy it into the chat.
    echo ----------------------------------------------------------------
    echo.
    pause
)

endlocal & exit /b %EXITCODE%
