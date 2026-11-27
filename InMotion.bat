@echo off
rem Abre o menu do InMotion sem janela de terminal (spec 024).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Ambiente nao encontrado. Instale seguindo o README: py -3.10 -m venv .venv
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m inmotion.menu
