@echo off
rem Instala o InMotion nesta pasta (Python 3.10, dependencias com voz e modelos).
rem Pode rodar de novo: o que ja esta pronto e pulado. Log em logs\instalar.log.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\instalar.ps1" %*
echo.
pause
