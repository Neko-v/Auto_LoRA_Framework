@echo off
title Auto LoRA Framework
color 0A

echo ===================================================
echo   Initializing Auto LoRA Framework (Portable)...
echo   Please wait, starting the engine...
echo ===================================================

set PYTHONPATH=%cd%\sd-scripts;%PYTHONPATH%
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

.\pyenv\python.exe gui.py

pause