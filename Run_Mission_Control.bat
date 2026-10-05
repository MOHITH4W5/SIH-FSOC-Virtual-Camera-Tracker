@echo off
title SIH26169 FSOC Mission Control Launcher
cd /d "%~dp0"
echo Launching SIH26169 FSOC Coarse Alignment Mission Control...
start "" "%~dp0SIH_FSOC_Tracker.exe"
exit
