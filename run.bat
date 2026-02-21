@echo off
echo ========================================================
echo  Sign Language Recognition - v2 (Enhanced Backend)
echo ========================================================
echo.
echo  New in v2:
echo    * GPU / CUDA auto-detected
echo    * Robust landmark normalization
echo    * Adaptive prediction smoother
echo    * Session logger (saved to logs/)
echo    * Frame-skip inference optimization
echo.
pip install -q pyttsx3 pyspellchecker
echo.
echo Starting v2...
echo.
python modernui_v2.py
pause
