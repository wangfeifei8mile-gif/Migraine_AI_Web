@echo off
echo ========================================================
echo       Migraine AI Prediction System
echo       Environment: tabpfn_env
echo ========================================================
echo.
echo Starting Web Application...
echo.

:: 直接使用绝对路径调用 tabpfn_env 里的 python
"F:\Anaconda3\envs\tabpfn_env\python.exe" -m streamlit run app.py

:: 如果运行结束或报错，暂停显示信息
pause
