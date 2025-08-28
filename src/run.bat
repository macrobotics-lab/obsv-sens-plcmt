:: Run this bath file in the directory where it is located.
@echo off

:: Run system identification scripts
cd system_id
call run_system_id.bat

:: Run all other scripts
cd ..
python optimization.py
python kalman_filter.py
python plot_rmse.py