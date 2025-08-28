:: Run this bath file in the directory where it is located.
@echo off

:: Run both MATLAB scripts
matlab -batch "run('nat_freqs.m'); run('damp_ratios.m')"

:: Run Python script
cd ..
python -m system_id.find_params

:: Return to the original directory
cd system_id

:: Delete temporary file
del temp_params.json
