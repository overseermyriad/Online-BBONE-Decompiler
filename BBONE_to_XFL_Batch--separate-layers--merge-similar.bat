@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo     Batch Processing Flash Animation Assets
echo ===================================================

:: Ensure python is installed and in the system PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to your Environment PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

set "PROCESSED_COUNT=0"

:: Loop through every .bbone file in the current directory
for %%F in (*.bbone) do (
    set "BBONE_FILE=%%F"
    set "BASE_NAME=%%~nF"

    echo.
    echo ---------------------------------------------------
    echo Checking asset group: [!BASE_NAME!]
    
    if exist "!BBONE_FILE!" (
        echo Found matching pair: !BBONE_FILE!
        echo Executing pipeline python script...
        
        python BBONE_to_XFL.py "!BBONE_FILE!" --separate-layers --merge-similar
        
        if !errorlevel! equ 0 (
            set /a PROCESSED_COUNT+=1
        ) else (
            echo [ERROR] Pipeline failed or encountered an error processing !BASE_NAME!
            echo Halting batch process.
            pause
            exit /b 1
        )
    )
)

echo.
echo ===================================================
echo Done! Successfully processed %PROCESSED_COUNT% animation asset(s).
echo ===================================================
pause