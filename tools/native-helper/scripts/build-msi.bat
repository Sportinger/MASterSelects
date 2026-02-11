@echo off
REM ============================================================
REM  Build MSI installer for MasterSelects Native Helper
REM
REM  Prerequisites:
REM    - WiX Toolset v3: winget install WiXToolset.WiXToolset
REM    - cargo-wix:      cargo install cargo-wix
REM    - FFMPEG_DIR set or ffmpeg/win64/ populated
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%"

echo.
echo ========================================
echo  MasterSelects Helper - MSI Build
echo ========================================
echo.

REM --- Step 1: Build release binary ---
echo [1/3] Building release binary...
cargo build --release
if errorlevel 1 (
    echo ERROR: cargo build failed!
    exit /b 1
)
echo       OK

REM --- Step 2: Copy FFmpeg DLLs to target\release ---
echo [2/3] Copying FFmpeg DLLs...

REM Determine FFmpeg bin directory
set "FFMPEG_BIN="

if defined FFMPEG_DIR (
    set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
)

if not defined FFMPEG_BIN (
    if exist "ffmpeg\win64\bin\avcodec-61.dll" (
        set "FFMPEG_BIN=%PROJECT_DIR%\ffmpeg\win64\bin"
    )
)

if not defined FFMPEG_BIN (
    echo ERROR: Cannot find FFmpeg DLLs. Set FFMPEG_DIR or place them in ffmpeg\win64\
    exit /b 1
)

echo       Source: %FFMPEG_BIN%

set "RELEASE_DIR=target\release"

copy /y "%FFMPEG_BIN%\avcodec-61.dll"    "%RELEASE_DIR%\" >nul 2>&1
copy /y "%FFMPEG_BIN%\avformat-61.dll"   "%RELEASE_DIR%\" >nul 2>&1
copy /y "%FFMPEG_BIN%\avutil-59.dll"     "%RELEASE_DIR%\" >nul 2>&1
copy /y "%FFMPEG_BIN%\swresample-5.dll"  "%RELEASE_DIR%\" >nul 2>&1
copy /y "%FFMPEG_BIN%\swscale-8.dll"     "%RELEASE_DIR%\" >nul 2>&1
copy /y "%FFMPEG_BIN%\ffmpeg.exe"        "%RELEASE_DIR%\" >nul 2>&1

REM Verify all files exist
set "MISSING=0"
for %%F in (masterselects-helper.exe avcodec-61.dll avformat-61.dll avutil-59.dll swresample-5.dll swscale-8.dll ffmpeg.exe) do (
    if not exist "%RELEASE_DIR%\%%F" (
        echo       MISSING: %%F
        set "MISSING=1"
    )
)
if "%MISSING%"=="1" (
    echo ERROR: Some required files are missing!
    exit /b 1
)
echo       OK (7 files)

REM --- Step 3: Build MSI ---
echo [3/3] Building MSI installer...
cargo wix --no-build --nocapture
if errorlevel 1 (
    echo ERROR: cargo wix failed!
    exit /b 1
)

echo.
echo ========================================
echo  MSI built successfully!
echo  Output: target\wix\*.msi
echo ========================================
echo.

dir /b target\wix\*.msi 2>nul
