@echo off
rem This script creates symbolic links for all matplotlib styles in current directory

rem Create the stylelib folder where matplotlib looks for styles.
mkdir "%USERPROFILE%\.matplotlib\stylelib"

rem Loop through all .mplstyle files in the current directory
for %%f in ("%CD%\*.mplstyle") do (
    rem Creating a symbolic link for %%~nxf
    rem mklink "%USERPROFILE%\.matplotlib\stylelib\%%~nxf" "%%f"
	mklink "%USERPROFILE%\.matplotlib\stylelib\%%~nxf" "%%f"
)

echo All style links created!
