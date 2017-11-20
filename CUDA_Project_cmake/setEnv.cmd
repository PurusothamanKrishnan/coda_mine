@echo off
set BUILD_TYPE=%1

set ARCH=%2


set CURR_PATH=%~dp0
set SOURCE_PATH=%~dp0
set PROJ_PATH=%CURR_PATH%
echo *************

if DEFINED CUDA_PATH (set CUDA_INSTALL_PATH=%CUDA_PATH%) ELSE (echo CUDA Path not set)

echo %CUDA_PATH%
set CUDA_BIN_PATH=%CUDA_PATH%\bin

set TOOLS_PATH=NOTFOUND
set TOOLS_BIN_PATH=NOTFOUND
set CMAKE_MODULE_PATH=NOTFOUND
set CUDA_LIB_PATH=NOTFOUND
echo ********************** Setting VS PATH ********************************
if exist %VCInstallDir% (
	echo *************
	set VS_PATH=%VCInstallDir%
	set VSPATHROOT=%VCInstallDir%
	set GCCCOMPIERPATH=%VCInstallDir%bin
) ELSE (
	echo *************
	echo VS not installed.
	echo *************
)

if "%ARCH%"=="64" (
	set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
) ELSE (
	set CUDA_LIB_PATH=%CUDA_PATH%\lib\Win32
)
echo VS path %VS_PATH%
	echo VS path Root %VSPATHROOT%
	echo VS Compiler Path %GCCCOMPIERPATH%
echo ********************** End of Setting VS PATH ********************************

echo ********************** Setting OPENCV PATH ********************************
set OPENCV_PATH=C:\openCV2.3.1

echo "%OPENCV_PATH%"
::SETLOCAL ENABLEDELAYEDEXPANSION
if exist "%OPENCV_PATH%" (
	set OPENCV_ROOT_PATH=%OPENCV_PATH%
	set OPENCV_INCLUDE_PATH=%OPENCV_PATH%\opencv\include
	if "%ARCH%"=="64" (
		set OPENCV_LIB_PATH=%OPENCV_PATH%\opencv\build\x64\vc10\lib
		set OPENCV_BIN_PATH=%OPENCV_PATH%\opencv\build\x64\vc10\bin
		set OPENCV_STATICLIB_PATH=%OPENCV_PATH%\opencv\build\x64\vc10\staticlib
	) ELSE (
		set OPENCV_LIB_PATH=%OPENCV_PATH%\opencv\build\x86\vc10\lib
		set OPENCV_BIN_PATH=%OPENCV_PATH%\opencv\build\x86\vc10\bin
		set OPENCV_STATICLIB_PATH=%OPENCV_PATH%\opencv\build\x86\vc10\staticlib
	)
	
) 
echo %OPENCV_LIB_PATH%
echo %OPENCV_BIN_PATH%
echo %OPENCV_STATICLIB_PATH%
::ENDLOCAL
echo ********************** End of Setting OPENCV PATH ********************************

echo ********************** Setting Tools PATH ********************************
if exist "%PROJ_PATH%tools" (
	set TOOLS_PATH=%PROJ_PATH%tools
	set TOOLS_BIN_PATH=%PROJ_PATH%tools\bin
	set CMAKE_MODULE_PATH=%PROJ_PATH%tools\bin
	echo Cmake Modules Path Set
)
echo ********************** End of Setting Tools PATH ********************************

rem Exports paths to PATH Enviroment
PATH=%TOOLS_BIN_PATH%;%GCCCOMPIERPATH%;%TOOLS_PATH%\cmake-2.8.11.2-win32-x86\bin;%OPENCV_ROOT_PATH%;%OPENCV_BIN_PATH%;%OPENCV_LIB_PATH%;%PATH%;
echo %PATH%

set CMAKE_MODULE_PATH=%CMAKE_MODULE_PATH:\=/%

echo "%CMAKE_MODULE_PATH%"
echo "%CAKE_MODULE_PATH%"
echo   + CMake path set

echo   + PATH set to include tools/bin, tools/cmake as well as opencv and opencv/bin

:done
prompt=$p[i386_vc]$g


