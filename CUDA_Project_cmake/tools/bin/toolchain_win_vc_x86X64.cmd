@echo off
echo * Configuring tool chain and CMake

rem expects Debug or Release
set BUILD_TYPE=%1
rem expects path
rem set SOURCE_DIR=%2
rem expects vs10 or vs13
set VS_VERSION=%3
rem expects 32 or 64
set BITS_ARCH=%4

set SOURCE_DIR=%SOURCE_PATH%
if NOT "%BITS_ARCH%" == "64" set BITS_ARCH="32"

echo   + Using build type %BUILD_TYPE%, %BITS_ARCH% bit architecture and source directory %SOURCE_DIR%

rem USE_ADTF will be unset by default. It will only be set to true if we are in the specific VS10 64bits case
if "%VS_VERSION%" == "vs10" (
  set USE_CPP11="0"
  if "%BITS_ARCH%" == "64" (
    set VS="Visual Studio 10 Win64"
  ) ELSE (
    set VS="Visual Studio 10"
  )
) ELSE (
  if "%BITS_ARCH%" == "64" (
    set VS="Visual Studio 12 Win64"
  ) ELSE (
    set VS="Visual Studio 12"
  )
)

echo   + Using %VS% because %VS_VERSION%
echo %SOURCE_DIR%
pause
::echo %CAKE_MODULE_PATH%
if exist %TOOLS_PATH%\bin\CMakeModules (
	echo  * Calling cmake...
	echo .
	echo ------
    echo %VS%
	echo %TOOLS_BIN_PATH%
	cmake -G %VS% -DWIN32=%WIN32APP% -DCMAKE_MODULE_PATH=%CMAKE_MODULE_PATH% -DCUDAINSTALL_PATH=%CUDA_INSTALL_PATH% -DCUDA_LIBRARY="%CUDA_LIB_PATH%" -DCUDA_BINARY="%CUDA_BIN_PATH%" -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin" -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin" -DOPENCV_DEBUG=%OPENCV_PATH% -DOPENCV_RELEASE=%OPENCV_PATH% -DOPENCV_LIB=%OPENCV_LIB_PATH% -DOPENCV_BIN=%OPENCV_BIN_PATH% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCPP11=%USE_CPP11% %SOURCE_DIR%
	echo ------
	echo .
) ELSE (
	echo   - Could not run cmake. Unable to locate CMakeModules (expected in '%TOOLS_PATH%\bin\CMakeModules')
)

echo .
echo Use 'Visual Studio 2012 IDE' to build.
echo Start IDE from this command prompt to ensure valid environment.
pause
rem echo  "%ProgramFiles%\Microsoft Visual Studio 12.0\Common7\IDE\vsdiag_regwcf.exe"
