@echo on
pushd .

call setEnv.cmd

set BUILD_DIR=buildFolder
if not exist %BUILD_DIR% (
	echo  * Creating build directory in parent folder
	md %BUILD_DIR%
)
cd %BUILD_DIR%

call toolchain_win_vc_x86X64 "Debug" ".."


if %ERRORLEVEL% NEQ 0 (
  echo erroorororo
  pause
) ELSE (
	start CUDA_OPENCV.sln
)

::@echo %cmdcmdline% | findstr /l "\"\"" >NUL
::popd
::exit
