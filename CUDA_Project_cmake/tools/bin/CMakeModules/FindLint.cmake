# - Find Gimpel's Lint
# This module looks for Gimpel's PC-Lint (see www.gimpel.com).
# Support Windows Lint under Linux using Wine.
#
# This module defines:
#
#  LINT_FOUND		 - Set true if found.
#  Lint_EXECUTABLE	 - Points to lint-nt.exe.
#  Lint_INCLUDE_DIRS - Path to ".lnt" files (not really header files, but Lint searches using the include path).
#  Lint_ARCH_PATH	 - Path to "lintarch.lnt" architecture specific file, needs user supplied LINT_ARCH, e.g. "omapl138_dsp",
#
# Requires the following to be set by user:
#
#  LINT_ARCH		 - Must be set by caller to search for architecture specific "lintarch.lnt" file.

include(FindPackageHandleStandardArgs)

if(NOT DEFINED LINT_ARCH)
	message(FATAL_ERROR "Please ensure that LINT_ARCH is set prior to includes, e.g. 'set(LINT_ARCH \"omapl138_dsp\")'")
endif()

find_program(Lint_EXECUTABLE NAMES lint-nt lint-nt.exe PATH_SUFFIXES "lint9")
find_path(Lint_INCLUDE_DIRS au-misra2.lnt PATH_SUFFIXES "lint9")
if(${CMAKE_BINARY_DIR} MATCHES "i386_win_vc")
	# Use MSVC specific version
	find_path(Lint_ARCH_PATH lintarch.lnt PATH_SUFFIXES "lint9" "lint9/arch/i386_win_vc")
else()
	# Use generic GCC version
	find_path(Lint_ARCH_PATH lintarch.lnt PATH_SUFFIXES "lint9" "lint9/arch/${LINT_ARCH}")
endif()

find_package_handle_standard_args(Lint DEFAULT_MSG Lint_EXECUTABLE Lint_INCLUDE_DIRS Lint_ARCH_PATH)

mark_as_advanced(Lint_EXECUTABLE)
