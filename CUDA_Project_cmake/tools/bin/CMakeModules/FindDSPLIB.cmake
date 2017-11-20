# - Find DSPLIB SDK.
# This module looks for TI DSPLIB SDK via the DSPLIB_DIR environment variable.
#
# This module defines:
#
#  DSPLIB_FOUND				- Set true if found.
#  DSPLIB_INCLUDE_DIR 		- Include directory.
#  DSPLIB_LIBRARY_DIR 		- Library directory for DSP target.
#

include(FindPackageHandleStandardArgs)

find_path(DSPLIB_INCLUDE_DIR "dsplib64plus.h" PATHS ENV DSPLIB_DIR PATH_SUFFIXES "include")
find_path(DSPLIB_LIBRARY_DIR "dsplib64plus.lib" PATHS ENV DSPLIB_DIR PATH_SUFFIXES "lib")

find_package_handle_standard_args(DSPLIB DEFAULT_MSG DSPLIB_INCLUDE_DIR DSPLIB_LIBRARY_DIR)

mark_as_advanced(DSPLIB_INCLUDE_DIR DSPLIB_LIBRARY_DIR)
