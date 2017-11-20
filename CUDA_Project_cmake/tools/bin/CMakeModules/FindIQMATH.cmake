# - Find IQMATH SDK.
# This module looks for TI IQMATH SDK via the IQMATH_DIR environment variable.
#
# This module defines:
#
#  IQMATH_FOUND				- Set true if found.
#  IQMATH_INCLUDE_DIR 		- Include directory.
#  IQMATH_LIBRARY_DIR 		- Library directory for DSP target.
#

include(FindPackageHandleStandardArgs)

find_path(IQMATH_INCLUDE_DIR "IQmath.h" PATHS ENV IQMATH_DIR PATH_SUFFIXES "include")
find_path(IQMATH_LIBRARY_DIR "IQmath_c64x+.lib" PATHS ENV IQMATH_DIR PATH_SUFFIXES "lib")

find_package_handle_standard_args(IQMATH DEFAULT_MSG IQMATH_INCLUDE_DIR IQMATH_LIBRARY_DIR)

mark_as_advanced(IQMATH_INCLUDE_DIR IQMATH_LIBRARY_DIR)
