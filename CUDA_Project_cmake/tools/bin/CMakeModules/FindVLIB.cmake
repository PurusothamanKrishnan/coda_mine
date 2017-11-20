# - Find VLIB SDK.
# This module looks for TI VLIB SDK via the VLIB_DIR environment variable.
#
# This module defines:
#
#  VLIB_FOUND				- Set true if found.
#  VLIB_INCLUDE_DIR 		- Include directory.
#  VLIB_LIBRARY_DIR 		- Library directory for DSP target.
#  VLIB_LIBRARY_HOST_DIR 	- Library directory for host PC.
#

include(FindPackageHandleStandardArgs)

find_path(VLIB_INCLUDE_DIR "VLIB_prototypes.h" PATHS ENV VLIB_DIR PATH_SUFFIXES "include")
find_path(VLIB_LIBRARY_DIR "vlib.l64p" PATHS ENV VLIB_DIR PATH_SUFFIXES "library/c64plus")
find_path(VLIB_LIBRARY_HOST_DIR "vlib.lib" PATHS ENV VLIB_DIR PATH_SUFFIXES "library/host")

find_package_handle_standard_args(VLIB DEFAULT_MSG VLIB_INCLUDE_DIR VLIB_LIBRARY_DIR)

mark_as_advanced(VLIB_INCLUDE_DIR VLIB_LIBRARY_DIR)
