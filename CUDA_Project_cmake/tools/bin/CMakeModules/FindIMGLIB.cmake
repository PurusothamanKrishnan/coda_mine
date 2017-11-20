# - Find IMGLIB SDK.
# This module looks for TI IMGLIB SDK via the IMGLIB_DIR environment variable.
#
# This module defines:
#
#  IMGLIB_FOUND				- Set true if found.
#  IMGLIB_INCLUDE_DIR 		- Include directory.
#  IMGLIB_LIBRARY_DIR 		- Library directory for DSP target.
#  IMGLIB_LIBRARY_HOST_DIR 	- Library directory for host PC.
#

include(FindPackageHandleStandardArgs)

find_path(IMGLIB_INCLUDE_DIR "imglib64plus.h" PATHS ENV IMGLIB_DIR PATH_SUFFIXES "include")
find_path(IMGLIB_LIBRARY_DIR "imglib2.l64P" PATHS ENV IMGLIB_DIR PATH_SUFFIXES "lib/target")
find_path(IMGLIB_LIBRARY_HOST_DIR "imglib2_host.lib" PATHS ENV IMGLIB_DIR PATH_SUFFIXES "lib/host")

find_package_handle_standard_args(IMGLIB DEFAULT_MSG IMGLIB_INCLUDE_DIR IMGLIB_LIBRARY_DIR)

mark_as_advanced(IMGLIB_INCLUDE_DIR IMGLIB_LIBRARY_DIR)
