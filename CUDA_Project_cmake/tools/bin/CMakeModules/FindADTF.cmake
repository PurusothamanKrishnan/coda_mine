# - Find ADTF SDK.
# This module looks for ADTF SDK framework via the ADTF_DIR environment variable.
#
# This module defines:
#
#  ADTF_FOUND			- Set true if found.
#  ADTF_INCLUDE_DIR 	- Include directory.
#  ADTF_LIBRARY_DIR 	- Library directory.
#

include(FindPackageHandleStandardArgs)

find_path(ADTF_INCLUDE_DIR "adtf_plugin_sdk.h" PATHS ENV ADTF_DIR PATH_SUFFIXES "include")
find_path(ADTF_LIBRARY_DIR "adtfutil_1100.lib" PATHS ENV ADTF_DIR PATH_SUFFIXES "lib")

find_package_handle_standard_args(ADTF DEFAULT_MSG ADTF_INCLUDE_DIR ADTF_LIBRARY_DIR)

mark_as_advanced(ADTF_INCLUDE_DIR ADTF_LIBRARY_DIR)
