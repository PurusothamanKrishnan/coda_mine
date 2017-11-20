# - Use Module for VLIB
# Support TI VLIB.
#

cmake_policy(SET CMP0015 NEW)
find_package(VLIB REQUIRED)
include_directories(${VLIB_INCLUDE_DIR})
link_directories(${VLIB_LIBRARY_DIR})
link_directories(${VLIB_LIBRARY_HOST_DIR})
