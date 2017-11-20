# - Use Module for DSPLIB
# Support TI DSPLIB.
#

cmake_policy(SET CMP0015 NEW)
find_package(DSPLIB REQUIRED)
include_directories(${DSPLIB_INCLUDE_DIR})
#link_directories(${DSPLIB_LIBRARY_DIR})
#link_directories(${DSPLIB_LIBRARY_HOST_DIR})
