# - Use Module for IMGLIB
# Support TI IMGLIB.
#

cmake_policy(SET CMP0015 NEW)
find_package(IMGLIB REQUIRED)
include_directories(${IMGLIB_INCLUDE_DIR})
#link_directories(${IMGLIB_LIBRARY_DIR})
#link_directories(${IMGLIB_LIBRARY_HOST_DIR})
