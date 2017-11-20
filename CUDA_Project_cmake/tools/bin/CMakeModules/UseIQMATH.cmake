# - Use Module for IQMATH
# Support TI IQMATH.
#

cmake_policy(SET CMP0015 NEW)
find_package(IQMATH REQUIRED)
include_directories(${IQMATH_INCLUDE_DIR})
link_directories(${IQMATH_LIBRARY_DIR})
#link_directories(${IQMATH_LIBRARY_HOST_DIR})
