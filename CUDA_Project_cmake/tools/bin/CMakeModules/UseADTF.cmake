# - Use Module for ADTF
# Support building an ADTF plugin which is a DLL by named with ".PBL" extension.
# Use of Qt4 is implied as ADTF uses this itself.
#
# Macros:
#
#  ADTF_SET_TARGET_PROPERITES	- Sets output target property to ".plb".
#

find_package(Qt4 REQUIRED)
include(UseQt4)
set(ALL_LIBS ${ALL_LIBS} ${QT_LIBRARIES})

cmake_policy(SET CMP0015 NEW)
find_package(ADTF REQUIRED)
include_directories(${ADTF_INCLUDE_DIR})
link_directories(${ADTF_LIBRARY_DIR})

#
# Set ADTF DLL output filename to "*.plb"
#
macro(ADTF_SET_TARGET_PROPERITES _target)
	set_target_properties(${_target} PROPERTIES SUFFIX ".plb")
endmacro()
