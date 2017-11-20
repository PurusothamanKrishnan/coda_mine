#
# $Id: Component.cmake 17142 2012-03-16 19:00:42Z kellawaytrevor $
#
# Component CMake support.
#

#
# Derive build type from binary directory name.
#
string(REGEX REPLACE ".*/([^/]+)" "\\1" BUILD_TYPE "${CMAKE_BINARY_DIR}")

# Release/Debug initialisation
if("${BUILD_TYPE}" MATCHES "release")
	set(CMAKE_BUILD_TYPE "Release")
else()
	set(CMAKE_BUILD_TYPE "Debug")
endif()

#
# Includes: Always include source directory, arch and binary directory
#
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/arch/${COMPONENT_ARCH}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

#
# Version.i
#
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.i.in" "${CMAKE_CURRENT_BINARY_DIR}/version.i")

#
# Library name
#
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
if((${VERSION_MAJOR} STREQUAL "00") AND (${VERSION_MINOR} STREQUAL "00"))
	set(LIBNAME "${TLA}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}")
else()
	set(LIBNAME "${TLA}_${COMPONENT_ARCH}_v${VERSION_MAJOR}.${VERSION_MINOR}_${COMPONENT_LIB_TYPE}")
endif()

#
# Executable output path
#
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/image")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/image")
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/image/${THIS_APP}.map")

#
# Install path
#
if(NOT DEFINED COMPONENT_INSTALL_ROOT)
	set(COMPONENT_INSTALL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../built")
endif()
if(NOT DEFINED COMPONENT_INSTALL_DIR)
	set(COMPONENT_INSTALL_DIR "${COMPONENT_INSTALL_ROOT}/${LIBNAME}")
endif()
file(MAKE_DIRECTORY ${COMPONENT_INSTALL_DIR})

#
# Import path (find it by looking for "config_all.cmake" which should be in the root.
#
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../config_all.cmake")
	set(COMPONENT_IMPORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../config_all.cmake")
	set(COMPONENT_IMPORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../config_all.cmake")
	set(COMPONENT_IMPORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../config_all.cmake")
	set(COMPONENT_IMPORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../config_all.cmake")
	set(COMPONENT_IMPORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
else()
	message(FATAL "Unable to locate COMPONENT_IMPORT_ROOT location whilst looking for config.all")
endif()

if(EXISTS "${COMPONENT_IMPORT_ROOT}/P1")
	set(COMPONENT_IMPORT_SUBROOT1 "${COMPONENT_IMPORT_ROOT}/P1")
endif()

if(EXISTS "${COMPONENT_IMPORT_ROOT}/components")
	set(COMPONENT_IMPORT_SUBROOT2 "${COMPONENT_IMPORT_ROOT}/components")
endif()

#message("COMPONENT_INSTALL_ROOT: ${COMPONENT_INSTALL_ROOT}")
#message("COMPONENT_INSTALL_DIR:  ${COMPONENT_INSTALL_DIR}")
#message("COMPONENT_IMPORT_ROOT:  ${COMPONENT_IMPORT_ROOT}")
#message("COMPONENT_IMPORT_SUBROOT1:  ${COMPONENT_IMPORT_SUBROOT1}")
#message("COMPONENT_IMPORT_SUBROOT2:  ${COMPONENT_IMPORT_SUBROOT2}")