# - Generic compiler support for TI C6000 family
# Later versions of CMake (>V2.4) will compile "file.c" to "file.c.obj", as we have
# no way of changing the output filename we set CMAKE_C_OUTPUT_EXTENSION_REPLACE and
# CMAKE_CXX_OUTPUT_EXTENSION_REPLACE to disable this behaviour, and restore the old
# behaviour of mapping "file.c" to "file.obj".
#
# This module defines:
#
#  TI_C6000		 	 - Set to True if this file is included.
#

if(NOT TI_C6000)
	# CMake will parse this file twice, stop this as we append to some variables below.
	set(TI_C6000 1)

	#
	# Find additional tools needed (compilers are discovered automatically)
	#
	find_program(AR_EXE "ar6x" PATHS ${CMAKE_FIND_ROOT_PATH} NO_DEFAULT_PATH)
	set(CMAKE_AR "${AR_EXE}" CACHE FILEPATH "TI archiver" FORCE)

	find_program(LNK_EXE "lnk6x" PATHS ${CMAKE_FIND_ROOT_PATH} NO_DEFAULT_PATH)
	set(LINKER "${LNK_EXE}" CACHE FILEPATH "TI linker" FORCE)

	#
	# Find include path
	#
	find_path(_INCLUDE_PATH "c6x.h" PATHS ${CMAKE_FIND_ROOT_PATH} NO_DEFAULT_PATH)
	include_directories(${_INCLUDE_PATH})

	#
	# Find library path
	#
	find_path(_LIB_PATH "rts6400.lib" PATHS ${CMAKE_FIND_ROOT_PATH} NO_DEFAULT_PATH)
	link_directories(${_LIB_PATH})

	#
	# By default CMake will try and compile "file.c" to "file.c.obj"(in >V2.4).
	# For TI this will fail, as there is no "-o" option, it will always generate
	# "file.obj". Setting CMAKE_C_OUTPUT_EXTENSION_REPLACE to true forces the old
	# styles behaviour of removing the ".c" extension.
	#
	set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)
	set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
	set(CMAKE_ASM_C6X_OUTPUT_EXTENSION_REPLACE 1)

	#
	# File extensions and search prefix/suffixes.
	#
	set(CMAKE_C_OUTPUT_EXTENSION		".obj")
	set(CMAKE_CXX_OUTPUT_EXTENSION		".obj")
	set(CMAKE_FIND_LIBRARY_PREFIXES		"")
	set(CMAKE_FIND_LIBRARY_SUFFIXES		".lib")
	set(CMAKE_STATIC_LIBRARY_PREFIX 	"")
	set(CMAKE_STATIC_LIBRARY_SUFFIX		".lib")
	set(CMAKE_SHARED_LIBRARY_SUFFIX		"")
	set(CMAKE_EXECUTABLE_SUFFIX			".out")

	#
	# Default compiler flags (these can only be overridden in the Toolchain file, not in subsequent CMakeLists.txt)
	#
	set(CMAKE_C_FLAGS_INIT					"${CMAKE_C_FLAGS_INIT} -c -pden")
	set(CMAKE_C_FLAGS_DEBUG_INIT			"${CMAKE_C_FLAGS_DEBUG_INIT}")
	set(CMAKE_C_FLAGS_MINSIZEREL_INIT		"")
	set(CMAKE_C_FLAGS_RELEASE_INIT			"${CMAKE_C_FLAGS_RELEASE_INIT}")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT	"")

	set(CMAKE_CXX_FLAGS_INIT				"${CMAKE_CXX_FLAGS_INIT}")
	set(CMAKE_CXX_FLAGS_DEBUG_INIT			"${CMAKE_CXX_FLAGS_DEBUG_INIT}")
	set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT		"")
	set(CMAKE_CXX_FLAGS_RELEASE_INIT		"${CMAKE_CXX_FLAGS_RELEASE_INIT}")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "")

	#
	# Default linker flags, and library search and link command line flags.
	#
	set(CMAKE_EXE_LINKER_FLAGS_INIT			"${CMAKE_EXE_LINKER_FLAGS_INIT}")
	set(CMAKE_LIBRARY_PATH_FLAG				"-i ")
	set(CMAKE_LINK_LIBRARY_FLAG				"-l")

	#
	# Program invocation.
	# Note: Under Windows invocation of ar6x and lnk6x using full Unix style path does nothing,
	# and we can't stop CMake from converting the pathname backslashes to forward slashes, even
	# after using FILE(TO_NATIVE_PATH) on them, therefore they must be in the user's path.
	#
	set(CMAKE_C_COMPILE_OBJECT 			"\"${CMAKE_C_COMPILER}\" <FLAGS> <DEFINES> -fr<OBJECT_DIR> -fs<OBJECT_DIR> -ff${CMAKE_BINARY_DIR}/lst <SOURCE>")
	set(CMAKE_C_CREATE_STATIC_LIBRARY	"\"${CMAKE_AR}\" aq <TARGET> <OBJECTS>")
	set(CMAKE_C_CREATE_SHARED_LIBRARY)
	set(CMAKE_C_CREATE_MODULE_LIBRARY)
	set(CMAKE_CXX_COMPILE_OBJECT 		"${CMAKE_C_COMPILE_OBJECT}")
	set(CMAKE_CXX_CREATE_STATIC_LIBRARY	"${CMAKE_C_CREATE_STATIC_LIBRARY}")
	set(CMAKE_CXX_CREATE_SHARED_LIBRARY)
	set(CMAKE_CXX_CREATE_MODULE_LIBRARY)

	#
	# We place our list files in a common directory so they are easier to find.
	#
	file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lst)
endif()

#
# TI_START_LINK_FLAGS and TI_END_LINK_FLAGS will only be set on second CMake pass, so ignore TI_C6000 and always do this.
#
set(CMAKE_C_LINK_EXECUTABLE	"\"${LINKER}\" -c ${TI_START_LINK_FLAGS} <LINK_FLAGS> -o <TARGET> -m <TARGET_BASE>.map <OBJECTS> <LINK_LIBRARIES> ${TI_END_LINK_FLAGS}")
set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_C_LINK_EXECUTABLE}")
