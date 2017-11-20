#
# $Id: ComponentArch.cmake 14913 2012-02-14 16:58:43Z kellawaytrevor $
#
# Component build architecture.
#

include("FindGraphvizDot")

#
# Always search current directory for includes.
#
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#
# Derive build type from binary directory name.
#
string(REGEX REPLACE ".*/([^/]+)" "\\1" COMPONENT_TYPE "${CMAKE_BINARY_DIR}")

# Debug versus release naming
if("${COMPONENT_TYPE}" MATCHES "release")
	set(COMPONENT_LIB_TYPE "r")
	if (${CMAKE_BINARY_DIR} MATCHES "i386_win_vc_ide")
		set(COMPONENT_LIB_INTERMEDIATE_DIR "Release/")
	else()
		set(COMPONENT_LIB_INTERMEDIATE_DIR "")
	endif()
else()
	set(COMPONENT_LIB_TYPE "d")
	if (${CMAKE_BINARY_DIR} MATCHES "i386_win_vc_ide")
		set(COMPONENT_LIB_INTERMEDIATE_DIR "Debug/")
	else()
		set(COMPONENT_LIB_INTERMEDIATE_DIR "")
	endif()

	add_definitions("-DSYS_DEBUG")
	add_definitions("-D_ASSERT_")
endif()

# Processor variants
if("${COMPONENT_TYPE}" MATCHES "i386_win")
	#
	# Intel 386 PC Windows
	#
	set(COMPONENT_ARCH "i386_win")
	set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)
	set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
	add_definitions("-Drestrict=")
	if("${COMPONENT_TYPE}" MATCHES "i386_win_vc")
		# Suppress MSVC warnings
		#TODO This is a horrible kludge, this should be in an i386 toolchain file really
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4068")		# Unknown pragmas
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")	# Unknown pragmas
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4996")		# Old style unsafe string functions
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")	# Old style unsafe string functions
	endif()
elseif("${COMPONENT_TYPE}" MATCHES "i386_linux")
	#
	# Intel 386 Linux
	#
	set(COMPONENT_ARCH "i386_linux")
	set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)
	set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
	add_definitions("-Drestrict=")
elseif("${COMPONENT_TYPE}" MATCHES "x64_linux")
	#
	# Intel x64 Linux
	#
	set(COMPONENT_ARCH "x64_linux")
	set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 1)
	set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
	add_definitions("-Drestrict=")
elseif("${COMPONENT_TYPE}" MATCHES "omapl138_arm")
	#
	# TI OMAPL138 ARM
	#
	set(COMPONENT_ARCH "omapl138_arm")
	include("FindTICL470Tools")
	include_directories("${CCS_INC_PATH}")
	include_directories("${TICL470Tools_BIOS_PATH}/packages")
	include_directories("${TICL470Tools_XDC_PATH}/packages")
elseif("${COMPONENT_TYPE}" MATCHES "omapl138_dsp")
	#
	# TI OMAPL138 DSP
	#
	set(COMPONENT_ARCH "omapl138_dsp")
	include("FindTICL6xTools")
	include_directories("${CCS_INC_PATH}")
	include_directories("${TICL6XTools_BIOS_PATH}/packages")
	include_directories("${TICL6XTools_XDC_PATH}/packages")
	add_definitions(-Dxdc_target_name__=C674)
	add_definitions(-Dxdc_target_types__=ti/targets/std.h)
	add_definitions(-Dxdc_cfg__header__=../configPkg/package/cfg/common_x674.h)
elseif("${COMPONENT_TYPE}" MATCHES "visionmid814x_a8")
	#
	# TI VisionMid ARM CortexA8
	#
	set(COMPONENT_ARCH "visionmid814x_a8")
	include("FindTICL470Tools")
	include_directories("${CCS_INC_PATH}")
	include_directories("${TICL470Tools_BIOS_PATH}/packages")
	include_directories("${TICL470Tools_XDC_PATH}/packages")
	include_directories("${TICL470Tools_XDC_PATH}/packages/ti/targets/arm/elf")	# Required so Lint can find "A8F.h"
	include_directories("${TICL470Tools_IPC_PATH}/packages")
	include_directories("${TICL470Tools_PSP_PATH}/packages")
	include_directories("${TICL470Tools_EDMA_PATH}/packages")
	include_directories("${TICL470Tools_XDAIS_PATH}/packages")
	include_directories("${TICL470Tools_NDK_PATH}/packages")
	include_directories("${TICL470Tools_NDK_PATH}/packages/ti/ndk/inc") # Required for Lint
	include_directories("${TICL470Tools_NDK_PATH}/packages/ti/ndk/inc/nettools")  # Required for Lint
	include_directories("${TICL470Tools_NSP_PATH}/packages")
	include_directories("${TICL470Tools_AVB_PATH}/AVBTP/IEEE1722/inc")
	include_directories("${TICL470Tools_AVB_PATH}/AVBTP/IEC61883")
	add_definitions(-Dxdc_target_name__=A8F)
	add_definitions(-Dxdc_target_types__=ti/targets/arm/elf/std.h)
elseif("${COMPONENT_TYPE}" MATCHES "visionmid814x_m3")
	#
	# TI VisionMid ARM CortexM3
	#
	set(COMPONENT_ARCH "visionmid814x_m3")
	include("FindTICL470Tools")
	include_directories("${CCS_INC_PATH}")
	include_directories("${TICL470Tools_BIOS_PATH}/packages")
	include_directories("${TICL470Tools_XDC_PATH}/packages")
	include_directories("${TICL470Tools_XDC_PATH}/packages/ti/targets/arm/elf")	# Required so Lint can find "M3.h"
	include_directories("${TICL470Tools_IPC_PATH}/packages")
	include_directories("${TICL470Tools_PSP_PATH}/packages")
	include_directories("${TICL470Tools_EDMA_PATH}/packages")
	include_directories("${TICL470Tools_XDAIS_PATH}/packages")
	include_directories("${TICL470Tools_XDAIS_PATH}/packages/ti/xdais")     # required for lint
	include_directories("${TICL470Tools_XDAIS_PATH}/packages/ti/xdais/dm")  # required for lint
	include_directories("${TICL470Tools_IVAHD_JPEGVDEC_PATH}/packages")
	include_directories("${TICL470Tools_FC_PATH}/packages")
	include_directories("${TICL470Tools_HDVPSS_PATH}/packages")
	include_directories("${TICL470Tools_IVAHD_HDVICP20_PATH}/packages")
	add_definitions(-Dxdc_target_name__=M3)
	add_definitions(-Dxdc_target_types__=ti/targets/arm/elf/std.h)
elseif("${COMPONENT_TYPE}" MATCHES "visionmid814x_dsp")
	#
	# TI VisionMid DSP
	#
	set(COMPONENT_ARCH "visionmid814x_dsp")
	include("FindTICL6xTools")
	include_directories("${CCS_INC_PATH}")
	include_directories("${TICL6XTools_BIOS_PATH}/packages")
	include_directories("${TICL6XTools_XDC_PATH}/packages")
	include_directories("${TICL6XTools_XDC_PATH}/packages/ti/targets")	# Required so Lint can find "C674.h"
	include_directories("${TICL6XTools_IPC_PATH}/packages")
	include_directories("${TICL6XTools_PSP_PATH}/packages")
	include_directories("${TICL6XTools_EDMA_PATH}/packages")
	include_directories("${TICL6XTools_XDAIS_PATH}/packages")
	include_directories("${TICL6XTools_FC_PATH}/packages")
	add_definitions(-Dxdc_target_name__=C674)
	add_definitions(-Dxdc_target_types__=ti/targets/std.h)
else()
	message(FATAL_ERROR "Unsupported build target")
endif()

#
# Setup Lint architecture.
#
set(LINT_ARCH ${COMPONENT_ARCH})

#
# Setup Lint logging if enabled
#
if(${CMAKE_BINARY_DIR} MATCHES "lintlog")
	if(${CMAKE_GENERATOR} MATCHES "JOM")
		message(FATAL_ERROR "For Lint Logging use toolchain_*_nmake to ensure single processor build")
	endif()
	set(LINTING_LOG "${CMAKE_BINARY_DIR}/lint_${COMPONENT_ARCH}.log")
endif()

#
# Start build list of reuse libraries
#
macro(COMPONENT_IMPORT_LIB_START app_name ver_major ver_minor)
	set(_dot_app_name ${app_name})
	set(_dot_ver_major ${ver_major})
	set(_dot_ver_minor ${ver_minor})
	file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${app_name}_modules.txt" "${app_name} v${ver_major}.${ver_minor} depends on:\n")
	if(GRAPHVIZ_DOT_FOUND)
		file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "digraph G {\n")
		file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "graph [rankdir=\"LR\"];\n")
		file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "edge [color=\"#666666\", arrowhead=\"open\"];\n")
		file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "node [style=filled, color=\"palegreen\"];\n")
	endif()
endmacro()

#
# Add reuse library to list.
#
macro(COMPONENT_IMPORT_LIB tla ver_major ver_minor)
	#message("COMPONENT_IMPORT_REUSE_LIB: ${tla} v${ver_major} ${ver_minor}")
	if((${ver_major} STREQUAL "00") AND (${ver_minor} STREQUAL "00"))
		if(DEFINED COMPONENT_IMPORT_SUBROOT1)
			include_directories("${COMPONENT_IMPORT_SUBROOT1}/${tla}/built/${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}")
		endif()
		if(DEFINED COMPONENT_IMPORT_SUBROOT2)
			include_directories("${COMPONENT_IMPORT_SUBROOT2}/${tla}/built/${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}")
			set(ALL_LIBS ${ALL_LIBS} "${COMPONENT_IMPORT_SUBROOT2}/${tla}/built/${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}${CMAKE_STATIC_LIBRARY_SUFFIX}")
		else()
			set(ALL_LIBS ${ALL_LIBS} "${COMPONENT_IMPORT_ROOT}/${tla}/built/${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}${CMAKE_STATIC_LIBRARY_SUFFIX}")
		endif()
		include_directories("${COMPONENT_IMPORT_ROOT}/${tla}/built/${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}")
	else()
		if(DEFINED COMPONENT_IMPORT_SUBROOT1)
			include_directories("${COMPONENT_IMPORT_SUBROOT1}/${tla}/built/${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}")
		endif()
		if(DEFINED COMPONENT_IMPORT_SUBROOT2)
			include_directories("${COMPONENT_IMPORT_SUBROOT2}/${tla}/built/${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}")
		endif()
		if(DEFINED COMPONENT_IMPORT_SUBROOT2)
			set(ALL_LIBS ${ALL_LIBS} "${COMPONENT_IMPORT_SUBROOT2/${tla}/built/${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}${tla}_${COMPONENT_ARCH}_${COMPONENT_LIB_TYPE}${CMAKE_STATIC_LIBRARY_SUFFIX}")		
		else()
			set(ALL_LIBS ${ALL_LIBS} "${COMPONENT_IMPORT_ROOT}/${tla}/built/${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}${CMAKE_STATIC_LIBRARY_SUFFIX}")
		endif()
		include_directories("${COMPONENT_IMPORT_ROOT}/${tla}/built/${tla}_${COMPONENT_ARCH}_v${ver_major}.${ver_minor}_${COMPONENT_LIB_TYPE}")
	endif()

	file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/${_dot_app_name}_modules.txt" "${tla} v${ver_major}.${ver_minor}\n")
	if(GRAPHVIZ_DOT_FOUND)
		FILE (APPEND "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "\"${tla} v${ver_major}.${ver_minor}\" -> \"${_dot_app_name} v${_dot_ver_major}.${ver_minor}\";\n")
	endif()
endmacro()

#
# End build list of reuse libraries
#
macro(COMPONENT_IMPORT_LIB_END)
	if(GRAPHVIZ_DOT_FOUND)
		# Do nothing at present
	endif()
endmacro()

#
# Generate reuse module diagram
#
macro(COMPONENT_DIAGRAM app_name target_name)
	if(GRAPHVIZ_DOT_FOUND)
		file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/modules.dot" "}\n")

		if(DEFINED LIBNAME)
			add_custom_command(	TARGET ${target_name}
            					POST_BUILD
            					COMMAND "${Graphviz_Dot_EXECUTABLE}" -Tpng ${CMAKE_CURRENT_BINARY_DIR}/modules.dot -o ${CMAKE_CURRENT_BINARY_DIR}/${app_name}_modules.png
								COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${TLA}_modules.txt ${COMPONENT_INSTALL_DIR}
								COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${TLA}_modules.png ${COMPONENT_INSTALL_DIR}
            					WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            					COMMENT "Generating module diagram")
		else()
			add_custom_command(	TARGET ${target_name}
            					POST_BUILD
            					COMMAND "${Graphviz_Dot_EXECUTABLE}" -Tpng ${CMAKE_CURRENT_BINARY_DIR}/modules.dot -o ${CMAKE_CURRENT_BINARY_DIR}/${app_name}_modules.png
            					WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            					COMMENT "Generating module diagram")
		endif()
	endif()
endmacro()

#
# Copy public header(s) and library and diagram to install directory.
# Note: Headers are copied at CMake config time, rest at build time.
#		Optional third argument is public header list.
#
macro(COMPONENT_INSTALL tla libname)
	if(NOT COMPONENT_INSTALL_DIR)
		set(COMPONENT_INSTALL_DIR COMPONENT_INSTALL_DIR_NOT_SET)
	endif()

	file(MAKE_DIRECTORY ${COMPONENT_INSTALL_DIR})

	#
	# Purge any headers that are no longer part of the declared ones.
	#
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h)
		list(APPEND _hdrs ${tla}.h)
	endif()
	file(GLOB _fullfoundhdrs "${COMPONENT_INSTALL_DIR}/*.h")
	foreach(arg ${_fullfoundhdrs})
		get_filename_component(_name ${arg} NAME)
		list(APPEND _foundhdrs ${_name})
	endforeach()
	if(_foundhdrs)
		foreach(arg ${ARGN})
			get_filename_component(_name ${arg} NAME)
			list(APPEND _hdrs ${_name})
		endforeach()

		foreach(arg ${_hdrs})
			get_filename_component(_name ${arg} NAME)
			list(REMOVE_ITEM _foundhdrs ${_name})
		endforeach()

		foreach(arg ${_foundhdrs})
			message("Purging dead header file: ${COMPONENT_INSTALL_DIR}/${arg}")
			execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${arg})
		endforeach()
	endif()

	#
	# Copy headers
	#
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h ${COMPONENT_INSTALL_DIR}/${tla}.h COPYONLY)
	endif()
	foreach(arg ${ARGN})
		get_filename_component(_name ${arg} NAME)
		string(REGEX MATCH "^arch/" _op ${arg})
		if(_op)
			set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${arg})
		else()
			string(REGEX MATCH "/" _op ${arg})
			if(_op)
				set(_src ${arg})
			else()
				set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${arg})
			endif()
	   	endif()
		configure_file(${_src} ${COMPONENT_INSTALL_DIR}/${_name} COPYONLY)
	endforeach()

	if("${libname}" MATCHES "adtf")
		add_custom_command(TARGET ${libname}
							POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.plb ${COMPONENT_INSTALL_DIR}
            	    		COMMENT "Install ADTF plugin ${TLA}")
		add_custom_target(deepclean_${TLA}
							COMMAND ${CMAKE_COMMAND} -E echo "Deep clean ${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}"
							COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.plb
							COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.plb)
	elseif("${libname}" MATCHES "rtmaps")
		add_custom_command(TARGET ${libname}
							POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.dll ${COMPONENT_INSTALL_DIR}
            	    		COMMENT "Install RTmaps ${TLA}")
		add_custom_target(deepclean_${TLA}
							COMMAND ${CMAKE_COMMAND} -E echo "Deep clean ${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}"
							COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.dll
							COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.dll)
	elseif("${libname}" MATCHES "i386_win_d")
		add_custom_command(TARGET ${libname}
							POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX} ${COMPONENT_INSTALL_DIR}
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}.pdb ${COMPONENT_INSTALL_DIR}
            	    		COMMENT "Install library ${TLA}")
		add_custom_target(deepclean_${TLA}
							COMMAND ${CMAKE_COMMAND} -E echo "Deep clean ${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}"
							COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}
							COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX})
	else()
		add_custom_command(TARGET ${libname}
							POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX} ${COMPONENT_INSTALL_DIR}
            	    		COMMENT "Install library ${TLA}")
		add_custom_target(deepclean_${TLA}
							COMMAND ${CMAKE_COMMAND} -E echo "Deep clean ${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}"
							COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/lib/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}
							COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX})
	endif()

	add_custom_target(revert_${TLA}
						COMMAND ${CMAKE_COMMAND} -E echo "Reverting ${COMPONENT_INSTALL_DIR}"
						COMMAND svn revert -R ${COMPONENT_INSTALL_DIR})
endmacro()

#
# Copy public header(s) and executable and diagram to install directory.
# Note: Headers are copied at CMake config time, rest at build time.
#		Optional third argument is public header list.
#
macro(COMPONENT_INSTALL_EXE tla exename)
	if(NOT COMPONENT_INSTALL_DIR)
		set(COMPONENT_INSTALL_DIR COMPONENT_INSTALL_DIR_NOT_SET)
	endif()

	file(MAKE_DIRECTORY ${COMPONENT_INSTALL_DIR})

	#
	# Purge all headers as there is a possibility they may have been deleted
	#
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h)
		list(APPEND _hdrs ${tla}.h)
	endif()
	file(GLOB _fullfoundhdrs "${COMPONENT_INSTALL_DIR}/*.h")
	foreach(arg ${_fullfoundhdrs})
		get_filename_component(_name ${arg} NAME)
		list(APPEND _foundhdrs ${_name})
	endforeach()
	if(_foundhdrs)
		foreach(arg ${ARGN})
			get_filename_component(_name ${arg} NAME)
			list(APPEND _hdrs ${_name})
		endforeach()

		foreach(arg ${_hdrs})
			get_filename_component(_name ${arg} NAME)
			list(REMOVE_ITEM _foundhdrs ${_name})
		endforeach()

		foreach(arg ${_foundhdrs})
			message("Purging dead header file: ${COMPONENT_INSTALL_DIR}/${arg}")
			execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${arg})
		endforeach()
	endif()

	#
	# Copy headers
	#
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${tla}.h ${COMPONENT_INSTALL_DIR}/${tla}.h COPYONLY)
	endif()
	foreach(arg ${ARGN})
		get_filename_component(_name ${arg} NAME)
		string(REGEX MATCH "^arch/" _op ${arg})
		if(_op)
			set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${arg})
		else()
			string(REGEX MATCH "/" _op ${arg})
			if(_op)
				set(_src ${arg})
			else()
				set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${arg})
			endif()
	   	endif()
		configure_file(${_src} ${COMPONENT_INSTALL_DIR}/${_name} COPYONLY)
	endforeach()

	if("${COMPONENT_TYPE}" MATCHES "adtf")
		message(FATAL_ERROR "Please use COMPONENT_INSTALL instead of COMPONENT_INSTALL_EXE for ADTF plugins")
	elseif("${COMPONENT_TYPE}" MATCHES "rtmaps")
		message(FATAL_ERROR "Please use COMPONENT_INSTALL instead of COMPONENT_INSTALL_EXE for RTmaps")
	else()
		add_custom_command(TARGET ${exename}
							POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/image/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${exename}${CMAKE_EXECUTABLE_SUFFIX} ${COMPONENT_INSTALL_DIR}
            	    		COMMENT "Install executable ${TLA}")
		add_custom_target(deepclean_${TLA}
							COMMAND ${CMAKE_COMMAND} -E echo "Deep clean ${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}"
							COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/image/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${exename}${CMAKE_EXECUTABLE_SUFFIX}
							COMMAND ${CMAKE_COMMAND} -E remove ${COMPONENT_INSTALL_DIR}/${COMPONENT_LIB_INTERMEDIATE_DIR}${CMAKE_STATIC_LIBRARY_PREFIX}${exename}${CMAKE_EXECUTABLE_SUFFIX})
	endif()

	add_custom_target(revert_${TLA}
						COMMAND ${CMAKE_COMMAND} -E echo "Reverting ${COMPONENT_INSTALL_DIR}"
						COMMAND svn revert -R ${COMPONENT_INSTALL_DIR})
endmacro()

