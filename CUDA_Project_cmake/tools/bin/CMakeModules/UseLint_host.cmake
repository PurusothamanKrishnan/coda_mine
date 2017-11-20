# - Use Module for Lint
# Sets up C and C++ to use Lint.
# It is assumed that "find_package(Lint)" has been performed.
# The header file include file list will use the same setting as the compiler.
# The Lint ".lnt" control files will be searched for in Lint_INCLUDE_DIRS (as set by FindLint).
#
# This module allows the user to customise behaviour via:
#
#  LINTING	 			- Enable/disable Lint for individual files at compile time (default to True)
#  LINTING_STRICT		- Control Lint return code, False uses "-zero" to never cause build termination (defaults to True).
#  LINTING_ALL			- Enable/disable Lint of all source files at once at link time (defaults to False).
#  LINTING_PREPROCESSED	- Generate preprocessed source file (default to False), for "file.c" generates "lint_file.c".
#  LINTING_LOG			- If set redirects Lint output to specified log file, overrides LINTING setting.
#
#  ENV("LINT_NONE")		- If "LINT_NONE" enviroment variable is present at CMake time automatically set LINTING False.
#
#  LINT_ALL_IGNORE_LIBS	- May be set by user to a list of libraries that should be excluded from Lint all.
#
#  LINT_FLAGS_COMMON	- Default flags always passed(defaults to "-b -v")
#  LINT_FLAGS_UNIT		- Default flags for unit(single source) Linting, e.g. "-u".
#  LINT_FLAGS_GROUP		- Default flags for multiple source Linting.
#  LINT_ARCH			- Specific architecture search for, e.g. "omapl138_dsp", used by FindLink.cmake for locating "lintarch.lnt"
#  LINT_FORCED			- Internal variable set by LINTING_LOG to force Linting on.
#
# Macros:
#
#  LINT_ADD_CUSTOM_CMD	- Generate custom commands to perform Lint depending on source files generated object file,
#                         this ensures Lint is rebuilt if any header dependencies change. Returns LINT_LIST which
#                         should be added as a dependency in add_library() or extend_add_library().
#
#  LINTALL_ADD_CUSTOM_CMD - Generate custom command to perform Lint of all files at once, returns LINT_TARGET which
#                           should be added as a dependency in add_executable() or extend_add_executable().
#
# Default user options.
#
if(NOT DEFINED LINTING)
	set(LINTING FALSE)
endif()

if(NOT DEFINED LINTING_STRICT)
	set(LINTING_STRICT TRUE)
endif()

if(NOT DEFINED LINTING_ALL)
	set(LINTING_ALL FALSE)
endif()

if(NOT DEFINED LINTING_PREPROCESSED)
	set(LINTING_PREPROCESSED FALSE)
endif()

#
# User can define LINT_NONE to suppress Lint.
#
if($ENV{LINT_NONE})
	set(LINTING FALSE)
	message("Ignoring Lint as environment variable LINT_NONE is set.")
endif()

if(LINTING_LOG)
	#
	# Redirect output to a log file, include Lint version banner and processed filename in output, don't stop on error.
	# Note appends to output filename, user is responsible for initial delete.
	#
	message("Forcing Lint enabled and appending all output to ${LINTING_LOG}")
	set(LINT_FLAGS_COMMON +os[${LINTING_LOG}] ++b +vm -zero -e830 -I\"${Lint_INCLUDE_DIRS}\" -I\"${Lint_ARCH_PATH}\" lintarch.lnt local.lnt)
	set(LINT_FORCED TRUE)
elseif(NOT DEFINED LINT_FLAGS_COMMON)
	#
	# Make Lint as quiet as possible.
	#
	set(LINT_FLAGS_COMMON -b -v -I\"${Lint_INCLUDE_DIRS}\" -I\"${Lint_ARCH_PATH}\" lintarch.lnt local.lnt)
	set(LINT_FORCED FALSE)
endif()
set(LINT_FLAGS_COMMON ${LINT_FLAGS_COMMON} CACHE STRING "Lint flags that are common between unit and group style Linting.")

#
# Flags for single source file unit checkout.
#
if(NOT DEFINED LINT_FLAGS_UNIT)
	set(LINT_FLAGS_UNIT -u)
endif()
set(LINT_FLAGS_UNIT ${LINT_FLAGS_UNIT} CACHE STRING "Lint flags that are specific for unit(single) file checkout only.")

#
# Flags for group(multipe file) checkout.
#
if(NOT DEFINED LINT_FLAGS_GROUP)
	set(LINT_FLAGS_GROUP -zero -e830 lintall.lnt)
endif()
set(LINT_FLAGS_GROUP ${LINT_FLAGS_GROUP} CACHE STRING "Lint flags that are specific for group multi-file checkout only.")

#
# LINT_ADD_CUSTOM_CMD(library name, source file list)
#
# Takes a library name(e.g. "hdw") and a list of sources files(e.g. hdw.cpp hdwio.cpp),
# and generates a list of custom build rules that depend on the object file built from
# each source as this is the only way to get the dependencies on any included headers, but
# a side effect is that Lint runs after the target compile.
#
# Returns LINT_LIST containing a list of dummy targets ending in *.lint, these targets should
# be added to the ultimate build target via ADD_LIBRARY() or ADD_EXECUTABLE().
#
macro(LINT_ADD_CUSTOM_CMD _libname)
	set(LINT_LIST "")

	if(LINTING OR LINT_FORCED)
		set(_op_dir ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_libname}.dir)
		set(_op_dir_phoney ${CMAKE_CURRENT_BINARY_DIR})

		set(_flags ${LINT_FLAGS_COMMON} ${LINT_FLAGS_UNIT} ${LINT_DEFINES})

		if(NOT LINTING_STRICT)
			set(_flags ${_flags} -zero)
		endif(NOT LINTING_STRICT)

		# Obtain include path list and generate Lint's -I list
		# Add this first so we always search source directory for local Lint files first.
		get_directory_property(_inc_paths INCLUDE_DIRECTORIES)
		set(_inc_list "")
		foreach(_inc ${_inc_paths})
			set(_inc_list ${_inc_list} -I${_inc})
		endforeach()
		set(_flags ${_inc_list} ${_flags})

		# Obtain defines list and generate Lint's -D list.
		# Note: This relies on the compiler also using -D for define definition.
		get_directory_property(_defs COMPILE_DEFINITIONS)
		set(_def_list "")
		foreach(_def ${_defs})
			set(_def_list ${_def_list} -D${_def})
		endforeach()
		set(_flags ${_flags} ${_def_list})

		foreach(arg ${ARGN})
			if(${arg} MATCHES "\\.c\$" OR ${arg} MATCHES "\\.cpp\$")
				if(${arg} MATCHES "\\.\\.")
					#message("LINT IGNORING RELATIVE PATH:" ${arg})
				else()
					get_filename_component(_name ${arg} NAME_WE)
					get_filename_component(_path ${arg} PATH)
					get_filename_component(_src_ext ${arg} EXT)
					set (_obj_ext ${CMAKE_C_OUTPUT_EXTENSION})

					if(UNIX)
						# Pass
					else()
						# Is it a full DOS path containing drive letter?
						if(${arg} MATCHES ":")
							string(REPLACE ":" "_" _path ${_path})
						endif()
					endif()

					set(_depends ${_op_dir}/${_path}/${_name}${_src_ext}${_obj_ext})

					if(LINTING_PREPROCESSED)
						add_custom_command(	OUTPUT ${_op_dir_phoney}/${_name}.lint ${_op_dir_phoney}/lint_${_name}${_src_ext}
											COMMAND ${Lint_EXECUTABLE} ${_flags} -p ${arg} > ${_op_dir_phoney}/lint_${_name}${_src_ext}
											COMMAND ${Lint_EXECUTABLE} ${_flags} ${arg}
											COMMAND touch ${_op_dir_phoney}/${_name}.lint
											DEPENDS ${_depends}
											WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
											COMMENT "Linting ${arg} and preprocessing to ${_op_dir_phoney}/lint_${_name}${_src_ext}" )
					else()
						add_custom_command(	OUTPUT ${_op_dir_phoney}/${_name}.lint
											COMMAND ${Lint_EXECUTABLE} ${_flags} ${arg}
											COMMAND touch ${_op_dir_phoney}/${_name}.lint
											DEPENDS ${_depends}
											WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
											COMMENT "Linting ${arg}" )
					endif()

					set(LINT_LIST ${LINT_LIST} ${_op_dir_phoney}/${_name}.lint)
				endif()
			endif()
		endFOREACH(arg)
	endif()

	_lint_generate_files(${_libname} ${ARGN})
	#message("LINT_LIST: ${LINT_LIST}")
endmacro()


#
# Read the list of *.files created by macro LINT_GENERATE_FILES and
# create one huge list of all the source files in the project; then create a custom
# build rule that depends on all the sources to allow Linting of all source files.
# Externally GenericBuildEpilogue.cmake implements LINTALL_IGNORE_LIBS which can
# be used to exclude third party modules.
#
macro(LINTALL_ADD_CUSTOM_CMD)
	if(LINTING_ALL)
		set(_group_flags ${LINT_FLAGS_COMMON} ${LINT_FLAGS_GROUP} ${LINT_DEFINES})

		# Obtain include path list and generate Lint's -I list
		get_directory_property(_inc_paths INCLUDE_DIRECTORIES)
		set(_inc_list "")
		foreach(_inc ${_inc_paths})
			set(_inc_list ${_inc_list} -I${_inc})
		endforeach()
		set(_group_flags ${_group_flags} ${_inc_list})

		# Obtain defines list and generate Lint's -D list.
		# Note: This relies on the compiler also using -D for define definition.
		get_directory_property(_defs COMPILE_DEFINITIONS)
		set(_def_list "")
		foreach(_def ${_defs})
			set(_def_list ${_def_list} -D${_def})
		endforeach()
		set(_group_flags ${_group_flags} ${_def_list})

		file(GLOB _files "${CMAKE_BINARY_DIR}/tmp/*.lint_files")
		file(WRITE "${CMAKE_BINARY_DIR}/tmp/lint.filelist.lnt" "")
		set(_allsources "")
		foreach(_name ${_files})
			file(READ ${_name} _contents)
			foreach(_asrc ${_contents})
				file(APPEND "${CMAKE_BINARY_DIR}/tmp/lint.filelist.lnt" "${_asrc}\n")
			endforeach()

			set(_allsources ${_allsources} ${_contents})
		endforeach()

  		add_custom_command(	OUTPUT ${_op_dir_phoney}/lintall.lint
                     		COMMAND ${Lint_EXECUTABLE} ${_group_flags} "${CMAKE_BINARY_DIR}/tmp/lint.filelist.lnt" ${ARGN}
                    		COMMAND touch ${_op_dir_phoney}/lintall.lint
							DEPENDS ${LINT_LIST}
							WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
							COMMENT "Linting combined source")
		set(LINT_TARGET ${_op_dir_phoney}/lintall.lint)
	endif()
endmacro()


#
# Given a source list generate a file in the output directory called ".../${_basename}.lint_files
# of full paths to the files as CMake semicolon separated list. If a full path is passed
# it is used directly, if no path is present the current source directory is appended.
# This is used when Linting the entire source together.
#
macro(_LINT_GENERATE_FILES _basename)
	if(LINTING OR LINT_FORCED)
		file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)

		set(found FALSE)
		foreach(arg ${LINT_ALL_IGNORE_LIBS})
			if(${_basename} STREQUAL ${arg})
				set(found TRUE)
			endif()
		endforeach()
		if(found)
			message("Lint all will ignore library '${_basename}', see LINT_ALL_IGNORE_LIBS")
		else()
			set(_fullsrc "")
			foreach(_name ${ARGN})
				if(${_name} MATCHES "\\.c")
					set(_is_root_pathname FALSE)
					if(UNIX)
						string(SUBSTRING ${_name} 0 1 _first_char)
						if("${_first_char}" STREQUAL "/")
							set(_is_root_pathname TRUE)
						endif()
					else()
						# Is it a full DOS path containing drive letter?
						if(${_name} MATCHES ":")
							set(_is_root_pathname TRUE)
						endif()
					endif()

					if(_is_root_pathname)
						set(_fullsrc ${_fullsrc} ${_name})
					else()
						set(_fullsrc ${_fullsrc} ${CMAKE_CURRENT_SOURCE_DIR}/${_name})
					endif()
				endif()
			endforeach()
			file(WRITE "${CMAKE_BINARY_DIR}/tmp/${_basename}.lint_files" "${_fullsrc}")
		endif()
	endif()
endmacro()
#
# Setup Lint logging if enabled
#
if(${CMAKE_BINARY_DIR} MATCHES "lintlog")
	if(${CMAKE_GENERATOR} MATCHES "JOM")
		message(FATAL_ERROR "For Lint Logging use _nmake to ensure single processor build")
	endif()
	set(LINTING_LOG "${CMAKE_BINARY_DIR}/DAG_SVS_lint_${BUILDNAME}.log")
endif()

