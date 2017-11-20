# - Use Module for UnderstandC
# Sets up C and C++ to generate UnderstandC/C++ database using command line "und".
# It is assumed that "find_package(Und)" has been performed.
#
# This module allows the user to customise behaviour via:
#
#  UNDERSTAND	- Enable/disable Understand database generation (default to True)
#
#  UNDERSTAND_INCLUDE_ADDFOUND		- Add any includes found
#  UNDERSTAND_INCLUDE_ADDFOUNDSYS	- Add any system includes found
#
# Macros:
#
#  UNDERSTAND_GENERATE_FILES	- Generates a file list for a library for later use by
#                                 UNDERSTAND_ADD_CUSTOM_CMD.
#
#  UNDERSTAND_ADD_CUSTOM_CMD	- Generate custom command to create/refresh database,
#                                 returns UNDERSTAND_TARGET which should be added as a
#                                 dependency to add_executable() or extend_add_executable().
#
# Default user options.
#
if(NOT DEFINED UNDERSTAND)
	set(UNDERSTAND TRUE)
endif()

if(NOT DEFINED UNDERSTAND_INCLUDE_ADDFOUND)
	set(UNDERSTAND_INCLUDE_ADDFOUND TRUE)
endif()

if(NOT DEFINED UNDERSTAND_INCLUDE_ADDFOUNDSYS)
	set(UNDERSTAND_INCLUDE_ADDFOUNDSYS FALSE)
endif()

#
# Given a source list generate a file in the output directory called ".../${_basename}.understand_files
# of full paths to the files as CMake semicolon separated list. If a full path is passed
# it is used directly, if no path is present the current source directory is appended.
# This is used when building an Understand C database.
#
macro(UNDERSTAND_GENERATE_FILES _basename)
	if(UNDERSTAND AND UNDERSTAND_FOUND)
		set(_fullsrc "")
		foreach(_name ${ARGN})
			if(${_name} MATCHES "\\.c")

				set(_is_root FALSE)
				if(UNIX)
					string(SUBSTRING ${_name} 0 1 _first_char)
					if("${_first_char}" STREQUAL "/")
						set(_is_root TRUE)
					endif()
				else()
					# Is it a full DOS path containing drive letter?
					string(SUBSTRING ${_name} 1 1 _second_char)
					if("${_second_char}" STREQUAL ":")
						set(_is_root TRUE)
					endif()
				endif()

				if(_is_root)
					set(_fullsrc ${_fullsrc} ${_name})
				else()
					set(_fullsrc ${_fullsrc} ${CMAKE_CURRENT_SOURCE_DIR}/${_name})
				endif()
			endif()
		endforeach()
		file(WRITE "${CMAKE_BINARY_DIR}/tmp/${_basename}.understand_files" "${_fullsrc}")
	endif()
endmacro()

#
# Read the list of *.understand_files created by macro UNDERSTAND_GENERATE_FILES and
# create one huge list of all the source files in the project; then create a custom
# build rule that depends on all the sources to build an Understand C database.
# UNDERSTAND_TARGET is set to the database name and should be used as the target for
# add_executable() or our extend_add_executable().
#
macro(UNDERSTAND_ADD_CUSTOM_CMD)
	if(UNDERSTAND AND UNDERSTAND_FOUND)
		file(GLOB _files "${CMAKE_BINARY_DIR}/tmp/*.understand_files")

		# Build include path response file.
		file(REMOVE ${CMAKE_BINARY_DIR}/tmp/understand_includes.rsp)
		get_directory_property(_inc_paths INCLUDE_DIRECTORIES)
		set(_inc_list "")
		foreach(_inc ${_inc_paths})
			file(APPEND ${CMAKE_BINARY_DIR}/tmp/understand_includes.rsp ${_inc} "\n")
		endforeach()

		# Build defines list
		get_directory_property(_defs COMPILE_DEFINITIONS)
		set(_def_list "")
		foreach(_def ${_defs})
			set(_def_list ${_def_list} -D${_def})
		endforeach()

		# Build source file list response file.
		file(WRITE "${CMAKE_BINARY_DIR}/tmp/understand.filelist" "")
		set(_allsources "")
		foreach(_name ${_files})
			file(READ ${_name} _contents)
			foreach(_asrc ${_contents})
				file(APPEND "${CMAKE_BINARY_DIR}/tmp/understand.filelist" "${_asrc}\n")
			endforeach()

			set(_allsources ${_allsources} ${_contents})
		endforeach()

		set(_db 		"${CMAKE_BINARY_DIR}/${PROJECT_NAME}_${CMAKE_BUILD_TYPE}.udb")
		set(_db_phoney	"${CMAKE_BINARY_DIR}/tmp/${PROJECT_NAME}_${CMAKE_BUILD_TYPE}.udb.created")

		set(_flags "")
		if(UNDERSTAND_INCLUDE_ADDFOUND)
			set(_flags ${_flags} -include_addfound On)
		endif()
		if(UNDERSTAND_INCLUDE_ADDFOUNDSYS)
			set(_flags ${_flags} -include_addfoundsys On)
		endif()

		add_custom_command(	OUTPUT ${_db_phoney}
				  			COMMAND ${Understand_EXECUTABLE} ARGS -quiet On -create -languages C++ -db ${_db} -define ${_defs} ${_flags} -include @tmp/understand_includes.rsp -addFiles @tmp/understand.filelist
							COMMAND touch ${_db_phoney}
				  			DEPENDS ${_files}
							WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
				  			COMMENT "Creating Understand C database ${PROJECT_NAME}_${CMAKE_BUILD_TYPE}.udb"
				  	  	)

		add_custom_command(	OUTPUT ${_db}
				  			COMMAND ${Understand_EXECUTABLE} ARGS -quiet On -db ${_db} -analyze
				  			DEPENDS ${_allsources} ${_db_phoney}
							WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
				  			COMMENT "Refreshing Understand C database ${PROJECT_NAME}_${CMAKE_BUILD_TYPE}.udb"
				  	  	)

		set(UNDERSTAND_TARGET ${_db})
	endif()
endmacro()
