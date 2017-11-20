# - Extend CMake with support for Lint, Understand etc.
# Lint must be found (why are you using this file if you don't have it?).
# Understand is optional and need not be found.
#
# Macros:
#
#  EXTEND_ADD_LIBRARY		- Extend add_library() to support Lint, Understand and ASLFSM.
#  EXTEND_ADD_EXECUTABLE	- Extend add_executable() to support Lint, Understand and ASLFSM.
#
#  EXTEND_UNDERSTAND_LIB	- True/Fasle: Generate Understand C database for a library
#

#
# Add assembler support if defined by toolchain. We'd really like to put this directly in the
# toolchain file but the causes CMake to lockup, so we set a variable in the toolchain and use
# it here instead.
#
if(EXTEND_ASM)
	enable_language(${EXTEND_ASM})
endif()

find_package(Lint REQUIRED)
include("UseLint")

find_package(Understand)
include("UseUnderstand")

#
# Take a library name(e.g. "hdw") and a list of source files,
# generate custom rules for Lint(if enabled), and add targets
# for library and Lint dummy targets(if required).
# Add understand file list.
#
macro(EXTEND_ADD_LIBRARY _libname)
	#
	# Build ASL-FSM first as it will generate source files,
	# which are included by Lint and Understand.
	#
#	generate_custom_aslfsm(${_libname} ${ARGN})
	lint_add_custom_cmd(${_libname} ${ARGN} ${SCD_LIST})

#	set(found FALSE)
#	foreach(arg ${LINTALL_IGNORE_LIBS})
#		if(${_libname} STREQUAL ${arg})
#			set(found TRUE)
#		endif()
#	endforeach()
#	if(found)
#		message("Lint of all files will ignore module '${_libname}', see LINTALL_IGNORE_LIBS")
#	else()
#		lint_generate_files(${_libname} ${ARGN} ${SCD_LIST})
#	endif()

	understand_generate_files(${_libname} ${ARGN} ${SCD_LIST})	
	if (${EXTEND_UNDERSTAND_LIB})
		understand_add_custom_cmd()
		add_library(${_libname} ${ARGN} ${LINT_LIST} ${SCD_LIST} ${UNDERSTAND_TARGET} )
	else()
		add_library(${_libname} ${ARGN} ${LINT_LIST} ${SCD_LIST})
	endif()
endmacro()


#
# Take an executable name e.g. "app") and a list of source files,
# generate custom rules for Lint (if enabled), and add target
# for the executable and Lint dummy targets (if required).
# Add understand file list.
#
macro(EXTEND_ADD_EXECUTABLE _exename)
	#
	# Build ASL-FSM first as it will generate source files,
	# which are included by Lint and Understand.
	#
#	generate_custom_aslfsm(${_exename} ${ARGN})
	lint_add_custom_cmd(${_exename} ${ARGN} ${SCD_LIST})
	understand_generate_files(${_exename} ${ARGN} ${SCD_LIST})
	understand_add_custom_cmd()
	lintall_add_custom_cmd()

	add_executable(${_exename} ${ARGN} ${LINT_LIST} ${SCD_LIST} ${UNDERSTAND_TARGET} ${LINT_TARGET})
endmacro()
