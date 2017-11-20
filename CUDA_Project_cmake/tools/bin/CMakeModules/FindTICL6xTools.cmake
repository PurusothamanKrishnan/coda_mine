# - Find TI CL6X Tools
# This module looks for TI DSP C6x Tools.
#
# This module defines:
#
# TICL6XTools_BIOS_DIR		- TI SYS BIOS (e.g. "C:/Program Files/Texas Instruments/bios_6_21_00_13")
# TICL6XTools_XDC_DIR	    - TI XDC Tools (e.g. "C:/Program Files/Texas Instruments/xdctools_3_16_02_32")
# TICL6XTools_IPC_DIR	    - TI IPC Tools (e.g. "C:/Program Files/Texas Instruments/ipc_1_00_05_60")
# TICL6XTOOLS_PSP_DIR		- TI BIOS PSP (e.g. "C:/Program Files/Texas Instruments/pspdrivers_02_20_00_02")
# TICL6XTOOLS_EDMA_DIR		- TI EDMA (e.g. "C:/Program Files/Texas Instruments/edma3_lld_02_10_03_04")
# TICL6XTools_XDAIS_DIR    	- TI XDAIS Tools (e.g. "C:/Program Files/Texas Instruments/xdais_6_25_01_08")
# TICL6XTools_FC_DIR    	- TI Framework (e.g. "C:/Program Files/Texas Instruments/framework_components_3_20_00_13_eng")

#
# TI BIOS version
#
if(NOT DEFINED TICL6XTOOLS_BIOS_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_BIOS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# XDC Tools
#
if(NOT DEFINED TICL6XTOOLS_XDC_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_XDC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# IPC Tools
#
if(NOT DEFINED TICL6XTOOLS_IPC_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_IPC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# BIOS PSP
#
if(NOT DEFINED TICL6XTOOLS_PSP_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_PSP_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# EDMA
#
if(NOT DEFINED TICL6XTOOLS_EDMA_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_EDMA_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# XDAIS Tools
#
if(NOT DEFINED TICL6XTOOLS_XDAIS_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_XDAIS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# Framework Components
#
if(NOT DEFINED TICL6XTOOLS_FC_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_FC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# Codec Engine
#
if(NOT DEFINED TICL6XTOOLS_CODEC_ENGINE_DIR)
	message(FATAL_ERROR "TICL6XTOOLS_CODEC_ENGINE_DIR has not been set in config file, set to \"\" if not used")
endif()

include(FindPackageHandleStandardArgs)

# We need to capitalise the first letter of the release notes name for Linux to work
string(SUBSTRING ${TICL6XTOOLS_BIOS_DIR} 0 1 _tmp_bios_name)
string(TOUPPER ${_tmp_bios_name} _tmp_bios_name)
string(LENGTH ${TICL6XTOOLS_BIOS_DIR} _len)
math(EXPR _len ${_len}-1)
string(SUBSTRING ${TICL6XTOOLS_BIOS_DIR} 1 ${_len} _tmp_bios_name_end)
set(_tmp_bios_name ${_tmp_bios_name}${_tmp_bios_name_end})

if(NOT ${TICL6XTOOLS_BIOS_DIR} STREQUAL "")
	find_path(TICL6XTools_BIOS_PATH "${TICL6XTOOLS_BIOS_DIR}_release_notes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_BIOS_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_BIOS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_BIOS_PATH)
	mark_as_advanced(TICL6XTools_BIOS_PATH)
	if(${TICL6XTools_BIOS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_BIOS_PATH} ${TICL6XTOOLS_BIOS_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_XDC_DIR} STREQUAL "")
	find_path(TICL6XTools_XDC_PATH "${TICL6XTOOLS_XDC_DIR}_release_notes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_XDC_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_XDC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_XDC_PATH)
	mark_as_advanced(TICL6XTools_XDC_PATH)
	if(${TICL6XTools_XDC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_XDC_PATH} ${TICL6XTOOLS_XDC_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_XDAIS_DIR} STREQUAL "")
	find_path(TICL6XTools_XDAIS_PATH "${TICL6XTOOLS_XDAIS_DIR}_eng_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_XDAIS_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_XDAIS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_XDAIS_PATH)
	mark_as_advanced(TICL6XTools_XDAIS_PATH)
	if(${TICL6XTools_XDAIS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_XDAIS_PATH} ${TICL6XTOOLS_XDAIS_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_IPC_DIR} STREQUAL "")
	find_path(TICL6XTools_IPC_PATH "ipc_1_23_05_40_asl_release_notes.html"
				PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_IPC_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_IPC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_IPC_PATH)
	mark_as_advanced(TICL6XTools_IPC_PATH)
	if(${TICL6XTools_IPC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_IPC_PATH} ${TICL6XTOOLS_IPC_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_PSP_DIR} STREQUAL "")
	find_path(TICL6XTools_PSP_PATH "Software-manifest.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_PSP_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_PSP_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_PSP_PATH)
	mark_as_advanced(TICL6XTools_PSP_PATH)
	if(${TICL6XTools_PSP_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_PSP_PATH} ${TICL6XTOOLS_PSP_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_EDMA_DIR} STREQUAL "")
	find_path(TICL6XTools_EDMA_PATH "release_notes_edma3_lld_02_11_03.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_EDMA_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_EDMA_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_EDMA_PATH)
	mark_as_advanced(TICL6XTools_EDMA_PATH)
	if(${TICL6XTools_EDMA_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_EDMA_PATH} ${TICL6XTOOLS_EDMA_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_FC_DIR} STREQUAL "")
	find_path(TICL6XTools_FC_PATH "framework_components_3_21_03_34_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_FC_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_FC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_FC_PATH)
	mark_as_advanced(TICL6XTools_FC_PATH)
	if(${TICL6XTools_FC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_FC_PATH} ${TICL6XTOOLS_FC_DIR}")
	endif()
endif()

if(NOT ${TICL6XTOOLS_CODEC_ENGINE_DIR} STREQUAL "")
	find_path(TICL6XTools_CODEC_ENGINE_PATH "${TICL6XTOOLS_CODEC_ENGINE_DIR}_ReleaseNotes.html"
				PATHS
				"$ENV{CCS_PATH}/${TICL6XTOOLS_CODEC_ENGINE_DIR}"
				"$ENV{CCS_PATH}/../${TICL6XTOOLS_CODEC_ENGINE_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL6XTools DEFAULT_MSG TICL6XTools_CODEC_ENGINE_PATH)
	mark_as_advanced(TICL6XTools_CODEC_ENGINE_PATH)
	if(${TICL6XTools_CODEC_ENGINE_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL6XTools_CODEC_ENGINE_PATH} ${TICL6XTOOLS_CODEC_ENGINE_DIR}")
	endif()
endif()
