# - Find TI CL470 Tools
# This module looks for TI ARM TMS470 Tools.
#
# This module defines:
#
# TICL470TOOLS_BIOS_DIR     - TI SYS BIOS (e.g. "C:/Program Files/Texas Instruments/bios_6_21_00_13")
# TICL470TOOLS_XDC_DIR	    - TI XDC Tools (e.g. "C:/Program Files/Texas Instruments/xdctools_3_16_02_32")
# TICL470TOOLS_IPC_DIR	    - TI IPC Tools (e.g. "C:/Program Files/Texas Instruments/ipc_1_00_05_60")
# TICL470TOOLS_PSP_DIR		- TI BIOS PSP (e.g. "C:/Program Files/Texas Instruments/pspdrivers_02_20_00_02")
# TICL470TOOLS_EDMA_DIR		- TI EDMA (e.g. "C:/Program Files/Texas Instruments/edma3_lld_02_10_03_04")
# TICL470TOOLS_XDAIS_DIR    - TI XDAIS Tools (e.g. "C:/Program Files/Texas Instruments/xdais_6_25_01_08")
# TICL470TOOLS_FC_DIR    	- TI Framework (e.g. "C:/Program Files/Texas Instruments/framework_components_3_20_00_13_eng")
# TICL470TOOLS_IVAHD_JPEGVDEC_DIR - TI IVAHD jpeg decoder (C:/Program Files/Texas Instruments/ivahd_jpegvdec_01_00_00_00_production
# TICL470TOOLS_HDVPSS_DIR    	- TI Framework (e.g. "C:/Program Files/Texas Instruments/hdvpss_01_00_01_26")
# TICL470TOOLS_NDK_DIR      - TI NDK (e.g. "C:/Program Files/Texas Instruments/ndk_2_20_03_24")
# TICL470TOOLS_NSP_DIR      - TI NSP (eg. "C:/Program Files/Texas Instruments/nsp_1_00_00_05_eng_centaurus_rel")
# TICL470TOOLS_AVB_DIR      - TI AVB stack (eg. "C:/Program Files/Texas Instruments/AVB")
# TICL470Tools_IVAHD_HDVICP20_DIR - TI IVAHD hdvicp2 (C:/Program Files/Texas Instruments/ivahd_hdvicp20api_01_00_00_19_production
# TI BIOS version
#
if(NOT DEFINED TICL470TOOLS_BIOS_DIR)
	message(FATAL_ERROR "TICL470TOOLS_BIOS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# XDC Tools
#
if(NOT DEFINED TICL470TOOLS_XDC_DIR)
	message(FATAL_ERROR "TICL470TOOLS_XDC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# IPC Tools
#
if(NOT DEFINED TICL470TOOLS_IPC_DIR)
	message(FATAL_ERROR "TICL470TOOLS_IPC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# BIOS PSP
#
if(NOT DEFINED TICL470TOOLS_PSP_DIR)
	message(FATAL_ERROR "TICL470TOOLS_PSP_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# EDMA
#
if(NOT DEFINED TICL470TOOLS_EDMA_DIR)
	message(FATAL_ERROR "TICL470TOOLS_EDMA_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# XDAIS Tools
#
if(NOT DEFINED TICL470TOOLS_XDAIS_DIR)
	message(FATAL_ERROR "TICL470TOOLS_XDAIS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# Framework Components
#
if(NOT DEFINED TICL470TOOLS_FC_DIR)
	message(FATAL_ERROR "TICL470TOOLS_FC_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# IVAHD Jpeg Decoder
#
if(NOT DEFINED TICL470TOOLS_IVAHD_JPEGVDEC_DIR)
  message(FATAL_ERROR "TICL470TOOLS_IVAHD_JPEGVDEC has not been set in config file, set to \"\" if not used")
endif()

#
# HDVPSS drivers
#
if(NOT DEFINED TICL470TOOLS_HDVPSS_DIR)
	message(FATAL_ERROR "TICL470TOOLS_HDVPSS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# ISS drivers
#
if(NOT DEFINED TICL470TOOLS_ISS_DIR)
	message(FATAL_ERROR "TICL470TOOLS_ISS_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# NDK
#
if(NOT DEFINED TICL470TOOLS_NDK_DIR)
	message(FATAL_ERROR "TICL470TOOLS_NDK_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# NSP
#
if(NOT DEFINED TICL470TOOLS_NSP_DIR)
	message(FATAL_ERROR "TICL470TOOLS_NSP_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# AVB
#
if(NOT DEFINED TICL470TOOLS_AVB_DIR)
	message(FATAL_ERROR "TICL470TOOLS_AVB_DIR has not been set in config file, set to \"\" if not used")
endif()

#
# IVAHD HDVICP20
#
if(NOT DEFINED TICL470TOOLS_IVAHD_HDVICP20_DIR)
  message(FATAL_ERROR "TICL470TOOLS_IVAHD_HDVICP20 has not been set in config file, set to \"\" if not used")
endif()

#
# CODEC ENGINE
#
if(NOT DEFINED TICL470TOOLS_CODEC_ENGINE_DIR)
  message(FATAL_ERROR "TICL470TOOLS_CODEC_ENGINE has not been set in config file, set to \"\" if not used")
endif()

include(FindPackageHandleStandardArgs)

# We need to capitalise the first letter of the release notes name for Linux to work
string(SUBSTRING ${TICL470TOOLS_BIOS_DIR} 0 1 _tmp_bios_name)
string(TOUPPER ${_tmp_bios_name} _tmp_bios_name)
string(LENGTH ${TICL470TOOLS_BIOS_DIR} _len)
math(EXPR _len ${_len}-1)
string(SUBSTRING ${TICL470TOOLS_BIOS_DIR} 1 ${_len} _tmp_bios_name_end)
set(_tmp_bios_name ${_tmp_bios_name}${_tmp_bios_name_end})

if(NOT ${TICL470TOOLS_BIOS_DIR} STREQUAL "")
	find_path(TICL470Tools_BIOS_PATH "${TICL470TOOLS_BIOS_DIR}_release_notes.html" 
    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_BIOS_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_BIOS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_BIOS_PATH)
	mark_as_advanced(TICL470Tools_BIOS_PATH)
	if(${TICL470Tools_BIOS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_BIOS_PATH} ${TICL470TOOLS_BIOS_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_XDC_DIR} STREQUAL "")
	find_path(TICL470Tools_XDC_PATH "${TICL470TOOLS_XDC_DIR}_release_notes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_XDC_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_XDC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_XDC_PATH)
	mark_as_advanced(TICL470Tools_XDC_PATH)
	if(${TICL470Tools_XDC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_XDC_PATH} ${TICL470TOOLS_XDC_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_XDAIS_DIR} STREQUAL "")
	find_path(TICL470Tools_XDAIS_PATH "${TICL470TOOLS_XDAIS_DIR}_eng_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_XDAIS_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_XDAIS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_XDAIS_PATH)
	mark_as_advanced(TICL470Tools_XDAIS_PATH)
	if(${TICL470Tools_XDAIS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_XDAIS_PATH} ${TICL470TOOLS_XDAIS_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_IPC_DIR} STREQUAL "")
	find_path(TICL470Tools_IPC_PATH "ipc_1_23_05_40_asl_release_notes.html"
				PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_IPC_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_IPC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_IPC_PATH)
	mark_as_advanced(TICL470Tools_IPC_PATH)
	if(${TICL470Tools_IPC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_IPC_PATH} ${TICL470TOOLS_IPC_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_PSP_DIR} STREQUAL "")
	find_path(TICL470Tools_PSP_PATH "Software-manifest.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_PSP_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_PSP_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_PSP_PATH)
	mark_as_advanced(TICL470Tools_PSP_PATH)
	if(${TICL470Tools_PSP_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_PSP_PATH} ${TICL470TOOLS_PSP_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_EDMA_DIR} STREQUAL "")
	find_path(TICL470Tools_EDMA_PATH "release_notes_edma3_lld_02_11_03.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_EDMA_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_EDMA_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_EDMA_PATH)
	mark_as_advanced(TICL470Tools_EDMA_PATH)
	if(${TICL470Tools_EDMA_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_EDMA_PATH} ${TICL470TOOLS_EDMA_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_FC_DIR} STREQUAL "")
	find_path(TICL470Tools_FC_PATH "framework_components_3_21_03_34_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_FC_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_FC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_FC_PATH)
	mark_as_advanced(TICL470Tools_FC_PATH)
	if(${TICL470Tools_FC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_FC_PATH} ${TICL470TOOLS_FC_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_IVAHD_JPEGVDEC_DIR} STREQUAL "")
	find_path(TICL470Tools_IVAHD_JPEGVDEC_PATH "readme_ivahd_jpegvdec_01_00_02_00_production.txt"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_IVAHD_JPEGVDEC_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_IVAHD_JPEGVDEC_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_IVAHD_JPEGVDEC_PATH)
	mark_as_advanced(TICL470Tools_IVAHD_JPEGVDEC_PATH)
	if(${TICL470Tools_IVAHD_JPEGVDEC_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_IVAHD_JPEGVDEC_PATH} ${TICL470TOOLS_IVAHD_JPEGVDEC_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_ISS_DIR} STREQUAL "")
	find_path(TICL470Tools_ISS_PATH "${TICL470TOOLS_ISS_DIR}_readme.txt"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_ISS_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_ISS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_ISS_PATH)
	mark_as_advanced(TICL470Tools_ISS_PATH)
	if(${TICL470Tools_ISS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_ISS_PATH} ${TICL470TOOLS_ISS_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_HDVPSS_DIR} STREQUAL "")
	find_path(TICL470Tools_HDVPSS_PATH "${TICL470TOOLS_HDVPSS_DIR}_readme.txt"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_HDVPSS_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_HDVPSS_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_HDVPSS_PATH)
	mark_as_advanced(TICL470Tools_HDVPSS_PATH)
	if(${TICL470Tools_HDVPSS_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_HDVPSS_PATH} ${TICL470TOOLS_HDVPSS_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_NDK_DIR} STREQUAL "")
	find_path(TICL470Tools_NDK_PATH "${TICL470TOOLS_NDK_DIR}_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_NDK_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_NDK_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_NDK_PATH)
	mark_as_advanced(TICL470Tools_NDK_PATH)
	if(${TICL470Tools_NDK_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_NDK_PATH} ${TICL470TOOLS_NDK_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_NSP_DIR} STREQUAL "")
	find_path(TICL470Tools_NSP_PATH "nsp_dm814x_01_00_00_08_ecu_asl_ReleaseNotes.html"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_NSP_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_NSP_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_NSP_PATH)
	mark_as_advanced(TICL470Tools_NSP_PATH)
	if(${TICL470Tools_NSP_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_NSP_PATH} ${TICL470TOOLS_NSP_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_AVB_DIR} STREQUAL "")
	find_path(TICL470Tools_AVB_PATH "AVB_Manifest.mht"
	    		PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_AVB_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_AVB_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_AVB_PATH)
	mark_as_advanced(TICL470Tools_AVB_PATH)
	if(${TICL470Tools_AVB_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_AVB_PATH} ${TICL470TOOLS_AVB_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_IVAHD_HDVICP20_DIR} STREQUAL "")
	find_path(TICL470Tools_IVAHD_HDVICP20_PATH "readme_hdvicp20.txt"
				PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_IVAHD_HDVICP20_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_IVAHD_HDVICP20_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_IVAHD_HDVICP20_PATH)
	mark_as_advanced(TICL470Tools_IVAHD_HDVICP20_PATH)
	if(${TICL470Tools_IVAHD_HDVICP20_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_IVAHD_HDVICP20_PATH} ${TICL470TOOLS_IVAHD_HDVICP20_DIR}")
	endif()
endif()

if(NOT ${TICL470TOOLS_CODEC_ENGINE_DIR} STREQUAL "")
	find_path(TICL470Tools_CODEC_ENGINE_PATH "${TICL470TOOLS_CODEC_ENGINE_DIR}_ReleaseNotes.html"
				PATHS
				"$ENV{CCS_PATH}/${TICL470TOOLS_CODEC_ENGINE_DIR}"
				"$ENV{CCS_PATH}/../${TICL470TOOLS_CODEC_ENGINE_DIR}"
				NO_DEFAULT_PATH
				NO_CMAKE_ENVIRONMENT_PATH
				NO_CMAKE_PATH
				NO_SYSTEM_ENVIRONMENT_PATH
				NO_CMAKE_SYSTEM_PATH
				NO_CMAKE_FIND_ROOT_PATH
	)
	find_package_handle_standard_args(TICL470Tools DEFAULT_MSG TICL470Tools_CODEC_ENGINE_PATH)
	mark_as_advanced(TICL470Tools_CODEC_ENGINE_PATH)
	if(${TICL470Tools_CODEC_ENGINE_PATH} MATCHES "NOTFOUND")
		message(FATAL_ERROR "${TICL470Tools_CODEC_ENGINE_PATH} ${TICL470TOOLS_CODEC_ENGINE_DIR}")
	endif()
endif()

