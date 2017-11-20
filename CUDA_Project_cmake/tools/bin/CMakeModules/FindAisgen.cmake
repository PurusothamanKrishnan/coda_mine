# - Find AISgen utility.
# This module looks for TI "AISgen" program.
#
# This module defines:
#
#  AISGEN_FOUND			- Set true if found.
#  Aisgen_EXECUTABLE	- points to AISgen.exe
#

include(FindPackageHandleStandardArgs)

find_program(Aisgen_EXECUTABLE NAMES AISgen_d800k006 AISgen_d800k006.exe PATHS
				"C:/Program Files/Texas Instruments/AISgen for D800K006"
				"D:/Program Files/Texas Instruments/AISgen for D800K006"
				"E:/Program Files/Texas Instruments/AISgen for D800K006"
			)

find_package_handle_standard_args(Aisgen DEFAULT_MSG Aisgen_EXECUTABLE)

mark_as_advanced(Aisgen_EXECUTABLE)
