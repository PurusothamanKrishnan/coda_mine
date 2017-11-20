# - Find Graphviz Dot utility.
# This module looks for Graphviz "dot" program.
#
# This module defines:
#
#  GRAPHVIZ_DOT_FOUND		- Set true if found.
#  Graphviz_Dot_EXECUTABLE	- points to dot.exe
#

include(FindPackageHandleStandardArgs)

find_program(Graphviz_Dot_EXECUTABLE NAMES dot dot.exe PATHS
				"C:/Program Files/ATT/Graphviz/bin"
				"D:/Program Files/ATT/Graphviz/bin"
			)

find_package_handle_standard_args(Graphviz_Dot DEFAULT_MSG Graphviz_Dot_EXECUTABLE)

mark_as_advanced(Graphviz_Dot_EXECUTABLE)
