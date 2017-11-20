# - Find UnderstandC command line tool
# This module looks for Scientific Toolworks' UnderstandC (see www.scitools.com).
#
# This module defines:
#
#  UNDERSTAND_FOUND			 - Set true if found.
#  Understand_EXECUTABLE	 - Points to und/und.exe.
#

include(FindPackageHandleStandardArgs)

find_program(Understand_EXECUTABLE NAMES und und.exe)

find_package_handle_standard_args(Understand DEFAULT_MSG Understand_EXECUTABLE)

mark_as_advanced(Understand_EXECUTABLE)
