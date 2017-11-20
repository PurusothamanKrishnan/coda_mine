# - Assembler information for TI C6000.
# Note that the C compiler is used rather than the assembler, and the C flags as well.
#
set(ASM_DIALECT "_C6X")
include(CMakeASMInformation)
set(CMAKE_ASM${ASM_DIALECT}_SOURCE_FILE_EXTENSIONS "s62")
set(CMAKE_ASM${ASM_DIALECT}_OUTPUT_EXTENSION ".obj" )
set(CMAKE_ASM${ASM_DIALECT}_COMPILE_OBJECT "custom_cl6x <CMAKE_ASM${ASM_DIALECT}_COMPILER> ${CMAKE_C_FLAGS} -OOO<OBJECT>  <SOURCE>")
set(ASM_DIALECT)