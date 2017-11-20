# - Assembler information for TI TMS470 (ARM).
# Note that the C compiler is used rather than the assembler, and the C flags as well.
#
set(ASM_DIALECT "_TMS470")
include(CMakeASMInformation)
set(CMAKE_ASM${ASM_DIALECT}_SOURCE_FILE_EXTENSIONS "asm")
set(CMAKE_ASM${ASM_DIALECT}_OUTPUT_EXTENSION ".obj" )
set(CMAKE_ASM${ASM_DIALECT}_COMPILE_OBJECT "<CMAKE_ASM${ASM_DIALECT}_COMPILER> ${CMAKE_C_FLAGS} -fr<OBJECT_DIR> -fs<OBJECT_DIR> -ff${CMAKE_BINARY_DIR}/lst <SOURCE>")
set(ASM_DIALECT)
