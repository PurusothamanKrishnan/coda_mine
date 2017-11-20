# - Assembler discovery for TI C6000.
# Note that the C compiler is used rather than the assembler.
#
set(ASM_DIALECT "_C6X")
set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT "cl6x")	# Not "asm6x"
include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
