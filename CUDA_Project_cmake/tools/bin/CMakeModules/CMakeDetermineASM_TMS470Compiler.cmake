# - Assembler discovery for TI TMS470 (ARM).
# Note that the C compiler is used rather than the assembler.
#
set(ASM_DIALECT "_TMS470")
set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT "cl470")	# Not "asm470"
include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
