function (addProject projectName)
	project(${projectName})
	message("Creating Project")
	message("${projectName}")
	set("${projectName}_SRCS" "" CACHE STRING "Source files for the project" FORCE)
	set("${projectName}_DIRS" "" CACHE STRING "Project folders for the project" FORCE)
	set("${projectName}_EXT_SRC" "" CACHE STRING "adding files from External sources" FORCE)
	set("${projectName}_DEPENDENT_PROJECT" "" CACHE STRING "adding files from External sources" FORCE)
	set("${projectName}_MAINFILES" "" CACHE STRING "main source files for the project" FORCE)
	set("${projectName}_CUDALIBS" "" CACHE STRING "CUDA library files for the project" FORCE)
	set("${projectName}_CUDASRCS" "" CACHE STRING "CUDA source files for the project" FORCE)
	set("${projectName}_OCVLIBS" "" CACHE STRING "Setting Source macro for the project" FORCE)
	set(TMP_COUNTER "0")
	add_subdirectory(${projectName})
endfunction()


function(addSources)
	message(" Adding source files")
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
		message("${filelist}")
	endforeach()
	set("${projectName}_SRCS"    "${${projectName}_SRCS};${filelist}" CACHE STRING "Setting list of files for the project" FORCE)
endfunction()

function(addSrcGroups groupName)
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
	endforeach()
	set("${projectName}_SRCS"    "${${projectName}_SRCS};${filelist}" CACHE STRING "Setting list of files for the project" FORCE)
	addGroup("${groupName}" "${filelist}")
endfunction()

function (addSubDirectory path)
	set(TMP_COUNTER "0" CACHE STRING "Reset counter for chile directory" FORCE)
	message("")
	message("Adding Path to the project")
	set(currentAbsolutePath "${CMAKE_CURRENT_SOURCE_DIR}/${path}/source")
	include_directories("${currentAbsolutePath}")
	set("${projectName}_DIRS"    "${${projectName}_DIRS};${currentAbsolutePath}" CACHE STRING "Setting list of CUDA libraries for the project" FORCE)
	add_subdirectory("${path}")
	if (${TMP_COUNTER} GREATER 0) 
		foreach(counter RANGE 1 ${TMP_COUNTER})
			set(groupVarName "TMP_GROUP_${counter}")
			set(filesVarName "TMP_FILELIST_${counter}")
			source_group("${${groupVarName}}" FILES "${${filesVarName}}")
			message(" (${counter}) Adding group '${${groupVarName}}'  with FILES '${${filesVarName}}' ")
		endforeach()
	endif()
endfunction()

function (addTest testName)
	message(" test mode is not set")
endfunction()
function(addGroup groupname filelist)
	set(counter "${TMP_COUNTER}")
	math(EXPR counter "${counter}+1")

	set(TMP_COUNTER ${counter} CACHE STRING "Setting the value to tmp counter" FORCE)
 	set(groupVarName "TMP_GROUP_${counter}")
	set(filelistVarName "TMP_FILELIST_${counter}")
	set(${groupVarName} "${groupname}" CACHE STRING " adding files to the local group" FORCE)
	set(${filelistVarName} "${filelist}" CACHE STRING " adding files to the local group" FORCE)
endfunction()

function(addMainSources)
	message(" Adding source files")
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
		message("${filelist}")
	endforeach()
	set("${projectName}_MAINFILES"    "${${projectName}_SRCS};${filelist}" CACHE STRING "Setting list of files for the project" FORCE)
endfunction()

function(addExternalProject projectnamevar)
	message(" adding source files of the ${projectnamevar} into ${projectName} project")
	set("${projectName}_EXT_SRC"    "${${projectName}_EXT_SRC};${${projectnamevar}_SRCS}" CACHE STRING "Setting list of files for the project" FORCE)
	set("${projectName}_DEPENDENT_PROJECT"    "${${projectName}_DEPENDENT_PROJECT};${projectnamevar}" CACHE STRING " list of dependency project names" FORCE)
	include_directories("${${projectnamevar}_DIRS}")
	
	message("${${projectnamevar}_DIRS}")

	source_group("${projectnamevar}" FILES ${${projectnamevar}_SRCS})
endfunction()




function(addCUDASources)
	message(" Adding CUDA source files")
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
	endforeach()
	set("${projectName}_CUDASRCS"    "${${projectName}_CUDASRCS};${filelist}" CACHE STRING "Setting list of files for the project" FORCE)
endfunction()

function(addMainFile)
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
	endforeach()
	set("${projectName}_MAINFILES"    "${${projectName}_MAINFILES};${filelist}" CACHE STRING "Setting list of files for the project" FORCE)
endfunction()


function(addOpenCVLIbraries)
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
	endforeach()
	set("${projectName}_OCVLIBS"    "${${projectName}_OCVLIBS};${filelist}" CACHE STRING "Setting list of openCV libraries for the project" FORCE)
endfunction()


function(addCUDALibs)
	set(listVar "")
	foreach( srcfile ${ARGN})
		LIST(APPEND listVar ${srcfile})
		set(filelist ${filelist} ${srcfile})
	endforeach()
	set("${projectName}_CUDALIBS"    "${${projectName}_CUDALIBS};${filelist}" CACHE STRING "Setting list of CUDA libraries for the project" FORCE)
endfunction()



function (linkOpenCVLibrary libname)
	message("Linking openCv library : ${libname}")
	target_link_libraries(${projectName} ${ARGN}
	debug "${OPENCV_DEBUG}/${libname}.so"
	optimized "${OPENCV_RELEASE}/${libname}.so")
	
	#file(COPY "${OPENCV_BIN}/${libname}d.dll"
	#DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
	
	#file(COPY "${OPENCV_BIN}/${libname}.dll"
	#DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
endfunction()
	
function (linkCUDALibs libName)
	message("${CUDA_LIBRARY}/${libName}.so")
	target_link_libraries(${projectName} "${CUDA_LIBRARY}/${libName}.so")
	
endfunction()


function (includePath pathVar) 
	include_directories("${pathVar}")
endfunction()

function(createProject)
	message("---------------------------------------------------------------------")
	message("${${projectName}_SRCS}")	
	#message("${${projectName}_SRCS}")
	message("${ALL_HEADERS}")
	message("---------------------------------------------------------------------")
	message("${${projectName}_MAINFILES}")
	message("********************************************************************")
	add_executable(${projectName}
	${ALL_HEADERS}
	${${projectName}_SRCS}	
	${${projectName}_CUDASRCS}
	${${projectName}_EXT_SRC}
	${${projectName}_MAINFILES} )
					
	foreach(libVal ${${projectName}_OCVLIBS})
		linkOpenCVLibrary("${libVal}")
	endforeach()
	#link threadLibrary
        target_link_libraries(${projectName} pthread)
endfunction()


function(createProjectCUDA)
	CUDA_ADD_EXECUTABLE(${projectName}
	${ALL_HEADERS}
	${${projectName}_SRCS}	
	${${projectName}_CUDASRCS}
	${${projectName}_EXT_SRC}
	${${projectName}_MAINFILES} )
					
	foreach(libVal ${${projectName}_OCVLIBS})
		linkOpenCVLibrary("${libVal}")
	endforeach()
	
	foreach(libCUDA ${${projectName}_CUDALIBS})
		message("${CUDA_LIBRARY}")
		linkCUDALibs("${libCUDA}")
	endforeach()
endfunction()

