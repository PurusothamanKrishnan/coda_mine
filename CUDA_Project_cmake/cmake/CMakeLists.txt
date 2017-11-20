#
# \brief  Add a new project, managing the global variables as needed
# \author Cedric De Brito
# \date 27.10.15
#
function (addProject projectName)
  project(${projectName})
  MESSAGE("")
  MESSAGE(" Creating project ${PROJECT_NAME}")
  set("${PROJECT_NAME}_SRC" "" CACHE STRING "Absolute paths of all files in project" FORCE)
  set("${PROJECT_NAME}_EXT_SRC" "" CACHE STRING "Absolute paths of all files from other projects" FORCE)
  set("${PROJECT_NAME}_DIRS" "" CACHE STRING "Absolute paths of all directories in project" FORCE)
  set("${PROJECT_NAME}_OCVLIBS" "" CACHE STRING "OpenCv libraries needed" FORCE)
  set("${PROJECT_NAME}_LEVMARLIBS" "" CACHE STRING "Levmar libraries needed" FORCE)
  set("${PROJECT_NAME}_ADTFSTREAMINGLIBS" "" CACHE STRING "Adtf Streaming libraries needed" FORCE)  
  set("${PROJECT_NAME}_MAIN" "" CACHE STRING "Main project's entry point" FORCE)  
  set("${PROJECT_NAME}_ADTF_MAIN" "" CACHE STRING "Main file for Adtf library" FORCE)  
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set("${PROJECT_NAME}_UNIT_TESTS" "" CACHE STRING "Unit test needed" FORCE)
  add_subdirectory(${PROJECT_NAME})
endfunction ()


#
# \brief  Do all the final touches to create the project correctly
# \author Cedric De Brito
# \date 27.10.15
#
function (createProject)
MESSAGE("Extra args: ${ARGN}")
  if ( ("${ARGN}" STREQUAL "--force")
   OR (NOT "${TESTING_MODE}" STREQUAL "only"))
    add_executable(${PROJECT_NAME} 
    ${ALL_HEADERS} 
    ${${PROJECT_NAME}_MAIN} 
    ${${PROJECT_NAME}_SRC}
    ${${PROJECT_NAME}_EXT_SRC}
	
    )
	
    if ("${${PROJECT_NAME}_UNIT_TESTS}" STREQUAL "enable")
	    MESSAGE(" adding library for unit test")
        add_library("${PROJECT_NAME}_library" 
          ${ALL_HEADERS} 
          ${${PROJECT_NAME}_SRC}
          ${${PROJECT_NAME}_EXT_SRC}
        )
        addUnitTest()
    endif()	
	
    foreach(ocvLib ${${PROJECT_NAME}_OCVLIBS})
      MESSAGE("Linking to opencv lib ${ocvLib}")
      linkOpenCvLib("${ocvLib}")
    endforeach()
		
    MESSAGE("Setting up post-build event to copy config")
    add_custom_command(TARGET ${PROJECT_NAME}
               PRE_BUILD
               COMMAND "${CMAKE_SOURCE_DIR}/../tools/astyle/AStyle.exe" --options="${CMAKE_SOURCE_DIR}/../tools/astyle/AStyleExtension-2017-3-29.ini" "${CMAKE_SOURCE_DIR}/*.cpp" "${CMAKE_SOURCE_DIR}/*.h" "${CMAKE_SOURCE_DIR}/*.hpp" "${CMAKE_SOURCE_DIR}/*.hxx"
               COMMENT "Auto-formatting with astyle")
    add_custom_command(TARGET ${PROJECT_NAME}
               POST_BUILD
               COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/config" "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/config"
               COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/common/config" "${CMAKE_BINARY_DIR}/common/config"
               COMMENT "Copying config to build")
    MESSAGE("Copying ffmpeg to the bin directory")		# To read videos
	set(FFMPEG_PATH "${OPENCV_RELEASE}/bin/opencv_ffmpeg300.dll")
	if (EXISTS ${FFMPEG_PATH})
      file(COPY "${FFMPEG_PATH}"
            DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug )
      file(COPY "${FFMPEG_PATH}"
            DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release )
	endif()
	set(FFMPEG_PATH "${OPENCV_RELEASE}/bin/opencv_ffmpeg300_64.dll")
	if (EXISTS ${FFMPEG_PATH})
      file(COPY "${FFMPEG_PATH}"
            DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug )
      file(COPY "${FFMPEG_PATH}"
            DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release )
	endif()
  endif()
endfunction ()


#
# \brief  Do all the final touches to create the project correctly
# \author Cedric De Brito
# \date 27.10.15
#
function (createProjectCUDA)
MESSAGE("Extra args: ${ARGN}")
  if ( ("${ARGN}" STREQUAL "--force")
   OR (NOT "${TESTING_MODE}" STREQUAL "only"))
    CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
    ${${PROJECT_NAME}_MAIN} 
    ${${PROJECT_NAME}_SRC}
	
    )
	
    foreach(ocvLib ${${PROJECT_NAME}_OCVLIBS})
      MESSAGE("Linking to opencv lib ${ocvLib}")
      linkOpenCvLib("${ocvLib}")
    endforeach()
  message("${CUDA_LIB_PATH}")
  #set(CUDA_PATH_DEV_LIB ${CUDA_LIB_PATH}\Win32\cudadevrt.lib)
  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS_PATH_DEV})
  endif()
endfunction ()
#
# \brief  Add all the files from another project
# \author Cedric De Brito
# \date 27.10.15
#
function (needsProject projectName)
  MESSAGE("")
  MESSAGE("Adding all files of project ${projectName} to current project")
  set("${PROJECT_NAME}_EXT_SRC" "${${projectName}_SRC};${${PROJECT_NAME}_EXT_SRC}" CACHE STRING "Add files of another project" FORCE)
  include_directories("${${projectName}_DIRS}")
  source_group("${projectName}" FILES ${${projectName}_SRC})
endfunction ()


#
# \brief  Add the main file
# \author Cedric De Brito
# \date 27.10.15
#
function (addMain)
  MESSAGE(" Adding main file for project ${PROJECT_NAME}")
  # We also want to allow the extraordinary use of several files passed as arguments
  foreach(filename ${ARGN})
    set(fileList ${fileList} ${filename})
  endforeach()
  set("${PROJECT_NAME}_MAIN" "${fileList}" CACHE STRING "Add project's entry point" FORCE)
  _addGroup("test" "${fileList}")
endfunction ()



#
# \brief  Add a directory to a project and manage the includes and groups
# \author Cedric De Brito
# \date 7.10.15
#
function (addSubDir path)
  set(TEMP_COUNTER "0" CACHE STRING "reset counter for child directories" FORCE)
  MESSAGE("")
  MESSAGE("Including directory '${CMAKE_CURRENT_SOURCE_DIR}/${path}' ")
  # Add to the list for later use
  set(folderPath "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
  set("${PROJECT_NAME}_DIRS" "${${PROJECT_NAME}_DIRS};${folderPath}" CACHE STRING "add files to source" FORCE)
  # Allow includescd
  include_directories("${folderPath}")
  add_subdirectory("${path}")
  if (${TEMP_COUNTER} GREATER 0)
    foreach(counter RANGE 1 ${TEMP_COUNTER})
      set(groupVarName "LOCAL_GROUP${counter}")
      set(fileListVarName "LOCAL_FILELIST${counter}")
      source_group("${${groupVarName}}" FILES ${${fileListVarName}})
      MESSAGE("(${counter}) Adding group '${${groupVarName}}' with FILES '${${fileListVarName}}' ")
    endforeach()
  endif ()
endfunction ()


#
# \brief  Add a directory of adtf files to a project and manage the includes and groups
# \author Cedric De Brito
# \date 7.10.15
#
function (addSubDirAdtf path)
  # Special adtf directories should not even be added if we are not compiling for ADTF
  if (USE_ADTF)
    addSubDir (${path})
  endif (USE_ADTF)
endfunction ()



#
# \brief  Add a set of files to the project and to a specific "folder" in VisualStudio
#     For user-friendliness, the function allows the file list to be in one or many arguments
#     All arguments after the first one are interpreted as file names
# \author Cedric De Brito
# \date 7.10.15
#
function (addGroup groupName)
  foreach(filename ${ARGN})
    set(fileList ${fileList} ${filename})
  endforeach()
  #_addGroup("${groupName}" "${fileList}")
  # Add files to PROJECT_SRC for later use
  set("${PROJECT_NAME}_SRC" "${${PROJECT_NAME}_SRC};${fileList}" CACHE STRING "add files to source" FORCE)
endfunction ()

function (_addGroup groupName fileList)
  # Update temp counter
  set(counter "${TEMP_COUNTER}")
  message("${TEMP_COUNTER}")
  math(EXPR counter "${counter}+1")
  set(TEMP_COUNTER "${counter}" CACHE STRING "increment counter" FORCE)
  # Create new variable names
  set(groupVarName "LOCAL_GROUP${counter}")
  set(fileListVarName "LOCAL_FILELIST${counter}")
  # Save name of group and list of files
  message(${${groupVarName}})
  set(${groupVarName} "${groupName}" CACHE STRING "set group name" FORCE)
  set(${fileListVarName} "${fileList}" CACHE STRING "set file list for local group" FORCE)
endfunction ()


#
# \brief  Link to an openCV library, in debug and release mode
#     Also copies the correct dll to the output binary folder
# \param dllName the name of the library, *without path or extension*
# \author Cedric De Brito
# \date 7.10.15
#
function (addOpenCvLib dllName)
  set("${PROJECT_NAME}_OCVLIBS" "${dllName};${${PROJECT_NAME}_OCVLIBS}" CACHE STRING "Update the list of needed OpenCv libraries" FORCE)
  message(${PROJECT_NAME}_OCVLIBS)
endfunction()

#
# \brief  Link to an openCV library, in debug and release mode
#     Also copies the correct dll to the output binary folder (must be done after creating the executable)
# \param dllName the name of the library, *without path or extension*
# \param opt Optional name ending for target name
# \author Cedric De Brito
# \date 7.10.15
#
function (linkOpenCvLib dllName)
	target_link_libraries(${PROJECT_NAME}${ARGN}
	   debug "${OPENCV_LIB}/${dllName}d.lib"
	   optimized "${OPENCV_LIB}/${dllName}.lib"
	)

    if ("${${PROJECT_NAME}_UNIT_TESTS}" STREQUAL "enable")
		target_link_libraries("${PROJECT_NAME}_library"${ARGN}
		   debug "${OPENCV_LIB}/${dllName}d.lib"
		   optimized "${OPENCV_LIB}/${dllName}.lib"
		)
    endif()
	
	file(	COPY "${OPENCV_BIN}/${dllName}d.dll"
			DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug )
	file(	COPY "${OPENCV_BIN}/${dllName}.dll"
			DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release )
endfunction()

#
# \brief  Look for the TNT and JAMA folders and if found, add to include_directories
# \author Cedric De Brito
# \date 17.2.16
#
macro (lookforLapack)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../lib_packages/tnt_126")
      include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../lib_packages/tnt_126")
    endif()
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../lib_packages/jama125")
      include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../lib_packages/jama125")
    endif()
endmacro()


#
# \brief  Look for a googletest folder and if found, sets the correct variables
# \author Cedric De Brito
# \date 9.2.16
#
macro (lookforGTest)
  if ("${TESTING_MODE}" STREQUAL "exclude")
    set(GTEST false)
  else()
    if (EXISTS "../../googletest/googletest")
      set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../googletest/googletest")
      set(GTEST true)
    elseif (EXISTS "../../googletest")
      set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../googletest")
      set(GTEST true)
    endif()

    if (${GTEST})
      message(" + Google Test folder found")
      set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/include")
      include_directories("${GTEST_INCLUDE_DIR}")
      
      set(FOLDER "${GTEST_ROOT}/msvc/gtest-md/Debug")
      if (EXISTS "${FOLDER}/gtestd.lib")
      set(GTEST_LIBRARY "${FOLDER}/gtestd.lib")
      elseif()
      message(" - Failed to find Google Test compiled debug library ['${FOLDER}/gtestd.lib' does not exist]")
      set(GTEST false)
      endif()
      
      if (EXISTS "${FOLDER}/gtest_main-mdd.lib")
      set(GTEST_MAIN_LIBRARY "${FOLDER}/gtest_main-mdd.lib")
      elseif()
      message(" - Failed to find Google Test compiled debug library ['${FOLDER}/gtest_main-mdd.lib' does not exist]")
      set(GTEST false)
      endif()
    endif()

    if (${GTEST})
      message(" + Google Test libraries found [using files from ${FOLDER}]")
    else()
      message(" - Not using Google Test. Ignoring all tests with this dependency")
      set(GTEST false)
    endif()
    message("")
  endif()
  if (${UNIT_TESTS_PER_MODULE})
    add_subdirectory(../lib_packages/googletest ${CMAKE_BINARY_DIR}/lib)
	enable_testing()
  endif()
endmacro()

