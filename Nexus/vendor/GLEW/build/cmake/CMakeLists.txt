if ( NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Build type" )
endif ()

project (glew)

cmake_minimum_required (VERSION 3.24)

include(GNUInstallDirs)

if (COMMAND cmake_policy)
  cmake_policy (SET CMP0003 NEW)
  cmake_policy (SET CMP0042 NEW)
endif()

set(CMAKE_DEBUG_POSTFIX d)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option (BUILD_UTILS "utilities" ON)
option (GLEW_REGAL "Regal mode" OFF)
option (GLEW_OSMESA "OSMesa mode" OFF)
option (GLEW_INSTALL "Install mode" OFF)

if (APPLE)
    option (BUILD_FRAMEWORK "Build Framework bundle for OSX" OFF)
endif ()

set (GLEW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# get version from config/version
file (STRINGS ${GLEW_DIR}/config/version  _VERSION_MAJOR_STRING REGEX "GLEW_MAJOR[ ]*=[ ]*[0-9]+.*")
string (REGEX REPLACE "GLEW_MAJOR[ ]*=[ ]*([0-9]+)" "\\1" CPACK_PACKAGE_VERSION_MAJOR ${_VERSION_MAJOR_STRING})
file (STRINGS ${GLEW_DIR}/config/version  _VERSION_MINOR_STRING REGEX "GLEW_MINOR[ ]*=[ ]*[0-9]+.*")
string (REGEX REPLACE "GLEW_MINOR[ ]*=[ ]*([0-9]+)" "\\1" CPACK_PACKAGE_VERSION_MINOR ${_VERSION_MINOR_STRING})
file (STRINGS ${GLEW_DIR}/config/version  _VERSION_PATCH_STRING REGEX "GLEW_MICRO[ ]*=[ ]*[0-9]+.*")
string (REGEX REPLACE "GLEW_MICRO[ ]*=[ ]*([0-9]+)" "\\1" CPACK_PACKAGE_VERSION_PATCH ${_VERSION_PATCH_STRING})
set (GLEW_VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package (OpenGL REQUIRED)

# X11 required when builing visualinfo and glewinfo,
# but not for Windows or Apple OSX platforms
if (BUILD_UTILS AND NOT WIN32 AND NOT APPLE)
  find_package (X11)
endif()

if (WIN32)
  set (GLEW_LIB_NAME glew32)
else ()
  set (GLEW_LIB_NAME GLEW)
  set (DLL_PREFIX lib)
endif ()

set (GLEW_LIBRARIES ${OPENGL_LIBRARIES})

add_definitions (-DGLEW_NO_GLU)

#### Regal mode ####

if (GLEW_REGAL)
  if (WIN32)
    set (REGAL_LIB_NAME regal32)
  else ()
    set (REGAL_LIB_NAME Regal)
  endif ()
  add_definitions (-DGLEW_REGAL)
  set (GLEW_LIBRARIES ${REGAL_LIB_NAME})
endif ()

#### OSMesa mode ####

if (GLEW_OSMESA)
  if (WIN32)
    set (OSMESA_LIB_NAME osmesa)
  else ()
    set (OSMESA_LIB_NAME OSMesa)
  endif ()
  add_definitions (-DGLEW_OSMESA)
  set (GLEW_LIBRARIES ${OSMESA_LIB_NAME} ${OPENGL_LIBRARIES})
  set (X11_LIBRARIES)
endif ()

#### GLEW ####

include_directories (${GLEW_DIR}/include)

set (GLEW_PUBLIC_HEADERS_FILES ${GLEW_DIR}/include/GL/wglew.h ${GLEW_DIR}/include/GL/glew.h ${GLEW_DIR}/include/GL/glxew.h)
set (GLEW_SRC_FILES ${GLEW_DIR}/src/glew.c)

if (WIN32)
  list (APPEND GLEW_SRC_FILES ${GLEW_DIR}/build/glew.rc)
endif ()

add_library (glew SHARED ${GLEW_PUBLIC_HEADERS_FILES} ${GLEW_SRC_FILES})
set_target_properties (glew PROPERTIES COMPILE_DEFINITIONS "GLEW_BUILD" OUTPUT_NAME "${GLEW_LIB_NAME}" PREFIX "${DLL_PREFIX}"
                                       VERSION ${GLEW_VERSION}
                                       SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR})
set_target_properties(glew PROPERTIES FOLDER "GLEW")
add_library (glew_s STATIC ${GLEW_PUBLIC_HEADERS_FILES} ${GLEW_SRC_FILES})
set_target_properties (glew_s PROPERTIES COMPILE_DEFINITIONS "GLEW_STATIC" OUTPUT_NAME "${GLEW_LIB_NAME}" PREFIX lib)
set_target_properties(glew_s PROPERTIES FOLDER "GLEW")

if (MSVC)
  # add options from visual studio project
  target_compile_definitions (glew PRIVATE "GLEW_BUILD;VC_EXTRALEAN")
  target_compile_definitions (glew_s PRIVATE "GLEW_STATIC;VC_EXTRALEAN")
  target_link_libraries (glew LINK_PRIVATE -BASE:0x62AA0000)
  # kill security checks which are dependent on stdlib
  target_compile_options (glew PRIVATE -GS-)
  target_compile_options (glew_s PRIVATE -GS-)
  # remove stdlib dependency
  target_link_libraries (glew LINK_PRIVATE -nodefaultlib -noentry)
  string(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
elseif (WIN32 AND ((CMAKE_C_COMPILER_ID MATCHES "GNU") OR (CMAKE_C_COMPILER_ID MATCHES "Clang")))
  # remove stdlib dependency on windows with GCC and Clang (for similar reasons
  # as to MSVC - to allow it to be used with any Windows compiler)
  target_compile_options (glew PRIVATE -fno-builtin -fno-stack-protector)
  target_compile_options (glew_s PRIVATE -fno-builtin -fno-stack-protector)
  target_link_libraries (glew LINK_PRIVATE -nostdlib)
endif ()

if (BUILD_FRAMEWORK)
  set_target_properties(glew PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION ${GLEW_VERSION}
    MACOSX_FRAMEWORK_IDENTIFIER net.sourceforge.glew
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${GLEW_VERSION}
    MACOSX_FRAMEWORK_BUNDLE_VERSION ${GLEW_VERSION}
    XCODE_ATTRIBUTE_INSTALL_PATH "@rpath"
    PUBLIC_HEADER "${GLEW_PUBLIC_HEADERS_FILES}"
    OUTPUT_NAME GLEW
  )
endif()

target_link_libraries (glew LINK_PUBLIC ${GLEW_LIBRARIES})
target_link_libraries (glew_s ${GLEW_LIBRARIES})

if(CMAKE_VERSION VERSION_LESS 2.8.12)
  set(MAYBE_EXPORT "")
else()
  target_compile_definitions(glew_s INTERFACE "GLEW_STATIC")
  foreach(t glew glew_s)
    target_include_directories(${t} PUBLIC $<INSTALL_INTERFACE:include>)
  endforeach()
  set(MAYBE_EXPORT EXPORT glew-targets)
endif()

set(targets_to_install "")
if(NOT DEFINED BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
  list(APPEND targets_to_install glew)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS OR NOT BUILD_SHARED_LIBS)
  list(APPEND targets_to_install glew_s)
endif()

install ( TARGETS ${targets_to_install}
          ${MAYBE_EXPORT}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}
)

if (BUILD_UTILS)
  set (GLEWINFO_SRC_FILES ${GLEW_DIR}/src/glewinfo.c)
  if (WIN32)
    list (APPEND GLEWINFO_SRC_FILES ${GLEW_DIR}/build/glewinfo.rc)
  endif ()
  add_executable (glewinfo ${GLEWINFO_SRC_FILES})
  target_link_libraries (glewinfo glew)
  if (NOT WIN32)
    target_link_libraries(glewinfo ${X11_LIBRARIES})
  endif ()

  set (VISUALINFO_SRC_FILES ${GLEW_DIR}/src/visualinfo.c)
  if (WIN32)
    list (APPEND VISUALINFO_SRC_FILES ${GLEW_DIR}/build/visualinfo.rc)
  endif ()
  add_executable (visualinfo ${VISUALINFO_SRC_FILES})
  target_link_libraries (visualinfo glew)
  if (NOT WIN32)
    target_link_libraries(visualinfo ${X11_LIBRARIES})
  endif ()

  install ( TARGETS glewinfo visualinfo
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()

set (prefix ${CMAKE_INSTALL_PREFIX})
set (exec_prefix ${CMAKE_INSTALL_PREFIX})
set (libdir ${CMAKE_INSTALL_FULL_LIBDIR})
set (includedir ${CMAKE_INSTALL_FULL_INCLUDEDIR})
set (version ${GLEW_VERSION})
set (libname ${GLEW_LIB_NAME})
set (cflags)
set (requireslib glu)

#  Mac OSX has no glu.pc unless optional X11/GLX is installed
if (APPLE)
  set (requireslib)
endif ()

configure_file (${GLEW_DIR}/glew.pc.in ${GLEW_DIR}/glew.pc @ONLY)

install(FILES ${GLEW_DIR}/glew.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

if(WIN32 AND MSVC AND (NOT MSVC_VERSION LESS 1600) AND (NOT CMAKE_VERSION VERSION_LESS "3.1"))
    install(
        FILES $<TARGET_PDB_FILE:glew>
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        CONFIGURATIONS Debug RelWithDebInfo
    )
endif()

install (FILES
    ${GLEW_DIR}/include/GL/wglew.h
    ${GLEW_DIR}/include/GL/glew.h
    ${GLEW_DIR}/include/GL/glxew.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GL)

if(MAYBE_EXPORT)
  install(EXPORT glew-targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glew
    NAMESPACE GLEW::)
  install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/glew-config.cmake
      ${CMAKE_CURRENT_SOURCE_DIR}/CopyImportedTargetProperties.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/glew)
endif()

if(GLEW_INSTALL AND NOT TARGET uninstall)
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
      IMMEDIATE @ONLY)

  add_custom_target(uninstall
      COMMAND ${CMAKE_COMMAND} -P
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
