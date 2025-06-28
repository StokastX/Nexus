cmake_minimum_required(VERSION 3.22)
project("Nexus")

enable_language(CUDA)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "Assimp lib only")
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "Assimp lib only")
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "Assimp lib only")
set(ASSIMP_BUILD_DOCS OFF CACHE BOOL "Assimp lib only")
set(ASSIMP_INSTALL OFF CACHE BOOL "Assimp lib only")
add_subdirectory(vendor/assimp)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW lib only")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW lib only")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only")
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "GLFW lib only")
add_subdirectory(vendor/GLFW)

set(BUILD_UTILS OFF CACHE BOOL "GLEW lib only")
set(GLEW_INSTALL OFF CACHE BOOL "GLEW lib only")
add_subdirectory(vendor/GLEW/build/cmake)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
set(GLM_BUILD_TESTS OFF CACHE BOOL "GLM lib only")
add_subdirectory(vendor/glm)

add_subdirectory(vendor/imgui)

set(NEXUSBVH_BUILD_TESTS OFF CACHE BOOL "NexusBVH lib only")
add_subdirectory(vendor/NexusBVH)

file(GLOB_RECURSE src_files src/*.h src/*.cpp src/*.cuh src/*.cu)

set (include_files
	src
	vendor/assimp/include
	vendor/GLFW/include
	vendor/GLEW/include
	vendor/glm
	vendor/imgui
	vendor/imgui/backends
	vendor/stb
	vendor/NexusBVH/src
)

add_executable(Nexus ${src_files})

target_link_libraries(Nexus
	glfw
	glew
	assimp
	imgui
	NexusBVH
)

target_include_directories(Nexus
	PUBLIC
	${include_files}
)

# CUDA optimizations
target_compile_options(Nexus PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
    --use_fast_math -G -g -lineinfo -rdc=true
  >
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
    --use_fast_math -rdc=true
  >
)

# CUDA device link-time optimization (DLTO)
target_link_options(Nexus PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:>
  $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:
    --dlink-time-opt
  >
)

# Set Nexus as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Nexus)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src_files}
)

