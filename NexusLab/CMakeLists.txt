cmake_minimum_required(VERSION 3.24)
project(NexusLab LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB_RECURSE src_files src/*.h src/*.cpp src/*.cuh src/*.cu)

set (include_files
	src
)

add_executable(NexusLab ${src_files})

target_link_libraries(NexusLab
	Nexus
)

target_include_directories(NexusLab
	PUBLIC
	${include_files}
)

# Set Nexus as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NexusLab)

# Match VS filters to directory structure on disk
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src_files}
)

