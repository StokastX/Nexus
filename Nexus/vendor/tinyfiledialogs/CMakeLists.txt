project(tinyfiledialogs LANGUAGES C CXX)

file(GLOB src
	tinyfiledialogs.c
	tinyfiledialogs.h
)

add_library(tinyfiledialogs STATIC ${src})

target_include_directories(tinyfiledialogs
    PUBLIC
    .
)

