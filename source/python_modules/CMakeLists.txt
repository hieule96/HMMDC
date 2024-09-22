cmake_minimum_required(VERSION 3.10)

set(CMAKE_VERBOSE_MAKEFILE ON)
project(pybindCabac)


## TODO: Set up code coverage
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake)

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage ${TEST_TARGET} coverage)
    SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
endif()


set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")
#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)
find_package( Threads REQUIRED )
add_subdirectory(thirdparty/pybind11)
add_subdirectory(DCT_HEVC)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../../../)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../../)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../)
