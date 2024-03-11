cmake_minimum_required(VERSION 3.12...3.18)

project(GraphKokkos LANGUAGES CXX)

if(UNIX)
    set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -fopenmp")
endif()

find_package(Kokkos REQUIRED)

set(SOURCE_EXE src/main.cpp)
add_library(grKokkosLib STATIC
            include/matrix.h
            include/matrix_la.h
            include/matrix_utils.h
            include/betweenness_centrality.h
            include/mmio.h
            src/mmio.cpp)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
    link_libraries(-g)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
    link_libraries(-O3)
endif()

target_include_directories(grKokkosLib PUBLIC include ${Kokkos_ROOT}/include)

add_executable(grKokkosExe ${SOURCE_EXE})

target_link_libraries(grKokkosExe grKokkosLib Kokkos::kokkos)

message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
