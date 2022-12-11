cmake_minimum_required (VERSION 3.0)
project (fractals)

set (CMAKE_CXX_STANDARD 14)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if(MSVC)
    add_definitions(/arch:AVX2)
endif()

find_package(OpenCV REQUIRED core highgui)
list(APPEND LINK_LIBS ${OpenCV_LIBS})

option(ENABLE_TBB "Enable multothreading using TBB." ON)
if(ENABLE_TBB)
    find_package(TBB)
	add_definitions(-DUSE_TBB)
    list(APPEND LINK_LIBS TBB::tbb)
endif()

add_executable(fractals fractals.cpp)
target_link_libraries(fractals ${LINK_LIBS})

