cmake_minimum_required(VERSION 2.8)
project(cuda_preprocessor)
# set(CMAKE_CXX_STANDARD 11)
set(BUILD_SHARED_LIBS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++11")


find_package(Boost 1.71 REQUIRED)
# FIND_PACKAGE(Boost 1.71 COMPONENTS python)
if(Boost_FOUND)
    message(" * BOOST ${BOOST_VERSION} was found")
endif()


find_package(CUDA)
if (CUDA_FOUND)
    message(" * CUDA ${CUDA_VERSION} was found")

    include(FindCUDA)
    include_directories(${CUDA_INCLUDE_DIRS}
        )

    set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

    if (NOT DEFINED CUDA_CAPABILITY_VERSION_CHECKER)
        set(CUDA_CAPABILITY_VERSION_CHECKER
            "${CATKIN_DEVEL_PREFIX}/lib/capability_version_checker")
    endif ()

    execute_process(COMMAND ${CUDA_CAPABILITY_VERSION_CHECKER}
        OUTPUT_VARIABLE CUDA_CAPABILITY_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if ("${CUDA_CAPABILITY_VERSION}" MATCHES "^[1-9][0-9]+$")
        set(CUDA_ARCH "sm_${CUDA_CAPABILITY_VERSION}")
    else ()
        set(CUDA_ARCH "sm_52")
    endif ()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CUDA_ARCH};-std=c++11;--ptxas-options=-v)

    set(SUBSYS_DESC "Point cloud ndt gpu library")

else ()
    message("CUDA was not found.")
endif ()

file(GLOB SOURCES "*.cc" "*.cu")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ./
)


link_libraries(cuda_preprocessor 
               boost_python-py35
               boost_numpy35
               ${Boost_LIBRARIES})
CUDA_ADD_LIBRARY(cuda_preprocessor  ${SOURCES})
