cmake_minimum_required(VERSION 3.10.2)

project(cpp_exports)


# Setup global options.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/info)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wall) 
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Werror -Og -ggdb3 -fno-inline)               
set(CMAKE_BUILD_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/info)


# set (CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++1z)
find_package(Armadillo REQUIRED)
find_package(PythonLibs REQUIRED)
# find_package(NumPy REQUIRED) 
find_program(PYTHON "python")
add_subdirectory(pybind11)
# find_package(PYBIND11 REQUIRED)

set(INC_DIR /usr/local/include)
set(LINK_DIR /usr/local/lib)
set(PYBIND_INC_DIR /Users/linux/venv/sq/include/site/python3.7/pybind11)
set(PYTHON_INC_DIR /usr/include/python3.6m)
# set(NUMPY_INC_DIR /home/linux/.local/lib/python3.6/site-packages/numpy/core/include/numpy)
include_directories(${INC_DIR} ${PYTHON_INC_DIR})
include_directories(${PYTHON_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
link_directories(${LINK_DIR})


set(PROJECT_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/arma_wrapper.h" 
                         "${CMAKE_CURRENT_SOURCE_DIR}/include/function_wrapper.h"
                         "${CMAKE_CURRENT_SOURCE_DIR}/include/armadillo_sparse.h")


aux_source_directory(./cpp DIR_SRCS)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
pybind11_add_module(cpp_exports ${DIR_SRCS})
target_link_libraries(cpp_exports PRIVATE ${ARMADILLO_LIBRARIES})


# add_executable(demo ${DIR_SRCS})
