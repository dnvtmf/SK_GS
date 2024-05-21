cmake_minimum_required(VERSION 3.17)
project(wan_C CXX CUDA)

set(PROJECT_NAME _C)

#------ CUDA configure -----------
set(CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_ARCHITECTURES "native")
find_package(CUDAToolkit REQUIRED)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

message(STATUS "Found CUDA ${CUDA_TOOLKIT_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

execute_process(COMMAND which python)
execute_process(COMMAND python3-config --includes OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS ${PYTHON_INCLUDE})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${PYTHON_INCLUDE})
set(PYTHON_MODULE_PREFIX "")
execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS outputname:  ${PYTHON_MODULE_PREFIX}${PROJECT_NAME}${PYTHON_MODULE_EXTENSION})

#-------------------------- Torch config begin ------------------------#
execute_process(COMMAND python3 -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCH_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "." ";" TORCH_VERSION ${TORCH_VERSION})
list(GET TORCH_VERSION 0 TORCH_VERSION_MAJOR)
list(GET TORCH_VERSION 1 TORCH_VERSION_MINOR)
list(GET TORCH_VERSION 2 TORCH_VERSION_BUILD)
# math(EXPR TORCH_VERSION_NO "(${TORCH_VERSION_MAJOR} * 1000 + ${TORCH_VERSION_MINOR}) * 1000 + ${TORCH_VERSION_BUILD}")
message(STATUS "The version of torch: ${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}.${TORCH_VERSION_BUILD}")
# add_definitions(-DTORCH_VERSION=${TORCH_VERSION_NO})

execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Torch_DIR "${TORCH_CMAKE_DIR}/Torch")
message(STATUS "Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED)
message(STATUS "Torch CXX FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
link_libraries(${TORCH_LIBRARIES})
link_directories(${TORCH_CMAKE_DIR}/../../lib/)
link_libraries(torch_python) # to avoid undefined symbol _ZN8pybind116detail11t...

# execute_process(COMMAND python3 -c "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(*COMMON_NVCC_FLAGS)" 
#   OUTPUT_VARIABLE COMMON_NVCC_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
# message(STATUS "COMMON_NVCC_FLAGS: ${COMMON_NVCC_FLAGS}")
# string(REPLACE " " ";" COMMON_NVCC_FLAGS ${COMMON_NVCC_FLAGS}) # string to list
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${COMMON_NVCC_FLAGS})

add_definitions(-DTORCH_EXTENSION_NAME=${PROJECT_NAME})
add_definitions(-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__) # avoid more than one operator < match these operands
#-------------------------- Torch config end ------------------------#

#----------- OpenMP ------------------#
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#------------ Eigen3 --------------#
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
link_libraries(${Eigen3_CMAKE_DIR})
message(STATUS "use Eigen3 ${Eigen3_CMAK_DIR}")

#----------------- Option ----------------#
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wall -g -Wno-deprecated-declarations")
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17") # set in TorchConfig.cmake
# set(CUDAGEN -gencode arch=compute_60,code=sm_60)
# set(CUDAGEN ${CUDAGEN} -gencode arch=compute_61,code=sm_61)
# set(CUDAGEN ${CUDAGEN} -gencode arch=compute_70,code=sm_70)
# set(CUDAGEN ${CUDAGEN} -gencode arch=compute_75,code=sm_75)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v)
# LIST(APPEND CUDA_NVCC_FLAGS --maxrregcount=64)
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
LIST(APPEND CUDA_NVCC_FLAGS -rdc=true)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -fno-gnu-unique)
message(STATUS "CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/glm)

# 输出cmake所有变量
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

## find sources files
set(SOURCE_DIR "src")
LIST(APPEND SOURCE_DIR "src/other")
LIST(APPEND SOURCE_DIR "src/nerf")
LIST(APPEND SOURCE_DIR "src/ops_3d")
foreach(SRC_DIR IN LISTS SOURCE_DIR)
  set(SRC_TEMP)
  aux_source_directory(${SRC_DIR} SRC_TEMP)
  LIST(LENGTH SRC_TEMP num_files)
  message(STATUS "directory '${SRC_DIR}' have ${num_files} files")
  LIST(APPEND SOURCES ${SRC_TEMP})
endforeach()
 

CUDA_ADD_LIBRARY(${PROJECT_NAME} ${SOURCES} SHARED OPTIONS ${CUDAGEN})
target_link_libraries(${PROJECT_NAME} cudadevrt)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
message(STATUS install path: ${PROJECT_SOURCE_DIR})
# install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR})