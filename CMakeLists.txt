cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -w -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w  -O3 -march=native")

# -O0不优化，实时性会下降的比较厉害，只用于debug
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0")

#[[
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
]]

set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#add_subdirectory(libtorch-yolov5)
#LIST(APPEND EXTRA_LIBS libtorch-yolov5)
#message(${EXTRA_LIBS})
#find_package(OpenCV 3.0 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)

#   if(NOT OpenCV_FOUND)
#      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()
#set(OpenCV_DIR "/home/liuyuzhen/rd3_party/opencv-3.4.3/build")
message(${CMAKE_MODULE_PATH})
find_package(OpenCV 3.0 QUIET)
message(${OpenCV_INCLUDE_DIRS})
#find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
SET(EIGEN3_INCLUDE_DIR "/usr/local/include")
message(${EIGEN3_INCLUDE_DIR})
find_package(Pangolin REQUIRED)


set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
  message(STATUS "Torch library found!")
  message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
  message(FATAL_ERROR "Could not locate Torch" \n)
endif()
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/home/zpk/SLOT/ORB_SLAM2/TensorRT-8.4.0.6/include/)
link_directories(/home/zpk/SLOT/ORB_SLAM2/TensorRT-8.4.0.6/lib/)

aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src DEEPSORT_SRC_DIR)
add_library(deepsort SHARED ${DEEPSORT_SRC_DIR})
target_link_libraries(deepsort
        ${CUDA_LIBS} ${OpenCV_LIBS}
        cudart nvinfer nvonnxparser
        )
target_include_directories(deepsort PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/deepsort/include
        )


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src DEEPSORT_SRC_DIR)
add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Parameters.cc
src/MapObject.cc
src/matrix_utils.cc
src/g2o_Object.cc
src/DetectionObject.cc
src/MapObjectPoint.cc
src/TwoViewReconstruction.cc
src/ObjectKeyFrame.cpp
src/ObjectLocalMapping.cpp
src/YOLOdetector.cc
        #${DEEPSORT_SRC_DIR}
        )

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${TORCH_LIBRARIES}
        deepsort
        #${CUDA_LIBS}
        #cudart
        #nvinfer
        #nvonnxparser
)


target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}

        #${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/deepsort/include
)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

#add_executable(stereo_mynteye
#        Examples/Stereo/stereo_mynteye.cpp)
#target_link_libraries(stereo_mynteye ${PROJECT_NAME})
#
#add_executable(stereo_pointgrey
#        Examples/Stereo/stereo_pointgrey.cpp)
#target_link_libraries(stereo_pointgrey ${PROJECT_NAME})