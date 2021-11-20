cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cnn-classification)
set(CMAKE_PREFIX_PATH /home/schwarm/pytorch-cpp-inference/libtorch/share/cmake)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

#find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${GTEST_INCLUDE_DIRS})
add_library(json11_shared SHARED json11.cpp)


add_executable(cnn-inference main.cc base64.cc ../infer.cc ../../../utils/opencvutils.cc ../../../utils/torchutils.cc)


target_link_libraries(cnn-inference "${TORCH_LIBRARIES}")
target_link_libraries(cnn-inference "${OpenCV_LIBS}")
target_link_libraries(cnn-inference ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
target_link_libraries(cnn-inference json11_shared)



#add_executable(MyTests test.cpp)
#target_link_libraries(MyTests ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
#add_test(Test MyTests)
#enable_testing()


set_property(TARGET cnn-inference PROPERTY CXX_STANDARD 11)
set_property(TARGET cnn-inference PROPERTY OUTPUT_NAME predict)
