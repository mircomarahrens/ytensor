message(STATUS "Processing directory " ${CMAKE_CURRENT_SOURCE_DIR})

cmake_minimum_required(VERSION 3.20)

set(CMAKE_TOOLCHAIN_FILE /home/mircomarahrens/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(tensor CXX)

# git clone https://github.com/microsoft/vcpkg.git
# activate vcpkg manifest mode
set(VCPKG_FEATURE_FLAGS "versions")

# cmake flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Googletest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# test via gtest
# message(STATUS "GoogleTest process triggered.")
# # GoogleTest: git clone https://github.com/google/googletest.git
# add_subdirectory(googletest)
# include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

enable_testing()

add_executable(tensor_test
        Test.cpp
        test/TensorTest.cpp)

target_include_directories(tensor_test PUBLIC
    include/)

target_link_libraries(tensor_test     
        ${CMAKE_THREAD_LIBS_INIT}
        gmock
        gtest
        gmock_main
        gtest_main)

include(GoogleTest)
gtest_discover_tests(tensor_test)
