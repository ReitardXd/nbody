cmake_minimum_required(VERSION 3.16)
project(nbody)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3  REQUIRED)
find_package(GLEW   REQUIRED)
find_package(OpenMP)

add_executable(nbody src/main.cpp)
target_link_libraries(nbody OpenGL::GL glfw GLEW::GLEW)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody OpenMP::OpenMP_CXX)
endif()
