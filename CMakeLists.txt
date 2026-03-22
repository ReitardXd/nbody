cmake_minimum_required(VERSION 3.16)
project(nbody)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3  REQUIRED)
find_package(GLEW   REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP)

# ImGui — compile from source
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui glfw OpenGL::GL)

add_executable(nbody src/main.cpp)

target_include_directories(nbody PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/vendor
    ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(nbody OpenGL::GL glfw GLEW::GLEW imgui ${FREETYPE_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody OpenMP::OpenMP_CXX)
endif()
