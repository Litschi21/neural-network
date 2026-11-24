cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME neural_network)
project(${PROJECT_NAME} VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
add_subdirectory(tests)
add_subdirectory(src)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/matrix.cpp
  src/nn.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wnull-dereference -Wformat=2 -Werror)
