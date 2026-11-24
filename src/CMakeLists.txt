add_library(project_library matrix.cpp nn.cpp)

target_compile_features(project_library PUBLIC cxx_std_23)
target_include_directories(project_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
