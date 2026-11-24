find_package(GTest REQUIRED)

add_executable(tests
  test_matrix.cpp
  test_nn.cpp
)

target_link_libraries(tests PRIVATE
  project_library
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)
