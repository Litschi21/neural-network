#include <gtest/gtest.h>

#include <stdexcept>

#include "matrix.h"
#include "nn.h"

TEST(TEST_MATRIX, TEST_CONSTRUCTOR) {
  Matrix2D mat1{};
  EXPECT_TRUE(mat1.getMat().empty());

  Matrix2D mat2{{{15, 20}, {20, 25}}};
  EXPECT_EQ(mat2.getMat().size(), 2);
}

TEST(TEST_MATRIX, TEST_DOT_PROD) {
  std::vector<double> vec1{1, 2};
  std::vector<double> vec2{3, 4};
  EXPECT_EQ(Matrix2D::dot(vec1, vec2), 11.0);

  vec2.emplace_back(5);
  EXPECT_THROW(Matrix2D::dot(vec1, vec2), std::invalid_argument);

  vec1 = {}, vec2 = {};
  EXPECT_EQ(Matrix2D::dot(vec1, vec2), 0.0);
}

TEST(TEST_MATRIX, TEST_MATVEC) {
  Matrix2D mat{{{1, 2}, {3, 4}}};
  std::vector<double> vec{5, 6};
  std::vector<double> ans{17, 39};
  EXPECT_EQ(mat.matvec(vec), ans);

  vec.emplace_back(7);
  EXPECT_THROW(mat.matvec(vec), std::invalid_argument);

  mat = {}, vec = {}, ans = {};
  EXPECT_THROW(mat.matvec(vec), std::invalid_argument);
}

TEST(TEST_MATRIX, TEST_TRANSPOSE) {
  Matrix2D mat{{{1, 2, 3}, {4, 5, 6}}};
  Matrix2D ans{{{1, 4}, {2, 5}, {3, 6}}};
  EXPECT_EQ(Matrix2D::transpose(mat), ans);
}

TEST(TEST_MATRIX, TEST_APPLY) {
  Matrix2D mat{{{-1000, 0}, {1, 1000}}};
  mat.apply(NeuralNet::ReLU);

  Matrix2D ans{{{0, 0}, {1, 1000}}};
  for (auto& row : mat.getMat()) {
    for (double val : row) {
      std::cout << val << ' ';
    }

    std::cout << '\n';
  }
  EXPECT_EQ(mat, ans);
}

TEST(TEST_MATRIX, TEST_MATMUL) {
  Matrix2D mat1{{{1, 2}, {3, 4}}};
  Matrix2D mat2{{{5, 6}, {7, 8}}};
  Matrix2D res{{{19, 22}, {43, 50}}};
  EXPECT_EQ(mat1 * mat2, res);

  mat2 = Matrix2D({{5}, {6}, {7}});
  EXPECT_THROW(mat1 * mat2, std::invalid_argument);

  mat1 = {}, res = {};
  EXPECT_EQ(mat1 * mat2, res);
}

TEST(TEST_MATRIX, TEST_MATADD) {
  Matrix2D mat1{{{1, 2}, {3, 4}}};
  Matrix2D mat2{{{5, 6}, {7, 8}}};
  Matrix2D ans{{{6, 8}, {10, 12}}};
  EXPECT_EQ(mat1 + mat2, ans);

  mat2 = {};
  EXPECT_THROW(mat1 + mat2, std::invalid_argument);

  mat1 = {}, ans = {};
  EXPECT_EQ(mat1 + mat2, ans);
}

TEST(TEST_MATRIX, TEST_MATSUB) {
  Matrix2D mat1{{{5, 6}, {7, 8}}};
  Matrix2D mat2{{{1, 2}, {3, 4}}};
  Matrix2D ans{{{4, 4}, {4, 4}}};
  EXPECT_EQ(mat1 - mat2, ans);

  mat2 = {};
  EXPECT_THROW(mat1 - mat2, std::invalid_argument);

  mat1 = {}, ans = {};
  EXPECT_EQ(mat1 - mat2, ans);
}
