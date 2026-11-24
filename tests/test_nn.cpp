#include <gtest/gtest.h>

#include <stdexcept>

#include "matrix.h"
#include "nn.h"

TEST(TEST_LL, TEST_CONSTRUCTOR) {
  LinearLayer ll{1, 1};
  for (const double bias : ll.getBiases()) {
    EXPECT_EQ(bias, 0.0);
  }

  Matrix2D weights{ll.getWeights()};
  for (size_t row{}; row < weights.getMat().size(); ++row) {
    for (size_t i{}; i < weights.getMat().at(row).size(); ++i) {
      const double item{weights.getMat().at(row).at(i)};
      EXPECT_TRUE(item >= 0.0 && item <= 1.0);
    }
  }
}

TEST(TEST_NN, TEST_RELU) {
  EXPECT_EQ(NeuralNet::ReLU(15), 15);
  EXPECT_EQ(NeuralNet::ReLU(-17), 0.0);
}

TEST(TEST_NN, TEST_TANH) {
  EXPECT_EQ(NeuralNet::Tanh(0), 0.0);
  EXPECT_EQ(NeuralNet::Tanh(1000), 1.0);
  EXPECT_EQ(NeuralNet::Tanh(-1000), -1.0);
}

TEST(TEST_NN, TEST_SIGMOID) {
  EXPECT_EQ(NeuralNet::Sigmoid(0), 0.5);
  EXPECT_EQ(NeuralNet::Sigmoid(1000), 1.0);
  EXPECT_EQ(NeuralNet::Sigmoid(-1000), 0.0);
}

TEST(TEST_NN, TEST_DRELU) {
  EXPECT_EQ(NeuralNet::DReLU(0.1), 1.0);
  EXPECT_EQ(NeuralNet::DReLU(-5), 0.0);
}

TEST(TEST_NN, TEST_DTANH) {
  EXPECT_EQ(NeuralNet::DTanh(0), 1.0);
  EXPECT_EQ(NeuralNet::DTanh(1000), 0.0);
  EXPECT_EQ(NeuralNet::DTanh(-1000), 0.0);
}

TEST(TEST_NN, TEST_DSIGMOID) {
  EXPECT_EQ(NeuralNet::DSigmoid(0), 0.25);
  EXPECT_EQ(NeuralNet::DSigmoid(1000), 0.0);
  EXPECT_EQ(NeuralNet::DSigmoid(-1000), 0.0);
}

TEST(TEST_NN, TEST_MSE) {
  Matrix2D pred({{1, 2}, {3, 4}});
  Matrix2D actual({{2, 4}, {6, 8}});

  EXPECT_NEAR(NeuralNet::MSE(pred, actual), 7.5, 1e-9);
  EXPECT_NEAR(NeuralNet::MSE(pred, pred), 0.0, 1e-9);

  Matrix2D wrong_size({{1.0, 2.0}});
  EXPECT_THROW(NeuralNet::MSE(pred, wrong_size), std::invalid_argument);
}

TEST(TEST_NN, TEST_CEL) {
  EXPECT_THROW(NeuralNet::CEL(2, {1, 1}, {0.9}), std::invalid_argument);
  EXPECT_THROW(NeuralNet::CEL(2, {1}, {0.9}), std::invalid_argument);
  EXPECT_THROW(NeuralNet::CEL(1, {1, 1}, {0.9, 0.87}), std::invalid_argument);
  EXPECT_NEAR(NeuralNet::CEL(2, {1, 1}, {1, 1}), 0, 1e-9);
  EXPECT_TRUE(NeuralNet::CEL(2, {1, 1}, {0, 0}) != 0);
}

TEST(TEST_NN, TEST_DMSE) {
  Matrix2D pred({{2.0, 4.0}});
  Matrix2D actual({{1.0, 2.0}});

  Matrix2D expected({{1.0, 2.0}});
  EXPECT_EQ(NeuralNet::DMSE(pred, actual), expected);
}

TEST(TEST_NN, TEST_DCEL) {
  std::vector<double> res{NeuralNet::DCEL(2, {1, 0}, {1, 0})};
  EXPECT_NEAR(res[0], -1, 1e-9);
  EXPECT_NEAR(res[1], 0, 1e-9);
}
