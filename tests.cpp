#include <functional>
#include <gtest/gtest.h>
#include "matrix.h"
#include "nn.h"
#include <stdexcept>

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
    EXPECT_EQ(mat.matvec(vec), ans);
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

TEST(TEST_LL, TEST_CONSTRUCTOR) {
    LinearLayer ll{ 1, 1 };
    for (const double bias : ll.getBiases()) {
        EXPECT_EQ(bias, 0.0);
    }

    Matrix2D weights{ ll.getWeights() };
    for (size_t row; row < weights.getMat().size(); ++row) {
        for (size_t i; i < weights.getMat().at(row).size(); ++i) {
            const double item{ weights.getMat().at(row).at(i) };
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
	EXPECT_THROW(NeuralNet::CEL(2, {1, 1}, {0.9}),       std::invalid_argument);
	EXPECT_THROW(NeuralNet::CEL(2, {1}, {0.9}),          std::invalid_argument);
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
	std::vector<double> res{ NeuralNet::DCEL(2, {1, 0}, {1, 0}) };
	EXPECT_NEAR(res[0], -1, 1e-9);
	EXPECT_NEAR(res[1], 0, 1e-9);
}
