#include <cmath>
#include <functional>
#include <iostream>
#include "matrix.h"
#include "nn.h"
#include <numbers>
#include <random>

static double randD(const double min, const double max) {
	thread_local std::mt19937 rng{ std::random_device{}() };
	std::uniform_real_distribution<double> dist{ min, max };
	return dist(rng);
}

ActivationLayer::ActivationLayer(std::function<Matrix2D(Matrix2D&)> f, std::function<double(double)> d) : func(f), deriv(d) {}
Matrix2D ActivationLayer::forward(const Matrix2D& input) { m_lastInput = input; return func(m_lastInput); }
Matrix2D ActivationLayer::backward(const Matrix2D& outputGradient, [[maybe_unused]] const double lr) {
	Matrix2D dZ{ m_lastInput };

	auto dZMat{ dZ.getMat() };
	auto oGmat{ outputGradient.getMat() };
	auto lastInMat{ m_lastInput.getMat() };

	for (size_t i{}; i < oGmat.size(); ++i) {
		for (size_t j{}; j < oGmat[i].size(); ++j) {
			dZMat[i][j] = oGmat[i][j] * deriv(lastInMat[i][j]);
		}
	}

	dZ.setMat(dZMat);
	return dZ;
}

LinearLayer::LinearLayer(const int n_in, const int n_out) {
        for (int i{}; i < n_out; ++i) {
            m_biases.emplace_back(0);
        }

        for (int i{}; i < n_out; ++i) {
            std::vector<double> row{};
            for (int j{}; j < n_in; ++j) {
				double lim{ std::sqrt(6 / (n_in + n_out)) };
                row.emplace_back(randD(-lim, lim));
            }

            m_weights.addRow(row);
        }
    }

std::vector<double> LinearLayer::getBiases() const { return m_biases; }
Matrix2D LinearLayer::getWeights() const { return m_weights; }
Matrix2D LinearLayer::forward(const Matrix2D& input) {
	m_lastInput = input;

	Matrix2D tInput{ Matrix2D::transpose(input) };
	Matrix2D Z{ m_weights * tInput }; // Bias Matrix

    auto mat{ Z.getMat() };
    for (size_t row{}; row < mat.size(); ++row) {
        for (size_t col{}; col < mat[row].size(); ++col) {
            mat[row][col] += m_biases[row];
        }
    }

	Z.setMat(mat);
    return Matrix2D::transpose(Z);
}

Matrix2D LinearLayer::backward(const Matrix2D& outputGradient, const double lr) {
	Matrix2D tGrad { Matrix2D::transpose(outputGradient) };
	Matrix2D tInput{ Matrix2D::transpose(m_lastInput) };

	Matrix2D dW{ tGrad * Matrix2D::transpose(tInput) };      // Gradient of Weights
	Matrix2D dX{ Matrix2D::transpose(m_weights) * tGrad };   // Gradient of Inputs

	std::vector<double> db(m_biases.size(), 0.0);            // Gradient of Biases
	auto ogMat{ tGrad.getMat() };
	for (size_t i{}; i < ogMat.size(); ++i) {
		for (size_t j{}; j < ogMat[i].size(); ++j) {
			db[i] += ogMat[i][j];
		}
	}

	for (size_t i{}; i < m_biases.size(); ++i) {
		m_biases[i] -= db[i] * lr;
	}

	m_weights = m_weights - dW * lr;
	return Matrix2D::transpose(dX);
}

void Sequential::addLayer(std::unique_ptr<Layer> layer) { m_layers.emplace_back(std::move(layer)); }
void Sequential::addLayer(const std::function<Matrix2D(Matrix2D)>& func, const std::function<double(double)>& deriv) {
	m_layers.emplace_back(std::make_unique<ActivationLayer>(func, deriv));
}

void Sequential::train(const Matrix2D& inputs, const Matrix2D& targets, const int epochs, const double lr) {
	for (int e{ 1 }; e <= epochs; ++e) {
		Matrix2D curr{ inputs };
		for (const auto& layer : m_layers) { curr = layer->forward(curr); }

		double loss{ NeuralNet::MSE(curr, targets) };
		Matrix2D grad{ NeuralNet::DMSE(curr, targets) };
		if (e <= 5 || e % 10 == 0) {
			std::cout << "Epoch #" << e << " - loss=" << loss << " | " << (1 - loss) * 100 << "%" << '\n';
		}

		for (size_t i{ m_layers.size() }; i > 0; --i) {
			grad = m_layers[i - 1]->backward(grad, lr);
		}
	}
}

double NeuralNet::ReLU(const double x) { return std::max(0.0, x); }
double NeuralNet::Tanh(const double x) { return std::tanh(x); }
double NeuralNet::Sigmoid(const double x) { return 1 / (1 + std::pow(std::numbers::e, -x)); }

Matrix2D NeuralNet::ApplyReLU(Matrix2D& mat) {
	mat.apply(static_cast<double(*)(double)>(NeuralNet::ReLU));
	return mat;
}

Matrix2D NeuralNet::ApplyTanh(Matrix2D& mat) {
	mat.apply(static_cast<double(*)(double)>(NeuralNet::Tanh));
	return mat;
}

Matrix2D NeuralNet::ApplySigmoid(Matrix2D& mat) {
	mat.apply(static_cast<double(*)(double)>(NeuralNet::Sigmoid));
	return mat;
}

double NeuralNet::DReLU(const double x) { return x > 0 ? 1 : 0; }
double NeuralNet::DTanh(const double x) { return 1 - std::pow(std::tanh(x), 2); }
double NeuralNet::DSigmoid(const double x) { return NeuralNet::Sigmoid(x) * (1 - NeuralNet::Sigmoid(x)); }

double NeuralNet::MSE(const Matrix2D& pred, const Matrix2D& actual) {
	const auto& predMat  { pred.getMat() };
	const auto& actualMat{ actual.getMat() };

	if (predMat.size() != actualMat.size() || predMat[0].size() != actualMat[0].size()) {
		throw std::invalid_argument("Prediction and Actual matrices must have identical dimensions.");
	}

	size_t rows{ predMat.size() };
	size_t cols{ predMat[0].size() };
	double n{ static_cast<double>(rows * cols) };

	double sum{};
	for (size_t i{}; i < rows; ++i) {
		for (size_t j{}; j < cols; ++j) {
			sum += std::pow(predMat[i][j] - actualMat[i][j], 2);
		}
	}

	return sum / n;
}

Matrix2D NeuralNet::DMSE(const Matrix2D& pred, const Matrix2D& actual) {
	const auto& predMat  { pred.getMat() };
	const auto& actualMat{ actual.getMat() };

	if (predMat.size() != actualMat.size() || predMat[0].size() != actualMat[0].size()) {
		throw std::invalid_argument("Prediction and Actual matrices must have identical dimensions.");
	}

	size_t rows{ predMat.size() };
	size_t cols{ predMat[0].size() };
	double n{ static_cast<double>(rows * cols) };

	std::vector<std::vector<double>> totalGrads(rows, std::vector<double>(cols));
	for (size_t i{}; i < rows; ++i) {
		for (size_t j{}; j < cols; ++j) {
			totalGrads[i][j] = 2.0 * (predMat[i][j] - actualMat[i][j]) / n;
		}
	}

	Matrix2D res;
	res.setMat(totalGrads);
	return res;
}

double NeuralNet::CEL(const size_t M, const std::vector<double>& pred, const std::vector<double>& actual) {
	if (pred.size() != actual.size()) { throw std::invalid_argument("Prediction and Actual vectors not same size."); }
	if (pred.size() != M) { throw std::invalid_argument("Class amount must match length of vectors."); }
	if (M < 2) { throw std::invalid_argument("Must have at least two classes."); }
	
	double loss{};
	for (size_t i{}; i < M; ++i) {
		loss += -(actual[i] * std::log(pred[i] + 1e-15));
	}

	return loss;
}

std::vector<double> NeuralNet::DCEL(const size_t M, const std::vector<double>& pred, const std::vector<double>& actual) {
	if (pred.size() != actual.size()) { throw std::invalid_argument("Prediction and Actual vectors not same size."); }
	if (pred.size() != M) { throw std::invalid_argument("Class amount must match length of vectors."); }
	if (M < 2) { throw std::invalid_argument("Must have at least two classes."); }

	std::vector<double> losses(pred.size());
	for (size_t i{}; i < pred.size(); ++i) {
		losses[i] = -actual[i] / (pred[i] + 1e-15);
	}

	return losses;
}
