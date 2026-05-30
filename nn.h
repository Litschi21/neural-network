#pragma once

#include "matrix.h"
#include <memory>
#include <vector>
#include <functional>
#include <string>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix2D forward(const Matrix2D& input) = 0;
    virtual Matrix2D backward(const Matrix2D& outputGradient, double lr) = 0;
};

class ActivationLayer: public Layer {
    std::function<Matrix2D(Matrix2D&)> func;
    std::function<double(double)> deriv;
    Matrix2D m_lastInput{};
    std::string m_type;

public:
    ActivationLayer(std::function<Matrix2D(Matrix2D&)> f, std::function<double(double)> d);

    Matrix2D forward(const Matrix2D& input) override;
    Matrix2D backward(const Matrix2D& outputGradient, double lr) override;
};

class LinearLayer: public Layer {
    std::vector<double> m_biases{};
    Matrix2D m_weights{};
    Matrix2D m_lastInput{};

public:
    explicit LinearLayer(const int inputSize, const int outputSize);

    std::vector<double> getBiases() const;
    Matrix2D getWeights() const;

    Matrix2D forward(const Matrix2D& input) override;
    Matrix2D backward(const Matrix2D& outputGradient, double lr) override;
};

class Sequential {
    std::vector<std::unique_ptr<Layer>> m_layers;

public:
    template <typename... Args>
    Sequential(Args&&... args) {
        (addLayer(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))), ...);
    }

    void addLayer(std::unique_ptr<Layer> layer);
    void addLayer(const std::function<Matrix2D(Matrix2D)>& func, const std::function<double(double)>& deriv);
    void train(const Matrix2D& inputs, const Matrix2D& targets, int epochs, double lr=1e-3);
};

class NeuralNet {
    // Parameters are passed to constructor but currently unused
    // Keeping them for future API compatibility
    [[maybe_unused]] int m_inputAmt;
    [[maybe_unused]] int m_hiddenAmt;
    [[maybe_unused]] int m_outputAmt;

public:
    NeuralNet(int inputAmt, int hiddenAmt, int outputAmt);

    static double ReLU(double x);
    static double Tanh(double x);
    static double Sigmoid(double x);

    static Matrix2D ApplyReLU(Matrix2D& mat);
    static Matrix2D ApplyTanh(Matrix2D& mat);
    static Matrix2D ApplySigmoid(Matrix2D& mat);

    static double DReLU(double x);
    static double DTanh(double x);
    static double DSigmoid(double x);

    static double MSE(const Matrix2D& pred, const Matrix2D& actual);
    static double CEL(size_t M, const std::vector<double>& pred, const std::vector<double>& actual);

    static Matrix2D DMSE(const Matrix2D& pred, const Matrix2D& actual);
    static std::vector<double> DCEL(size_t M, const std::vector<double>& pred, const std::vector<double>& actual);
};
