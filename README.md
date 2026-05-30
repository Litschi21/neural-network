# neural-network
A lightweight, user-friendly Neural Network written from scratch without external libraries. It includes forward- and backpropagation as well as gradient descent.

## Features
- 2D Matrix Library (`Matrix2D`): Custom implementation supporting Matrix Multiplication, Addition, Subtraction, Transposition, and more which are required for Neural Networks.
- Layers: Implements a base `Layer` class as well as two subclasses (`LinearLayer` and `ActivationLayer`).
- Activation Functions: Built-in implementations of `ReLU`, `Tanh`, and `Sigmoid` alongside their derivatives.
- Loss Functions: `MSE` (Mean Squared Error) and `CEL` (Cross-Entropy Loss) also come built-in.
- Flexible Network Architecture: Features a `Sequential` class similar to that of [Pytorch](https://github.com/pytorch/pytorch), where you can customize the layout of your Neural Network easily.

## Prerequisites
- C++ Compiler supporting C++23
- GoogleTest (only required for running the test suite)

## Running the Main Application
You can quickly compile and run the main application by using the included `Makefile`.
```bash
make
./main.exe
```

## Running the Tests
The same applies for compiling and running the tests.
```bash
make tests
./tests.exe
```

## Usage
```cpp
#include "matrix.h"
#include "nn.h"

int main() {
    Matrix2D inputs;
    Matrix2D outputs;
    
    inputs.addRow({0.1, 0.2});
    outputs.addRow({0.2});

    // Define Model Architecture
    Sequential model(
        LinearLayer(2, 4), 
        ActivationLayer(
            NeuralNet::ApplyTanh, 
            NeuralNet::DTanh
        ),
        LinearLayer(4, 1)
    );

    // Train the model (inputs, targets, epochs, learning rate)
    model.train(inputs, outputs, 1000, 0.1);

    return 0;
}
```
