#include <chrono>
#include <cmath>
#include <iostream>
#include "matrix.h"
#include "nn.h"
#include <vector>

int main() {
	namespace ch = std::chrono;

	Matrix2D inputs;
	Matrix2D outputs;

	int epochs{ 1'000 };
	for (int i{}; i < epochs; ++i) {
		inputs.addRow({static_cast<double>(i) / epochs});
		outputs.addRow({std::pow(i, 2) / std::pow(epochs, 2)});
	}

	Sequential seq(
		LinearLayer(1, 4), ActivationLayer(NeuralNet::ApplyTanh, NeuralNet::DTanh),
		LinearLayer(4, 1)
	);

	auto start{ ch::system_clock::now() };
	seq.train(inputs, outputs, epochs, 0.1);
	auto end{ ch::system_clock::now() };
	auto dur{ ch::duration_cast<ch::milliseconds>(end - start) };

	std::cout << "\nTook " << dur.count() << "ms to run.\n";
	return 0;
}
