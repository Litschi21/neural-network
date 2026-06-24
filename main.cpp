#include <chrono>
#include <cmath>
#include <iostream>
#include "matrix.h"
#include "nn.h"
#include <numbers>
#include <random>
#include <vector>

double randInt(const int min, const int max) {
	thread_local std::mt19937 rng{ std::random_device{}() };
	std::uniform_int_distribution<int> dist{ min, max };
	return dist(rng);
}

int main() {
	namespace ch = std::chrono;

	int epochs{ 1'000 };
	Matrix2D inputs;
	Matrix2D outputs;

	for (int i{}; i < epochs; ++i) {
		double x1{ randInt(-2, 2) };
		inputs.addRow({x1});

		double P0{ std::pow(std::numbers::e, -(std::pow((x1 + 0.5), 2) / 0.2)) };
		double P1{ std::pow(std::numbers::e, -(std::pow((x1 - 0.5), 2) / 0.2)) };
		outputs.addRow({P1 > P0 ? 1.0 : 0.0});
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
