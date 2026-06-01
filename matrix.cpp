#include <functional>
#include <limits>
#include "matrix.h"
#include <stdexcept>
#include <vector>

Matrix2D::Matrix2D() {}
Matrix2D::Matrix2D(const std::vector<std::vector<double>>& matrix) :
    m_matrix{ matrix } {}

Matrix2D::Matrix2D(const Matrix2D& matrix) :
	m_matrix{ matrix.getMat() } {}

const std::vector<std::vector<double>>& Matrix2D::getMat() const { return m_matrix; }
void Matrix2D::setMat(const Matrix2D& mat) { m_matrix = mat.m_matrix; }
void Matrix2D::setMat(const std::vector<std::vector<double>>& mat) { m_matrix = mat; }
void Matrix2D::addRow(const std::vector<double>& vec) { m_matrix.emplace_back(vec); }

double Matrix2D::dot(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) throw std::invalid_argument("Vectors have different size.");

        double dotProd{};
        for (size_t i{}; i < vec1.size(); ++i) {
            dotProd += vec1[i] * vec2[i];
        }

        return dotProd;
    }

std::vector<double> Matrix2D::matvec(const std::vector<double>& vec) const {
    if (m_matrix.empty() || m_matrix[0].size() != vec.size()) throw std::invalid_argument("Matrix & Vector have different sizes.");

    std::vector<double> output{};
    for (size_t row{}; row < m_matrix.size(); ++row) {
        output.emplace_back(dot(m_matrix[row], vec));
    }

    return output;
}

Matrix2D Matrix2D::transpose(const Matrix2D& mat) {
	const auto& data{ mat.getMat() };
	if (data.empty()) return {};

	size_t rows{ data.size() };
	size_t cols{ data[0].size() };

	Matrix2D transposed{};
	for (size_t col{}; col < cols; ++col) {
		std::vector<double> newRow;
		newRow.reserve(rows);

		for (size_t row{}; row < rows; ++row) { newRow.emplace_back(data[row][col]); }
		transposed.addRow(newRow);
	}

	return transposed;
}

void Matrix2D::apply(std::function<double(double)> func) {
	for (auto& row : m_matrix) {
		for (double& val : row) {
			val = func(val);
		}
	}
}

bool Matrix2D::operator==(const Matrix2D& mat) const {
    if (m_matrix.size() != mat.m_matrix.size()) return false;

    for (size_t row{}; row < m_matrix.size(); ++row) {
        if (m_matrix[row].size() != mat.m_matrix[row].size()) return false;
        for (size_t i{}; i < m_matrix[row].size(); ++i) {
            if (m_matrix[row][i] != mat.m_matrix[row][i]) return false;
        }
    }

    return true;
}

Matrix2D Matrix2D::operator*(const Matrix2D& mat) const {
	if (m_matrix.size() == 0 || mat.m_matrix.size() == 0) { return Matrix2D(); }

	const size_t r1{ m_matrix.size() };
	const size_t c1{ m_matrix[0].size() };
	const size_t r2{ mat.m_matrix.size() };
	const size_t c2{ mat.m_matrix[0].size() };

	if (c1 != r2) { throw std::invalid_argument("Matrix A's column amount not equal to Matrix B's row amount. Unable to multiply."); }
	
	Matrix2D res{};
	res.m_matrix.reserve(r1);
	for (size_t i{}; i < r1; ++i) {
		std::vector<double> row{};
		row.reserve(c2);

		for (size_t j{}; j < c2; ++j) {
			std::vector<double> col{};
			for (size_t k{}; k < r2; ++k) {
				col.emplace_back(mat.m_matrix[k][j]);
			}

			row.emplace_back(Matrix2D::dot(m_matrix[i], col));
		}

		res.addRow(row);
	}

	return res;
}

Matrix2D Matrix2D::operator*(const double d) const {
	Matrix2D newMat{ *this };
	for (auto& row : newMat.m_matrix) {
		for (double& val : row) {
			val *= d;
		}
	}

	return newMat;
}

Matrix2D Matrix2D::operator+(const Matrix2D& mat) const {
    if (m_matrix.size() != mat.m_matrix.size()) throw std::invalid_argument("Matrices have different size.");
    
    Matrix2D newMat;
    newMat.m_matrix.reserve(m_matrix.size());
    for (size_t row{}; row < m_matrix.size(); ++row) {
        if (m_matrix[row].size() != mat.m_matrix[row].size()) throw std::invalid_argument("Row size mismatch");

        std::vector<double> sumRow{};
        sumRow.reserve(m_matrix[row].size());
        for (size_t i{}; i < m_matrix[row].size(); ++i) {
            sumRow.emplace_back(m_matrix[row][i] + mat.m_matrix[row][i]);
        }

        newMat.addRow(sumRow);
    }

    return newMat;
}

Matrix2D Matrix2D::operator-(const Matrix2D& mat) const {
    if (m_matrix.size() != mat.m_matrix.size()) throw std::invalid_argument("Matrices have different size.");

	Matrix2D newMat;
	for (size_t row{}; row < m_matrix.size(); ++row) {
		std::vector<double> subRow{};
		for (size_t i{}; i < m_matrix[row].size(); ++i) {
			subRow.emplace_back(m_matrix[row][i] - mat.m_matrix[row][i]);
		}

		newMat.addRow(subRow);
	}

	return newMat;
}
