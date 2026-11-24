#pragma once

#include <functional>
#include <vector>

class Matrix2D {
  std::vector<std::vector<double>> m_matrix;

 public:
  Matrix2D(const std::vector<std::vector<double>>& matrix);
  Matrix2D(const Matrix2D& matrix);
  Matrix2D();

  Matrix2D& operator=(const Matrix2D& matrix) = default;

  const std::vector<std::vector<double>>& getMat() const;
  void setMat(const Matrix2D& mat);
  void setMat(const std::vector<std::vector<double>>& mat);
  void addRow(const std::vector<double>& vec);

  static double dot(const std::vector<double>& vec1, const std::vector<double>& vec2);
  static Matrix2D transpose(const Matrix2D& mat);
  std::vector<double> matvec(const std::vector<double>& vec) const;
  void apply(std::function<double(double)> func);

  bool operator==(const Matrix2D& mat) const;
  Matrix2D operator*(const Matrix2D& mat) const;
  Matrix2D operator*(double d) const;
  Matrix2D operator+(const Matrix2D& mat) const;
  Matrix2D operator-(const Matrix2D& mat) const;
};
