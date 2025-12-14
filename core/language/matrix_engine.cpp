#include "matrix_engine.hpp"
#include <iomanip>

Matrix7x7::Matrix7x7() { data.fill(Complex(0, 0)); }

Matrix7x7 Matrix7x7::Identity() {
  Matrix7x7 m;
  for (int i = 0; i < 7; ++i) {
    m.at(i, i) = Complex(1.0, 0.0);
  }
  return m;
}

Matrix7x7 Matrix7x7::Diagonal(const std::array<double, 7> &values) {
  Matrix7x7 m;
  for (int i = 0; i < 7; ++i) {
    m.at(i, i) = Complex(values[i], 0.0);
  }
  return m;
}

Complex &Matrix7x7::at(int row, int col) { return data[row * 7 + col]; }

const Complex &Matrix7x7::at(int row, int col) const {
  return data[row * 7 + col];
}

Matrix7x7 Matrix7x7::operator*(const Matrix7x7 &other) const {
  Matrix7x7 result;
  // Standard Matrix Multiplication
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      Complex sum(0, 0);
      for (int k = 0; k < 7; ++k) {
        sum += this->at(i, k) * other.at(k, j);
      }
      result.at(i, j) = sum;
    }
  }
  return result;
}

Matrix7x7 Matrix7x7::operator*(double scalar) const {
  Matrix7x7 result;
  for (int i = 0; i < 49; ++i) {
    result.data[i] = this->data[i] * scalar;
  }
  return result;
}

Matrix7x7 Matrix7x7::operator+(const Matrix7x7 &other) const {
  Matrix7x7 result;
  for (int i = 0; i < 49; ++i) {
    result.data[i] = this->data[i] + other.data[i];
  }
  return result;
}

Matrix7x7 Matrix7x7::adjoint() const {
  Matrix7x7 adj;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      adj.at(j, i) = std::conj(this->at(i, j));
    }
  }
  return adj;
}

Complex Matrix7x7::trace() const {
  Complex tr(0, 0);
  for (int i = 0; i < 7; ++i) {
    tr += this->at(i, i);
  }
  return tr;
}

bool Matrix7x7::is_unitary(double epsilon) const {
  Matrix7x7 prod = (*this) * this->adjoint();
  Matrix7x7 id = Matrix7x7::Identity();

  // Check norm difference
  double error = 0.0;
  for (int i = 0; i < 49; ++i) {
    error += std::abs(prod.data[i] - id.data[i]);
  }

  return error < epsilon;
}

void Matrix7x7::print() const {
  std::cout << std::fixed << std::setprecision(3);
  for (int i = 0; i < 7; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < 7; ++j) {
      const Complex &c = at(i, j);
      std::cout << c.real();
      if (c.imag() >= 0)
        std::cout << "+" << c.imag() << "j ";
      else
        std::cout << c.imag() << "j ";
    }
    std::cout << "]" << std::endl;
  }
}
