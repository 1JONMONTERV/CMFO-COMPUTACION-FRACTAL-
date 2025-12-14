#include "matrix_engine.hpp"
#include <Python.h>
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

extern "C" {
#ifdef _WIN32
#define CMFO_API __declspec(dllexport)
#else
#define CMFO_API
#endif

CMFO_API void *Matrix7x7_Create() { return new Matrix7x7(); }
CMFO_API void Matrix7x7_Destroy(void *ptr) {
  delete static_cast<Matrix7x7 *>(ptr);
}

CMFO_API void Matrix7x7_SetIdentity(void *ptr) {
  if (ptr)
    *static_cast<Matrix7x7 *>(ptr) = Matrix7x7::Identity();
}

CMFO_API void Matrix7x7_Multiply(void *a, void *b, void *out) {
  if (a && b && out) {
    *static_cast<Matrix7x7 *>(out) =
        (*static_cast<Matrix7x7 *>(a)) * (*static_cast<Matrix7x7 *>(b));
  }
}

CMFO_API void Matrix7x7_Apply(void *mat_ptr, double *input_vec_real,
                              double *input_vec_imag, double *out_vec_real,
                              double *out_vec_imag) {
  if (!mat_ptr)
    return;
  auto *mat = static_cast<Matrix7x7 *>(mat_ptr);

  // Manual unrolled 7x7 multiplication for simple pointer arrays
  for (int i = 0; i < 7; i++) {
    std::complex<double> sum(0, 0);
    for (int j = 0; j < 7; j++) {
      std::complex<double> vec_val(input_vec_real[j], input_vec_imag[j]);
      sum += mat->at(i, j) * vec_val;
    }
    out_vec_real[i] = sum.real();
    out_vec_imag[i] = sum.imag();
  }
}

CMFO_API void Matrix7x7_Get(void *ptr, double *buffer_real,
                            double *buffer_imag) {
  if (!ptr)
    return;
  auto *mat = static_cast<Matrix7x7 *>(ptr);
  auto *data = mat->raw_data();
  for (int i = 0; i < 49; i++) {
    auto c = data[i];
    buffer_real[i] = c.real();
    buffer_imag[i] = c.imag();
  }
}

CMFO_API void Matrix7x7_Set(void *ptr, double *buffer_real,
                            double *buffer_imag) {
  if (!ptr)
    return;
  auto *mat = static_cast<Matrix7x7 *>(ptr);
  auto *data = mat->raw_data();
  for (int i = 0; i < 49; i++) {
    data[i] = std::complex<double>(buffer_real[i], buffer_imag[i]);
  }
}
}

// Python Module Boilerplate
static PyModuleDef CmfoCoreNativeModule = {PyModuleDef_HEAD_INIT,
                                           "cmfo_core_native",
                                           "CMFO Core Native Engine",
                                           -1,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL};

PyMODINIT_FUNC PyInit_cmfo_core_native(void) {
  return PyModule_Create(&CmfoCoreNativeModule);
}
