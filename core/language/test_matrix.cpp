#include "matrix_engine.hpp"
#include <iostream>
#include <vector>

void test_identity() {
  std::cout << "Test 1: Identity Multiplication... ";
  Matrix7x7 id = Matrix7x7::Identity();
  Matrix7x7 m = Matrix7x7::Diagonal({1, 2, 3, 4, 5, 6, 7});

  Matrix7x7 res = m * id;

  // Check diagonal
  bool ok = true;
  for (int i = 0; i < 7; ++i) {
    if (std::abs(res.at(i, i) - Complex(i + 1, 0)) > 1e-9)
      ok = false;
  }

  if (ok)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}

void test_unitary() {
  std::cout << "Test 2: Unitary Logic (Rotation)... ";

  // Construct a Rotation Matrix (Unitary)
  Matrix7x7 rot = Matrix7x7::Identity();
  double theta = std::atan(1.0 / PHI);
  double c = std::cos(theta);
  double s = std::sin(theta);

  // Rotate dimensions 0 and 1
  rot.at(0, 0) = Complex(c, 0);
  rot.at(0, 1) = Complex(-s, 0);
  rot.at(1, 0) = Complex(s, 0);
  rot.at(1, 1) = Complex(c, 0);

  bool is_u = rot.is_unitary();

  if (is_u)
    std::cout << "PASS" << std::endl;
  else {
    std::cout << "FAIL (Not Unitary)" << std::endl;
    rot.print();
  }
}

void test_composition() {
  std::cout << "Test 3: Composition (Trace Check)... ";

  // A and B
  std::array<double, 7> vals = {1, 1, 1, 1, 1, 1, 1};
  Matrix7x7 A = Matrix7x7::Diagonal(vals);
  Matrix7x7 B = Matrix7x7::Diagonal(vals);

  // A * B = I * I = I
  Matrix7x7 C = A * B;

  Complex tr = C.trace();
  if (std::abs(tr - Complex(7, 0)) < 1e-9)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL (Trace = " << tr << ")" << std::endl;
}

int main() {
  std::cout << "=== CMFO C++ CORE ENGINE TEST ===" << std::endl;

  test_identity();
  test_unitary();
  test_composition();

  std::cout << "=================================" << std::endl;
  return 0;
}
