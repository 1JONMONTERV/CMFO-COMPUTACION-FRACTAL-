
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


// Include C Header
extern "C" {
#include "cmfo_core.h"
}

// Modern C++ Wrapper Class
namespace cmfo {

class Vector7 {
   public:
    std::array<double, 7> data;

    Vector7() { data.fill(0.0); }
    Vector7(std::initializer_list<double> list) {
        if (list.size() != 7) throw std::invalid_argument("Size must be 7");
        std::copy(list.begin(), list.end(), data.begin());
    }

    // Tensor Product Operator (^)
    Vector7 operator^(const Vector7& other) const {
        Vector7 result;
        cmfo_tensor7(result.data.data(), this->data.data(), other.data.data());
        return result;
    }

    void print() const {
        std::cout << "[ ";
        for (auto v : data) std::cout << std::fixed << std::setprecision(4) << v << " ";
        std::cout << "]" << std::endl;
    }
};

}  // namespace cmfo

int main() {
    std::cout << "=== CMFO C++ Interop Demo ===" << std::endl;
    std::cout << "Phi Constant: " << cmfo_phi() << std::endl;

    cmfo::Vector7 a = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    cmfo::Vector7 b = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};

    std::cout << "\nVector A:" << std::endl;
    a.print();
    std::cout << "Vector B:" << std::endl;
    b.print();

    std::cout << "\nCalculating Tensor Product (A ^ B)..." << std::endl;
    cmfo::Vector7 c = a ^ b;  // Using overloaded operator calling C core

    std::cout << "Result:" << std::endl;
    c.print();

    std::cout << "\n[SUCCESS] C++ Wrapper functional." << std::endl;
    return 0;
}
