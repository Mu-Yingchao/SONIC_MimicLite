#ifndef COEFFS_HPP
#define COEFFS_HPP

#include <vector>
#include <cmath>

namespace parallel_ankle {

inline double chebyshev_T(int n, double x) {
    return std::cos(n * std::acos(std::max(-1.0, std::min(1.0, x))));
}

inline double chebyshev_T_deriv(int n, double x) {
    if (n == 0) return 0.0;
    double theta = std::acos(std::max(-1.0, std::min(1.0, x)));
    double sin_theta = std::sin(theta);
    if (std::fabs(sin_theta) < 1e-10) {
        return n * n * (n % 2 == 0 ? 1 : -1);
    }
    return n * std::sin(n * theta) / sin_theta;
}

struct ChebyshevConfig {
    int a_order;
    int b_order;
    double a_min;
    double a_max;
    double b_min;
    double b_max;
    std::vector<std::vector<double>> pit_coeffs;
    std::vector<std::vector<double>> rol_coeffs;
};

void load_coeffs(ChebyshevConfig& config);

} // namespace parallel_ankle

#endif // COEFFS_HPP
