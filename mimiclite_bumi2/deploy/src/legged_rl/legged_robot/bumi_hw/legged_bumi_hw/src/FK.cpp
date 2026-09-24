#include "FK.hpp"
#include "coeffs.hpp"
#include <cmath>

namespace parallel_ankle {

// inline double chebyshev_T(int n, double x) {
//     return std::cos(n * std::acos(std::max(-1.0, std::min(1.0, x))));
// }

void FK(double a, double b, double& Pit, double& Rol, const ChebyshevConfig& config) {
    // 归一化到[-1, 1]区间
    double a_norm = 2.0 * (a - config.a_min) / (config.a_max - config.a_min) - 1.0;
    double b_norm = 2.0 * (b - config.b_min) / (config.b_max - config.b_min) - 1.0;

    // 使用Chebyshev多项式计算Pit
    Pit = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double T_i = chebyshev_T(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            Pit += config.pit_coeffs[i][j] * T_i * chebyshev_T(j, b_norm);
        }
    }

    // 使用Chebyshev多项式计算Rol
    Rol = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double T_i = chebyshev_T(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            Rol += config.rol_coeffs[i][j] * T_i * chebyshev_T(j, b_norm);
        }
    }
}

} // namespace parallel_ankle
