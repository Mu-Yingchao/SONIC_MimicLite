#include "Jacobian.hpp"
#include "IK.hpp"
#include "FK.hpp"
#include "coeffs.hpp"
#include <cmath>

namespace parallel_ankle {

// inline double chebyshev_T(int n, double x) {
//     return std::cos(n * std::acos(std::max(-1.0, std::min(1.0, x))));
// }

// inline double chebyshev_T_deriv(int n, double x) {
//     if (n == 0) return 0.0;
//     double theta = std::acos(std::max(-1.0, std::min(1.0, x)));
//     double sin_theta = std::sin(theta);
//     if (std::fabs(sin_theta) < 1e-10) {
//         return n * n * (n % 2 == 0 ? 1 : -1);
//     }
//     return n * std::sin(n * theta) / sin_theta;
// }

void FK_V_JB(double a, double b, double /*Pit*/, double /*Rol*/, double jacob[4], const ChebyshevConfig& config) {
    // 归一化
    double a_norm = 2.0 * (a - config.a_min) / (config.a_max - config.a_min) - 1.0;
    double b_norm = 2.0 * (b - config.b_min) / (config.b_max - config.b_min) - 1.0;

    double da_norm = 2.0 / (config.a_max - config.a_min);
    double db_norm = 2.0 / (config.b_max - config.b_min);

    // dPit/da
    double dPit_da = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double dT_i = chebyshev_T_deriv(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            double T_j = chebyshev_T(j, b_norm);
            dPit_da += config.pit_coeffs[i][j] * T_j * dT_i * da_norm;
        }
    }

    // dPit/db
    double dPit_db = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double T_i = chebyshev_T(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            double dT_j = chebyshev_T_deriv(j, b_norm);
            dPit_db += config.pit_coeffs[i][j] * T_i * dT_j * db_norm;
        }
    }

    // dRol/da
    double dRol_da = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double dT_i = chebyshev_T_deriv(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            double T_j = chebyshev_T(j, b_norm);
            dRol_da += config.rol_coeffs[i][j] * T_j * dT_i * da_norm;
        }
    }

    // dRol/db
    double dRol_db = 0.0;
    for (int i = 0; i <= config.a_order; ++i) {
        double T_i = chebyshev_T(i, a_norm);
        for (int j = 0; j <= config.b_order; ++j) {
            double dT_j = chebyshev_T_deriv(j, b_norm);
            dRol_db += config.rol_coeffs[i][j] * T_i * dT_j * db_norm;
        }
    }

    // 填充雅可比矩阵 [J11, J12, J21, J22]
    jacob[0] = dPit_da;  // dPit/da
    jacob[1] = dPit_db;  // dPit/db
    jacob[2] = dRol_da;  // dRol/da
    jacob[3] = dRol_db;  // dRol/db
}

void IK_V_JB(double Pit, double Rol, double jacob[4]) {
    // 使用中心差分法计算数值微分
    const double h = 1e-8;

    // 计算Pit方向的偏导数
    double a_Pit_h, b_Pit_h;
    double a_Pit_mh, b_Pit_mh;
    IK(Pit + h, Rol, a_Pit_h, b_Pit_h);
    IK(Pit - h, Rol, a_Pit_mh, b_Pit_mh);

    // 计算Rol方向的偏导数
    double a_Rol_h, b_Rol_h;
    double a_Rol_mh, b_Rol_mh;
    IK(Pit, Rol + h, a_Rol_h, b_Rol_h);
    IK(Pit, Rol - h, a_Rol_mh, b_Rol_mh);

    // 中心差分计算雅可比元素
    double da_dPit = (a_Pit_h - a_Pit_mh) / (2 * h);
    double db_dPit = (b_Pit_h - b_Pit_mh) / (2 * h);
    double da_dRol = (a_Rol_h - a_Rol_mh) / (2 * h);
    double db_dRol = (b_Rol_h - b_Rol_mh) / (2 * h);

    // 填充雅可比矩阵 [J11, J12, J21, J22]
    jacob[0] = da_dPit;  // da/dPit
    jacob[1] = da_dRol;  // da/dRol
    jacob[2] = db_dPit;  // db/dPit
    jacob[3] = db_dRol;  // db/dRol
}

} // namespace parallel_ankle
