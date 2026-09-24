#include "IK.hpp"
#include <cmath>
#include <iostream>


namespace parallel_ankle {

// 机构几何参数
constexpr double U = 40;
constexpr double V = 15.5;
constexpr double W = 9;

constexpr double UA = 0;
constexpr double VA = 15.5;

constexpr double L_LONG = 155.7;
constexpr double L_SHORT = 96.7;
constexpr double H_LONG = 146.58;
constexpr double H_SHORT = 87.58;
constexpr double R = 41;

constexpr double MOTOR_PI_OFFSET = 0;

void IK(double Pit, double Rol, double& a, double& b) {
    double cos_Pit = std::cos(Pit);
    double sin_Pit = std::sin(Pit);
    double cos_Rol = std::cos(Rol);
    double sin_Rol = std::sin(Rol);

    // 计算左右末端点位置
    double TEMP_L = -V * sin_Rol - W * cos_Rol;
    double TEMP_R = V * sin_Rol - W * cos_Rol;

    double X_left = U * cos_Pit + sin_Pit * TEMP_L;
    double Y_left = -V * cos_Rol + W * sin_Rol;
    double Z_left = -U * sin_Pit + cos_Pit * TEMP_L;

    double X_right = U * cos_Pit + sin_Pit * TEMP_R;
    double Y_right = V * cos_Rol + W * sin_Rol;
    double Z_right = -U * sin_Pit + cos_Pit * TEMP_R;

    // 构建向量
    double LP[3] = {X_left + UA, Y_left + VA, Z_left - H_LONG};
    double RP[3] = {X_right + UA, Y_right - VA, Z_right - H_SHORT};

    // 计算长度
    double LLP = std::sqrt(LP[0] * LP[0] + LP[2] * LP[2]);
    double RRP = std::sqrt(RP[0] * RP[0] + RP[2] * RP[2]);

    // 计算中间变量
    double LT = (LP[0] * LP[0] + LP[1] * LP[1] + LP[2] * LP[2] + R * R - L_LONG * L_LONG) / (2 * R);
    double RT = (RP[0] * RP[0] + RP[1] * RP[1] + RP[2] * RP[2] + R * R - L_SHORT * L_SHORT) / (2 * R);

    // 计算关节角
    double arg_a = LT / LLP;
    double arg_b = RT / RRP;

    // 限制参数在有效范围内
    arg_a = std::max(-1.0, std::min(1.0, arg_a));
    arg_b = std::max(-1.0, std::min(1.0, arg_b));

    a = -std::acos(arg_a) - std::atan2(LP[2], LP[0]) + MOTOR_PI_OFFSET;
    b = -std::acos(arg_b) - std::atan2(RP[2], RP[0]) + MOTOR_PI_OFFSET;
}

} // namespace parallel_ankle
