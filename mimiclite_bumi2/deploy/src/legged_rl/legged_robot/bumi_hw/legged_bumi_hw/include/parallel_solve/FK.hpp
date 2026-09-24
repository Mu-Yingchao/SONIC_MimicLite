#ifndef FK_HPP
#define FK_HPP

#include "coeffs.hpp"

namespace parallel_ankle {

/**
 * @brief 正运动学求解（基于Chebyshev多项式拟合）
 * @param a 关节角a（弧度）
 * @param b 关节角b（弧度）
 * @param Pit 输出：俯仰角（弧度）
 * @param Rol 输出：横滚角（弧度）
 * @param config Chebyshev拟合配置（包含系数矩阵）
 */
void FK(double a, double b, double& Pit, double& Rol, const ChebyshevConfig& config);

} // namespace parallel_ankle

#endif // FK_HPP
