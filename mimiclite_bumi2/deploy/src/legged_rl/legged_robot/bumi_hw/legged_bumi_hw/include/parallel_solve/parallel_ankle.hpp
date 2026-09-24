#ifndef PARALLEL_ANKLE_HPP
#define PARALLEL_ANKLE_HPP

#include <vector>
#include "coeffs.hpp"
#include "Types.h"

namespace parallel_ankle {

/**
 * @brief 独立的电机数据结构。
 *
 * 当前 ParallelAnkle 的公开接口实际使用的是 BumiMotorData。
 * 这个结构体暂时保留，是为了兼容早期 parallel_solve 里独立测试/移植时的写法。
 * 如果后面确认没有任何地方使用 MotorData，可以再单独删除，避免一次重构动太多。
 */
struct MotorData {
    double pos_;        // 当前位置（弧度）
    double vel_;        // 当前速度（弧度/秒）
    double tau_;        // 当前力矩（Nm）
    double pos_des_;    // 期望位置（弧度）
    double vel_des_;    // 期望速度（弧度/秒）
    double kp_;         // PD控制器比例系数
    double kd_;         // PD控制器微分系数
    double ff_;         // 前馈力矩（Nm）
};

/**
 * @brief 一组二自由度并联机构的索引和符号配置。
 *
 * 这里的“一组”可以是：
 * - 当前正在用的脖子 pitch/roll 并联机构；
 * - 后面重新接回去的左脚踝；
 * - 后面重新接回去的右脚踝。
 *
 * 这样做的目的，是把原来散落在代码里的魔法数字集中起来：
 * - 不在算法里硬写 joint index；
 * - 不在算法里硬写 roll/pitch 的交换关系；
 * - 不在算法里硬写左右腿或者脖子的符号差异。
 *
 * 注意：这个结构体目前使用聚合初始化，字段顺序非常重要。
 * 初始化时推荐按下面顺序写，并且每一行加注释，防止把 offset/sign/limit 填串。
 */
struct ParallelPairConfig {
    // 仅用于日志和调试，例如 "neck"、"left_ankle"、"right_ankle"。
    const char* name;

    // 并联机构的两个真实电机在 joint_data 里的索引。
    // 数学库里的 IK/FK 使用 q1、q2，这里对应 motor_1_idx、motor_2_idx。
    int motor_1_idx;
    int motor_2_idx;

    // 对外暴露的任务空间关节索引。
    // pitch_idx 表示 pitch 这个任务轴写回/读取 joint_data 的位置。
    // roll_idx 表示 roll 这个任务轴写回/读取 joint_data 的位置。
    int pitch_idx;
    int roll_idx;

    // 任务空间符号修正。
    // 用于处理机械镜像、坐标系定义差异，或者像脖子这种复用脚踝算法但方向不同的情况。
    // 例如 roll_sign = -1.0 表示算法算出的 roll 写回系统时需要取反。
    double pitch_sign;
    double roll_sign;

    // 电机零位偏置。
    // 从 joint_data 读入电机角度进入 FK 前，需要加上该偏置；
    // 从 IK 算出电机角度写回命令时，需要按同一套定义减回去。
    double motor_offset;

    // 单组机构的安全位置阈值。
    // 检查的是 motor_1_idx/motor_2_idx 对应的真实电机位置，而不是 pitch/roll 任务空间位置。
    double safety_limit;

};

/**
 * @brief 并联机构求解适配层。
 *
 * 这个类本身不实现具体的并联机构数学模型，真正的 FK/IK/Jacobian 来自 parallel_solve 数学库。
 * 它负责把 BumiHW 的 joint_data 映射到数学库需要的 q1/q2、pitch/roll，再把结果写回。
 *
 * 公开接口保持原来的两个函数名，是为了尽量少改 BumiHW 调用侧：
 * - parallel_ankle_solve_state：把电机反馈状态转换成 pitch/roll 状态；
 * - parallel_ankle_solve_cmd：把 pitch/roll 命令转换成电机命令/前馈。
 *
 * 内部通过 pairs_ 支持多组机构。
 * 当前可以只放一组 neck；后面要恢复双腿时，只需要再追加 left_ankle/right_ankle 配置。
 */
class ParallelAnkle {
public:
    ParallelAnkle();

    // 状态方向：真实电机空间 -> 任务空间。
    // 典型流程：
    // joint_data[motor_1/motor_2].pos_/vel_/tau_ -> FK/Jacobian ->
    // joint_data[pitch/roll].pos_/vel_/tau_。
    void parallel_ankle_solve_state(std::vector<BumiMotorData>& joint_data);

    // 命令方向：任务空间 -> 真实电机空间。
    // 典型流程：
    // joint_data[pitch/roll].pos_des_/vel_des_/kp_/kd_ -> IK/Jacobian ->
    // joint_data[motor_1/motor_2].pos_des_/vel_des_/ff_。
    void parallel_ankle_solve_cmd(std::vector<BumiMotorData>& joint_data);

    // 只要任意一组机构触发安全检查，这个 flag 就应该变成 false。
    bool is_motor_ready() const { return ankle_motor_ready_flag_; }

private:
    // 处理单组机构的状态正解。
    // 所有索引、符号、offset 都必须从 pair 读取，避免再出现硬编码的 roll/pitch 搞反问题。
    void solve_state_pair(const ParallelPairConfig& pair,
                          std::vector<BumiMotorData>& joint_data);

    // 处理单组机构的命令逆解。
    // 和 solve_state_pair 使用同一个 pair 配置，保证状态方向和命令方向的映射是互逆的。
    void solve_cmd_pair(const ParallelPairConfig& pair,
                        std::vector<BumiMotorData>& joint_data);

    // 检查单组机构是否处于安全范围内。
    // 注意这里应该检查真实电机索引 motor_1_idx/motor_2_idx，而不是 pitch_idx/roll_idx。
    bool is_pair_safe(const ParallelPairConfig& pair,
                      const std::vector<BumiMotorData>& joint_data) const;

    // 当前启用的所有并联机构配置。
    // 现在做脖子时只需要一个元素；以后恢复双腿时加配置，不需要改求解主流程。
    std::vector<ParallelPairConfig> pairs_;

    // 并联机构整体安全状态。
    // true 表示所有 pair 当前都通过安全检查；false 表示至少一组不安全。
    bool ankle_motor_ready_flag_;

    // Chebyshev 拟合系数配置，供 FK/FK_V_JB 使用。
    // 构造函数里通过 load_coeffs 初始化。
    ChebyshevConfig chebyshev_config_;
};

} // namespace parallel_ankle

#endif // PARALLEL_ANKLE_HPP
