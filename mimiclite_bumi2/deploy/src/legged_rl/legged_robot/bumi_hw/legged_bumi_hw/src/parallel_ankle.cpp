#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>

#include "Types.h"

#include <FK.hpp>
#include <IK.hpp>
#include <Jacobian.hpp>
#include <parallel_solve/parallel_ankle.hpp>

namespace parallel_ankle {

static std::ofstream& get_cmd_debug_log() {
    static std::ofstream f;
    if (!f.is_open()) {
        std::filesystem::path log_dir = "/home/noetix/gsy/legged_bumi_v2_lab_new_0722/legged_bumi_v2_lab_new/src/legged_rl/legged_robot/bumi_hw/legged_bumi_hw/data_logs/bumi/debug_logs";
        if (!std::filesystem::exists(log_dir)) std::filesystem::create_directories(log_dir);
        time_t t = time(0);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
        std::string path = (log_dir / ("cmd_debug_" + std::string(buf) + ".csv")).string();
        f.open(path, std::ios::trunc);
        f << "loop,name,roll_sign,roll_pos,roll_des,kp,tau_roll,math_roll,math_tau_roll,IK_q1,IK_q2,J00,J01,J10,J11,motor_ff1,motor_ff2" << std::endl;
    }
    return f;
}

ParallelAnkle::ParallelAnkle()
    : ankle_motor_ready_flag_(true) {
    // 加载 FK/FK_V_JB 使用的 Chebyshev 拟合系数。
    load_coeffs(chebyshev_config_);

    // 当前只启用脖子这一组二自由度并联机构。
    //
    // 字段顺序必须和 ParallelPairConfig 完全一致：
    // name,
    // motor_1_idx, motor_2_idx,
    // pitch_idx, roll_idx,
    // pitch_sign, roll_sign,
    // motor_offset,
    // safety_limit。
    //
    // 这里把“真实电机索引”和“任务空间 pitch/roll 索引”分开写，
    // 目的就是避免以前 roll/pitch 互相写反的问题。
    pairs_ = {
    {
        "left_ankle",
        8,
        9,
        8,
        9,
        1,
        -1,
        -0.0027925268,
        2.0
    },
    {
        "right_ankle",
        18,
        19,
        18,
        19,
        1,
        1,
        -0.0027925268,
        2.0
    }
    };

}

/*
如果是腿，就是两组：
pairs_ = {
    {
        "left_ankle",
        8,
        9,
        8,
        9,
        -0.0027925268,
        1,
        1
    },
    {
        "right_ankle",
        18,
        19,
        right_pitch_idx,
        right_roll_idx,
        -0.0027925268,
        1,
        1
    }
};
*/

void ParallelAnkle::parallel_ankle_solve_state(std::vector<BumiMotorData>& joint_data) {
    // 每次状态求解都重新评估安全状态。
    // 只要任意一组 pair 不安全，helper 会把该 flag 置为 false。
    ankle_motor_ready_flag_ = true;

    for (const auto& pair : pairs_) {
        solve_state_pair(pair, joint_data);
    }
}

void ParallelAnkle::solve_state_pair(const ParallelPairConfig& pair,
                                     std::vector<BumiMotorData>& joint_data) {
    if (!is_pair_safe(pair, joint_data)) {
        ankle_motor_ready_flag_ = false;
        return;
    }

    // 从真实电机空间读取状态。
    // motor_offset 是数学模型零位和硬件反馈零位之间的偏置。
    const double q_1 = joint_data[pair.motor_1_idx].pos_ + pair.motor_offset;
    const double q_2 = joint_data[pair.motor_2_idx].pos_ + pair.motor_offset;
    const double dq_1 = joint_data[pair.motor_1_idx].vel_;
    const double dq_2 = joint_data[pair.motor_2_idx].vel_;
    const double tau_1 = joint_data[pair.motor_1_idx].tau_;
    const double tau_2 = joint_data[pair.motor_2_idx].tau_;

    double q_pitch = 0.0;
    double q_roll = 0.0;
    double jacob_fk[4] = {0.0, 0.0, 0.0, 0.0};
    double jacob_ik[4] = {0.0, 0.0, 0.0, 0.0};

    // 正运动学：真实电机角 q1/q2 -> 数学模型里的 pitch/roll。
    FK(q_1, q_2, q_pitch, q_roll, chebyshev_config_);

    // 速度雅可比：d(pitch, roll) / d(q1, q2)。
    FK_V_JB(q_1, q_2, q_pitch, q_roll, jacob_fk, chebyshev_config_);

    // 逆雅可比：d(q1, q2) / d(pitch, roll)。
    // 力矩从电机空间映射到任务空间时使用 J_ik^T。
    IK_V_JB(q_pitch, q_roll, jacob_ik);

    const double dq_pitch = jacob_fk[0] * dq_1 + jacob_fk[1] * dq_2;
    const double dq_roll = jacob_fk[2] * dq_1 + jacob_fk[3] * dq_2;

    const double tau_pitch = jacob_ik[0] * tau_1 + jacob_ik[2] * tau_2;
    const double tau_roll = jacob_ik[1] * tau_1 + jacob_ik[3] * tau_2;

    // 写回任务空间。
    // pitch 永远写 pitch_idx，roll 永远写 roll_idx；
    // 方向差异只通过 sign 处理，不再靠交换变量名硬凑。
    joint_data[pair.pitch_idx].pos_ = pair.pitch_sign * q_pitch;
    joint_data[pair.roll_idx].pos_ = pair.roll_sign * q_roll;

    joint_data[pair.pitch_idx].vel_ = pair.pitch_sign * dq_pitch;
    joint_data[pair.roll_idx].vel_ = pair.roll_sign * dq_roll;

    joint_data[pair.pitch_idx].tau_ = pair.pitch_sign * tau_pitch;
    joint_data[pair.roll_idx].tau_ = pair.roll_sign * tau_roll;
}

void ParallelAnkle::solve_cmd_pair(const ParallelPairConfig& pair,
                                   std::vector<BumiMotorData>& joint_data) {
    // 安全检查已在 solve_state_pair 中基于电机原始位置完成。
    // 此处不再重复检查，因为 motor_idx 和 pitch_idx 共用同一索引，
    // motor 空间的 pos_ 已被任务空间的 pitch/roll 覆盖，
    // 再次检查会使用错误的数据。

    // 从任务空间读取当前状态，并转回数学模型的 pitch/roll 坐标。
    // state 写回时用了 public = sign * math；
    // 这里反过来 math = sign * public，因为 sign 只取 +1/-1。
    const double q_pitch = pair.pitch_sign * joint_data[pair.pitch_idx].pos_;
    const double q_roll = pair.roll_sign * joint_data[pair.roll_idx].pos_;

    // 当前底层主要使用前馈力矩 ff_，所以这里先保留原逻辑：
    // 用任务空间的 kp/kd/pos_des/vel_des 算出 pitch/roll 力矩，
    // 再通过雅可比转成两个真实电机的前馈力矩。
    const double tau_pitch =
        joint_data[pair.pitch_idx].kp_ *
            (joint_data[pair.pitch_idx].pos_des_ - joint_data[pair.pitch_idx].pos_) +
        joint_data[pair.pitch_idx].kd_ *
            (joint_data[pair.pitch_idx].vel_des_ - joint_data[pair.pitch_idx].vel_);

    const double tau_roll =
        joint_data[pair.roll_idx].kp_ *
            (joint_data[pair.roll_idx].pos_des_ - joint_data[pair.roll_idx].pos_) +
        joint_data[pair.roll_idx].kd_ *
            (joint_data[pair.roll_idx].vel_des_ - joint_data[pair.roll_idx].vel_);

    // 力矩也需要转到数学模型坐标系下，再交给雅可比做映射。
    const double math_tau_pitch = pair.pitch_sign * tau_pitch;
    // const double math_tau_roll = pair.roll_sign * tau_roll; // cmd 与 state 需要单独控制符号
    const double math_tau_roll = pair.roll_sign * tau_roll;
    // const double math_tau_pitch = 0;

    double q_1 = 0.0;
    double q_2 = 0.0;
    double jacob_fk[4] = {0.0, 0.0, 0.0, 0.0};

    // 逆运动学：数学模型 pitch/roll -> 数学模型 q1/q2。
    IK(q_pitch, q_roll, q_1, q_2);

    // 力矩逆映射使用 J_fk^T：
    // [tau_1, tau_2]^T = J_fk^T * [tau_pitch, tau_roll]^T。
    FK_V_JB(q_1, q_2, q_pitch, q_roll, jacob_fk, chebyshev_config_);

    const double tau_1 = jacob_fk[0] * math_tau_pitch + jacob_fk[2] * math_tau_roll;
    const double tau_2 = jacob_fk[1] * math_tau_pitch + jacob_fk[3] * math_tau_roll;

    // ========== DEBUG: cmd 侧完整计算链（终端 + 文件） ==========
    static int debug_print_counter = 0;
    if (debug_print_counter % 50 == 0) {
        auto& log = get_cmd_debug_log();
        log << debug_print_counter << ","
            << pair.name << ","
            << pair.roll_sign << ","
            << std::fixed << std::setprecision(6)
            << joint_data[pair.roll_idx].pos_ << ","
            << joint_data[pair.roll_idx].pos_des_ << ","
            << joint_data[pair.roll_idx].kp_ << ","
            << tau_roll << ","
            << q_roll << ","
            << math_tau_roll << ","
            << q_1 << ","
            << q_2 << ","
            << jacob_fk[0] << "," << jacob_fk[1] << ","
            << jacob_fk[2] << "," << jacob_fk[3] << ","
            << tau_1 << "," << tau_2
            << std::endl;

        if (debug_print_counter % 500 == 0) {
            std::cout << "[CMD_DEBUG] " << pair.name
                      << " | task_roll(pos=" << joint_data[pair.roll_idx].pos_
                      << ", des=" << joint_data[pair.roll_idx].pos_des_
                      << ", kp=" << joint_data[pair.roll_idx].kp_
                      << ", tau_roll=" << tau_roll
                      << ") | math_roll=" << q_roll
                      << ", math_tau_roll=" << math_tau_roll
                      << " | IK: q1=" << q_1 << ", q2=" << q_2
                      << " | J_fk=[" << jacob_fk[0] << "," << jacob_fk[1]
                      << ";" << jacob_fk[2] << "," << jacob_fk[3] << "]"
                      << " | motor_ff: tau1=" << tau_1 << ", tau2=" << tau_2
                      << std::endl;
            log.flush();
        }
    }
    debug_print_counter++;
    // =====================================================

    joint_data[pair.motor_1_idx].ff_ = tau_1;
    joint_data[pair.motor_2_idx].ff_ = tau_2;
}

void ParallelAnkle::parallel_ankle_solve_cmd(std::vector<BumiMotorData>& joint_data) {
    // 命令侧也重新检查安全状态。
    // 这样即使某一帧没有先调用 solve_state，也不会沿用旧的 ready flag。
    ankle_motor_ready_flag_ = true;

    for (const auto& pair : pairs_) {
        solve_cmd_pair(pair, joint_data);
    }
    if (!ankle_motor_ready_flag_) {
        for (const auto& pair : pairs_) {
            joint_data[pair.motor_1_idx].ff_ = 0.0;
            joint_data[pair.motor_2_idx].ff_ = 0.0;
        }
    }
}

bool ParallelAnkle::is_pair_safe(const ParallelPairConfig& pair,
                                 const std::vector<BumiMotorData>& joint_data) const {
    // 安全检查看真实电机位置，而不是任务空间 pitch/roll。
    return std::fabs(joint_data[pair.motor_1_idx].pos_) <= pair.safety_limit &&
           std::fabs(joint_data[pair.motor_2_idx].pos_) <= pair.safety_limit;
}

} // namespace parallel_ankle
