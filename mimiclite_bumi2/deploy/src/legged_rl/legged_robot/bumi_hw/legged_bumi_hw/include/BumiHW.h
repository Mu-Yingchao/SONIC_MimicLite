
//
// Created by hang on 5/10/24.
//

#pragma once

//legged_base
#include <legged_hw/LeggedHW.h>

// standard include
#include <iostream>
#include <fstream>
#include <ostream>
#include <memory.h>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <time.h>
#include <iomanip>
// msgs
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include "std_msgs/Float64MultiArray.h"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <controller_manager_msgs/SwitchController.h>
// tf
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>

// gaoqing_hw.h
#include <cstdio>
// #include "Console.hpp"
// #include "command.h"
// #include "transmit.h"
#include "Types.h"

// bumi Host Computer Program include
#include "utilities.h"
#include "RemoteUserParameter.h"
#include "orientation_tools.h"

#include "Timer.h"
// #include "MotorConfig.h"
//#include "parallel_solve/include_parallel_solve.h"
#include "parallel_ankle.hpp"
//#include <parallel_solve/parallel_ankle.hpp>
#include "log_recorder_hw.h"

#include <future>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <functional>

#include "config_parse.h"
#include "hw_controller.h"
#include "imu_driver.h"


namespace legged
{

  
  const std::vector<std::string> CONTACT_SENSOR_NAMES = {"RF_FOOT", "LF_FOOT", "RH_FOOT", "LH_FOOT"};

  struct BumiImuData
  {
    double ori[4];
    double ori_cov[9];
    double angular_vel[3];
    double angular_vel_cov[9];
    double linear_acc[3];
    double linear_acc_cov[9];
  };

  class BumiHW : public LeggedHW
  {
  public:
    BumiHW()
    {
    }
    ~BumiHW()
    {
      std::cout << "~BumiHW_END" << std::endl;
    }

    /** \brief Get necessary params from param server. Init hardware_interface.
     *
     * Get params from param server and check whether these params are set. Load urdf of robot. Set up transmission and
     * joint limit. Get configuration of can bus and create data pointer which point to data received from Can bus.
     *
     * @param root_nh Root node-handle of a ROS node.
     * @param robot_hw_nh Node-handle for robot hardware.
     * @return True when init successful, False when failed.
     */
    bool init(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh) override;

    /** \brief Communicate with hardware. Get data, status of robot.
     *
     * Call @ref Gsmp_LEGGED_SDK::UDP::Recv() to get robot's state.
     *
     * @param time Current time
     * @param period Current time - last time
     */
    void read(const ros::Time &time, const ros::Duration &period) override;

    /** \brief Comunicate with hardware. Publish command to robot.
     *
     * Propagate joint state to actuator state for the stored
     * transmission. Limit cmd_effort into suitable value. Call @ref Gsmp_LEGGED_SDK::UDP::Recv(). Publish actuator
     * current state.
     *
     * @param time Current time
     * @param period Current time - last time
     */
    void write(const ros::Time &time, const ros::Duration &period) override;

    // void parallel_ankle_solve_state(std::vector<BumiMotorData> &joint_data);
    // void parallel_ankle_solve_cmd(std::vector<BumiMotorData> &joint_data);

    

  private:

    bool setupJoints();

    bool setupImu();

    bool setupContactSensor(ros::NodeHandle &nh);

    void CalibrateImu(const bool is_sim);

    
    std::vector<BumiMotorData> joint_data_;

    BumiImuData imu_data_{};
    bool contact_state_[4]{};
    int contact_threshold_{};
    
    uint64_t hs = 0;
  
    //ImuRc* imu_rc;
    vector_t motor_pos_feedback_;
    vector_t motor_vel_feedback_;
    vector_t motor_tau_feedback_;
    vector_t joint_planned_torque_;
    ros::Subscriber odom_sub_;
    ros::Publisher motorPosPublisher_;
    ros::Publisher motorVelPublisher_;
    ros::Publisher motorTorquePublisher_;
    OrientationTools ori_tools_;

    tf::TransformListener *listener_;

    const std::vector<int> direction_motor{1, 1, -1, -1, -1, 1,
                                          1, 1, -1, -1, 1, -1,
                                          -1, -1, 1, -1,
                                           1, -1, 1, 1,
                                           1};

    float bias_motor[21] = { 0, 0, 0, 0, 
                             -0.07, 0, 0, 0.07, 0, 0, 
                             0, 0, 0, 0,
                             -0.07, 0, 0, 0.07, 0, 0, 
                             0}; //T-pose

    bool ankle_motor_ready_flag_{false};

    bool real_log_{true};
    std::unique_ptr<LogRecorderHw> logRecorderHw_;
    std::string log_path_;
    int actuatedDofNum_{0};

    // 调试日志：同时记录电机空间 + 任务空间（踝关节，区分左右腿）
    std::ofstream debug_ankle_log_file_;
    std::string debug_ankle_log_path_;
    uint64_t debug_log_count_{0};
    struct MotorSpaceState {
      double m1_pos, m1_vel, m1_tau;
      double m2_pos, m2_vel, m2_tau;
    };
    MotorSpaceState saved_motor_state_left_;
    MotorSpaceState saved_motor_state_right_;
    // log_hw.csv 的表头名。顺序必须和 config/hardware.yaml 的 actuator_list
    // 完全一致——数据是 LogRecorderHw::WriteJointState*() 按 joint_data_[i]
    // 下标顺序写的, 而 joint_data_ 就是按 actuator_list 建立的。
    //
    // 2026-09-09 修正: 原来这里是"腿在前"的顺序(l_leg/.../r_leg/l_arm/...),
    // 而 actuator_list 是"手臂在前"(l_arm/l_elbow/l_leg/l_ankle/r_arm/...),
    // 导致 log_hw.csv 每一列的表头名和数据都对不上。
    //
    // 注意: logRecorderHw_ 在 BumiHW::init() 第 56 行构造, 早于第 74 行加载
    // actuator_list_cfg_, 所以这里没法改成运行时从 cfg 生成; 改 hardware.yaml
    // 的 actuator_list 顺序时, 必须同步改这里。
    std::vector<std::string> jointNamesHw_{
      "l_arm_pitch_joint",   "l_arm_roll_joint",   "l_arm_yaw_joint",  "l_elbow_pitch_joint",
      "l_leg_pitch_joint",   "l_leg_roll_joint",   "l_leg_yaw_joint",  "l_knee_pitch_joint",
      "l_ankle_pitch_joint", "l_ankle_roll_joint",
      "r_arm_pitch_joint",   "r_arm_roll_joint",   "r_arm_yaw_joint",  "r_elbow_pitch_joint",
      "r_leg_pitch_joint",   "r_leg_roll_joint",   "r_leg_yaw_joint",  "r_knee_pitch_joint",
      "r_ankle_pitch_joint", "r_ankle_roll_joint",
      "waist_yaw_joint"
    };

    YAML::EthercatConfig ecat_cfg_;
    YAML::CanuNetworkConfig canu_network_cfg_;
    YAML::ActuatorListConfig actuator_list_cfg_;
    YAML::ImuConfig imu_cfg_;
    std::string robot_model_{"bumi1"};   // 机型: bumi1 / bumi2
    legged::HwControllerPtr hw_ctrl_;
    ImuDriver  imudriver_;
    parallel_ankle::ParallelAnkle parallel_ankle_solver_;

  };
} // namespace legged
