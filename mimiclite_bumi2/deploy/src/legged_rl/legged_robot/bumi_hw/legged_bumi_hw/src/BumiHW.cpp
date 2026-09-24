#include "BumiHW.h"
#include <ros/package.h>

double global_time = 0.0;
// Medulla* node_1;
// Medulla* node_2;
// Medulla* node_3;
// Medulla* node_4;
// ImuRc* imu_rc;

namespace legged
{

  bool BumiHW::init(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh)
  { 
    //发布话题
    motorPosPublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_pos", 1);
    motorVelPublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_vel", 1);
    motorTorquePublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_torque", 1);
    motor_pos_feedback_.setZero();
    motor_vel_feedback_.setZero();
    motor_tau_feedback_.setZero();
    joint_planned_torque_.setZero();

    std::string user_param_path;
    std::string hardware_param_path;
  
    if (!root_nh.getParam("/user_param_path", user_param_path)) {
        user_param_path = "config/config.yaml";
        ROS_WARN("Parameter user_param_path not set. Using default: %s", user_param_path.c_str());
    }

    if (!root_nh.getParam("/hardware_param_path", hardware_param_path)) {
        hardware_param_path = "config/hardware.yaml";
        ROS_WARN("Parameter hardware_param_path not set. Using default: %s", hardware_param_path.c_str());
    }

    // 如果是相对路径，则解析为绝对路径（相对于legged_bumi_hw包目录）
    std::string pkg_path = ros::package::getPath("legged_bumi_hw");
    if (!user_param_path.empty() && user_param_path[0] != '/') {
        user_param_path = pkg_path + "/" + user_param_path;
    }
    if (!hardware_param_path.empty() && hardware_param_path[0] != '/') {
        hardware_param_path = pkg_path + "/" + hardware_param_path;
    }

    
   
    // log_path_ 独立于 real_log_，确保调试日志始终能创建
    if (!root_nh.getParam("/logFileHw", log_path_) || log_path_.empty()) {
      log_path_ = "/home/noetix/gsy/legged_bumi_v2_lab_new_0722/legged_bumi_v2_lab_new/src/legged_rl/legged_robot/bumi_hw/legged_bumi_hw/data_logs/bumi/log_hw.csv";
      std::cout << "[DEBUG] logFileHw not set, using default: " << log_path_ << std::endl;
    }

    if(real_log_) {
      logRecorderHw_ = std::make_unique<LogRecorderHw>(log_path_, actuatedDofNum_, jointNamesHw_);
    }

    std::cout<<"user_param_path:"<<user_param_path<<std::endl;
    // 创建用户参数对象并从YAML文件初始化
    std::shared_ptr<RemoteUserParameter> user_param = std::make_shared<RemoteUserParameter>();
    try {
        user_param->initializeFromYamlFile(user_param_path);
    } catch(std::exception& e) {
        printf("Failed to initialize robot parameters from yaml file: %s\n", e.what());
    }
   

    YAML::Node cfg_node = YAML::LoadFile(hardware_param_path.data());

    robot_model_ = cfg_node["robot_model"] ? cfg_node["robot_model"].as<std::string>("bumi1") : "bumi1";
    printf("robot_model: %s\n", robot_model_.c_str());

    actuator_list_cfg_    = cfg_node["actuator_list"].as<YAML::ActuatorListConfig>();

    ecat_cfg_ = cfg_node["ethercat"].as<YAML::EthercatConfig>();
    canu_network_cfg_ = cfg_node["canu_network"].as<YAML::CanuNetworkConfig>();
    
    // 从canu_network中查找启用IMU的CANU名称
    std::string imu_canu_name = "body";  // 默认值
    for (const auto& canu_cfg : canu_network_cfg_) {
      if (canu_cfg.imu_enable) {
        imu_canu_name = canu_cfg.name;
        break;
      }
    }
    
    // 解析IMU配置
    if (cfg_node["imu"]) {
      imu_cfg_ = cfg_node["imu"].as<YAML::ImuConfig>();
      // 如果配置文件中没有指定ethercat_canu，则使用从canu_network中查找到的名称
      if (imu_cfg_.source == YAML::ImuSource::ETHERCAT && imu_cfg_.ethercat_canu.empty()) {
        imu_cfg_.ethercat_canu = imu_canu_name;
      }
    } else {
      // 如果没有imu配置节，则默认使用ethercat模式
      imu_cfg_.source = YAML::ImuSource::ETHERCAT;
      imu_cfg_.ethercat_canu = imu_canu_name;
      imu_cfg_.usb_port = "/dev/ttyACM0";
    }

    hw_ctrl_.reset(HwController::GetInstance());


    for (const auto& canu_cfg : canu_network_cfg_) {
      if (!canu_cfg.enable) continue;
      bool ret = hw_ctrl_->CreateCanu(canu_cfg.name, canu_cfg.ecat_id, robot_model_);
      if (ret == false) { printf("Canu %s register failed.", canu_cfg.name.data());}
    
      for (size_t ch = 0; ch <= (size_t)CtrlChannel::CTRL_CH3; ch++) {
        for (const auto& actr : canu_cfg.ch[ch]) {
          ret = hw_ctrl_->AttachActuator(canu_cfg.name, (CtrlChannel)ch, StringToType(actr.type),
                                            actr.name, actr.can_id);
          if (ret == false) { printf("Actuator %s register failed.", canu_cfg.name.data());}                     
        }
      }
    }

    bool ret = hw_ctrl_->Init(ecat_cfg_.ifname, ecat_cfg_.cycle_time_ns, ecat_cfg_.enable_dc);
    if (!ret) {
      printf("XyberController start failed.");
      return false;
    }

    // 启动时清除电机错误（可通过 hardware.yaml 中 clean_actuator_error 配置）
    bool clean_error = cfg_node["clean_actuator_error"] ? cfg_node["clean_actuator_error"].as<bool>() : false;
    if (clean_error) {
      bool clean_ret = hw_ctrl_->CleanAllActuaorError();
      if (clean_ret) {
        printf("All actuator errors cleared successfully.");
      } else {
        printf("Warning: Failed to clear actuator errors.");
      }
    }

    //actuatedDofNum_ = actuator_list_cfg_.size();
    joint_data_.resize(actuator_list_cfg_.size());
    // 显式初始化joint_data_所有字段为零，
    for (auto& jd : joint_data_) {
      jd.pos_ = 0; jd.vel_ = 0; jd.tau_ = 0;
      jd.pos_des_ = 0; jd.vel_des_ = 0; jd.kp_ = 0; jd.kd_ = 0; jd.ff_ = 0;
    }
    // Hardware interface
    actuatedDofNum_ = jointNamesHw_.size();

    motor_pos_feedback_.resize(actuator_list_cfg_.size());
    motor_pos_feedback_.setZero();

    motor_vel_feedback_.resize(actuator_list_cfg_.size());
    motor_vel_feedback_.setZero();

    motor_tau_feedback_.resize(actuator_list_cfg_.size());
    motor_tau_feedback_.setZero();

    joint_planned_torque_.resize(actuator_list_cfg_.size());
    joint_planned_torque_.setZero();

    listener_ = new tf::TransformListener(root_nh, ros::Duration(5.0), true);

    if (!LeggedHW::init(root_nh, robot_hw_nh))
    {
      printf("flase_bc_leggedHW::init\n");
      return false;
    }

    // 根据配置初始化IMU数据来源
    if (imu_cfg_.source == YAML::ImuSource::USB) {
      printf("IMU source: USB (%s)\n", imu_cfg_.usb_port.c_str());
      imudriver_.init(imu_cfg_.usb_port);
    } else {
      printf("IMU source: EtherCAT (CANU: %s)\n", imu_cfg_.ethercat_canu.c_str());
    }

    setupJoints();
    
    setupImu();

    // ========== 创建踝关节调试日志（电机空间 + 任务空间，带时间戳） ==========
    {
      std::time_t t = std::time(nullptr);
      std::tm tm = *std::localtime(&t);
      std::ostringstream oss;
      oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
      std::string timestamp = oss.str();

      std::filesystem::path log_dir = std::filesystem::path(log_path_).parent_path() / "debug_logs";
      if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directories(log_dir);
      }

      debug_ankle_log_path_ = (log_dir / ("hw_ankle_motor_task_" + timestamp + ".csv")).string();
      debug_ankle_log_file_.open(debug_ankle_log_path_, std::ios::trunc);

      if (debug_ankle_log_file_.is_open()) {
        std::cout << "[DEBUG] Ankle log file created: " << debug_ankle_log_path_ << std::endl;

        // 写 CSV 表头：每条记录 = 左右腿各一组，每组包含电机空间(m1,m2)和任务空间(pitch,roll)的反馈+命令
        debug_ankle_log_file_ << "loop,time_sec,leg,"
          // 电机空间 Motor Space (反馈)
          << "m1_pos,m1_vel,m1_tau,"
          << "m2_pos,m2_vel,m2_tau,"
          // 任务空间 Task Space - pitch (反馈+命令)
          << "pitch_pos,pitch_vel,pitch_tau,"
          << "pitch_pos_des,pitch_vel_des,pitch_kp,pitch_kd,pitch_ff,pitch_torque_cmd,"
          // 任务空间 Task Space - roll (反馈+命令)
          << "roll_pos,roll_vel,roll_tau,"
          << "roll_pos_des,roll_vel_des,roll_kp,roll_kd,roll_ff,roll_torque_cmd"
          << std::endl;
      }
      debug_log_count_ = 0;
    }
    // ==========================================================================

    //setupContactSensor(robot_hw_nh);
    return true;
  }
  template <int row_>

  using Vector = Eigen::Matrix<double, row_, 1>;

  Vector<12> m_q; // motor feedback (prior conversion)
  Vector<12> m_v; // motor feedback (prior conversion)
  Vector<12> m_t; // motor feedback (prior conversion)

  void BumiHW::read(const ros::Time &time, const ros::Duration &period)
  { 
    int i = 0;
    if(real_log_) {
      logRecorderHw_->Open();
      logRecorderHw_->WriteScalar(time.toSec());
    }

    ++hs;
    hw_ctrl_->SetHeartCnt(hs);

    hw_ctrl_->rt_ethercat_get_data();
    // std::cout << "actr_cfg.name = ";
    for (auto& actr_cfg : actuator_list_cfg_) {

      joint_data_[i].tau_ = hw_ctrl_->GetEffort(actr_cfg.name) * actr_cfg.direction;
      joint_data_[i].vel_ = hw_ctrl_->GetVelocity(actr_cfg.name) * actr_cfg.direction;
      joint_data_[i].pos_ = hw_ctrl_->GetPosition(actr_cfg.name) * actr_cfg.direction + actr_cfg.bias;
      // std::cout << actr_cfg.name;
      i++; 
    }
    // std::cout << std::endl;

    
    
      /*
    元生艾欸姆尤
    */ 
    // 根据配置从不同来源获取IMU数据
    if (imu_cfg_.source == YAML::ImuSource::USB) {
      // 从USB串口IMU获取数据
      ImuRcData imu_rc_data_ = imudriver_.getimudata();
      imu_data_.ori[0] = imu_rc_data_.q1;
      imu_data_.ori[1] = imu_rc_data_.q2; 
      imu_data_.ori[2] = imu_rc_data_.q3;
      imu_data_.ori[3] = imu_rc_data_.q0;
      imu_data_.angular_vel[0] = imu_rc_data_.gyr_x;
      imu_data_.angular_vel[1] = imu_rc_data_.gyr_y;
      imu_data_.angular_vel[2] = imu_rc_data_.gyr_z;     
      imu_data_.linear_acc[0] = imu_rc_data_.acc_x;
      imu_data_.linear_acc[1] = imu_rc_data_.acc_y;
      imu_data_.linear_acc[2] = imu_rc_data_.acc_z;
    } else {
      // 从EtherCAT获取IMU数据
      legged::ImuData ethercat_imu_data;
      bool imu_ret = hw_ctrl_->GetImuData(imu_cfg_.ethercat_canu, ethercat_imu_data);
      if (imu_ret) {
        imu_data_.ori[0] = ethercat_imu_data.ori[0];
        imu_data_.ori[1] = ethercat_imu_data.ori[1];
        imu_data_.ori[2] = ethercat_imu_data.ori[2];
        imu_data_.ori[3] = ethercat_imu_data.ori[3];
        imu_data_.angular_vel[0] = ethercat_imu_data.angular_vel[0];
        imu_data_.angular_vel[1] = ethercat_imu_data.angular_vel[1];
        imu_data_.angular_vel[2] = ethercat_imu_data.angular_vel[2];
        imu_data_.linear_acc[0] = ethercat_imu_data.linear_acc[0];
        imu_data_.linear_acc[1] = ethercat_imu_data.linear_acc[1];
        imu_data_.linear_acc[2] = ethercat_imu_data.linear_acc[2];
      } else {
        // 获取IMU数据失败，打印调试信息（每1000次打印一次避免刷屏）
        static int fail_count = 0;
        if (fail_count % 1000 == 0) {
          printf("[WARN] GetImuData failed for CANU: %s (count: %d)\n", imu_cfg_.ethercat_canu.c_str(), fail_count);
        }
        fail_count++;
      }
    }


    if(real_log_) {
      logRecorderHw_->WriteJointStatePos(&joint_data_[0], actuatedDofNum_);
      logRecorderHw_->WriteJointStateVel(&joint_data_[0], actuatedDofNum_);
      logRecorderHw_->WriteJointStateTor(&joint_data_[0], actuatedDofNum_);
    }
   
    // std::cout << "joint_data[8].ff_ & tau_" << joint_data[0 * (10) +  8].ff_ << "\t\t" << joint_data[0 * (10) +  8].tau_ << std::endl;
    // std::cout << "joint_data[9].ff_ & tau_" << joint_data_[0 * (10) +  9].ff_ << "\t\t" << joint_data_[0 * (10) +  9].tau_ << std::endl;

    // ========== 调试日志：并联解算前保存电机空间原始数据 ==========
    // 左踝: motor_1_idx=8, motor_2_idx=9
    saved_motor_state_left_.m1_pos = joint_data_[8].pos_;
    saved_motor_state_left_.m1_vel = joint_data_[8].vel_;
    saved_motor_state_left_.m1_tau = joint_data_[8].tau_;
    saved_motor_state_left_.m2_pos = joint_data_[9].pos_;
    saved_motor_state_left_.m2_vel = joint_data_[9].vel_;
    saved_motor_state_left_.m2_tau = joint_data_[9].tau_;
    // 右踝: motor_1_idx=18, motor_2_idx=19
    saved_motor_state_right_.m1_pos = joint_data_[18].pos_;
    saved_motor_state_right_.m1_vel = joint_data_[18].vel_;
    saved_motor_state_right_.m1_tau = joint_data_[18].tau_;
    saved_motor_state_right_.m2_pos = joint_data_[19].pos_;
    saved_motor_state_right_.m2_vel = joint_data_[19].vel_;
    saved_motor_state_right_.m2_tau = joint_data_[19].tau_;
    // ===================================================================

    // need debug
    parallel_ankle_solver_.parallel_ankle_solve_state(joint_data_);

    // ========== 调试日志：并联解算后同时写入电机空间 + 任务空间（含命令） ==========
    if (debug_ankle_log_file_.is_open() && (debug_log_count_ % 10 == 0)) {
      double t_sec = debug_log_count_ * 0.002;  // 500Hz

      // 计算任务空间 PD 力矩（和 parallel_ankle_solve_cmd 中一致的公式）
      auto calc_torque_cmd = [](BumiMotorData& d) {
        return d.ff_ + d.kp_ * (d.pos_des_ - d.pos_) + d.kd_ * (d.vel_des_ - d.vel_);
      };

      // 写左腿 (pitch_idx=8, roll_idx=9)
      double pitch_tau_cmd_l = calc_torque_cmd(joint_data_[8]);
      double roll_tau_cmd_l = calc_torque_cmd(joint_data_[9]);
      debug_ankle_log_file_ << debug_log_count_ << ","
        << std::fixed << std::setprecision(4) << t_sec << ",left,"
        // 电机空间
        << saved_motor_state_left_.m1_pos << "," << saved_motor_state_left_.m1_vel << "," << saved_motor_state_left_.m1_tau << ","
        << saved_motor_state_left_.m2_pos << "," << saved_motor_state_left_.m2_vel << "," << saved_motor_state_left_.m2_tau << ","
        // 任务空间 pitch (反馈 + 命令)
        << joint_data_[8].pos_ << "," << joint_data_[8].vel_ << "," << joint_data_[8].tau_ << ","
        << joint_data_[8].pos_des_ << "," << joint_data_[8].vel_des_ << ","
        << joint_data_[8].kp_ << "," << joint_data_[8].kd_ << "," << joint_data_[8].ff_ << "," << pitch_tau_cmd_l << ","
        // 任务空间 roll (反馈 + 命令)
        << joint_data_[9].pos_ << "," << joint_data_[9].vel_ << "," << joint_data_[9].tau_ << ","
        << joint_data_[9].pos_des_ << "," << joint_data_[9].vel_des_ << ","
        << joint_data_[9].kp_ << "," << joint_data_[9].kd_ << "," << joint_data_[9].ff_ << "," << roll_tau_cmd_l
        << std::endl;

      // 写右腿 (pitch_idx=18, roll_idx=19)
      double pitch_tau_cmd_r = calc_torque_cmd(joint_data_[18]);
      double roll_tau_cmd_r = calc_torque_cmd(joint_data_[19]);
      debug_ankle_log_file_ << debug_log_count_ << ","
        << std::fixed << std::setprecision(4) << t_sec << ",right,"
        // 电机空间
        << saved_motor_state_right_.m1_pos << "," << saved_motor_state_right_.m1_vel << "," << saved_motor_state_right_.m1_tau << ","
        << saved_motor_state_right_.m2_pos << "," << saved_motor_state_right_.m2_vel << "," << saved_motor_state_right_.m2_tau << ","
        // 任务空间 pitch (反馈 + 命令)
        << joint_data_[18].pos_ << "," << joint_data_[18].vel_ << "," << joint_data_[18].tau_ << ","
        << joint_data_[18].pos_des_ << "," << joint_data_[18].vel_des_ << ","
        << joint_data_[18].kp_ << "," << joint_data_[18].kd_ << "," << joint_data_[18].ff_ << "," << pitch_tau_cmd_r << ","
        // 任务空间 roll (反馈 + 命令)
        << joint_data_[19].pos_ << "," << joint_data_[19].vel_ << "," << joint_data_[19].tau_ << ","
        << joint_data_[19].pos_des_ << "," << joint_data_[19].vel_des_ << ","
        << joint_data_[19].kp_ << "," << joint_data_[19].kd_ << "," << joint_data_[19].ff_ << "," << roll_tau_cmd_r
        << std::endl;
    }
    debug_log_count_++;
    // 每 100 个周期（约 0.2s）flush 一次，防止 Ctrl+C 丢数据
    if (debug_log_count_ % 100 == 0 && debug_ankle_log_file_.is_open()) {
      debug_ankle_log_file_.flush();
    }
    // ===================================================================

    for (int i = 0; i < actuator_list_cfg_.size(); ++i) {
        motor_pos_feedback_(i) = joint_data_[i].pos_;
        motor_vel_feedback_(i) = joint_data_[i].vel_;
        motor_tau_feedback_(i) = joint_data_[i].tau_;
    }

    motorTorquePublisher_.publish(createFloat64MultiArrayFromVector(motor_tau_feedback_));
    motorPosPublisher_.publish(createFloat64MultiArrayFromVector(motor_pos_feedback_));
    motorVelPublisher_.publish(createFloat64MultiArrayFromVector(motor_vel_feedback_));

    // Set feedforward and velocity cmd to zero to avoid for safety when not controller setCommand
    std::vector<std::string> names = hybridJointInterface_.getNames();
    for (const auto &name : names)
    {
      HybridJointHandle handle = hybridJointInterface_.getHandle(name);
      handle.setFeedforward(0.);
      handle.setVelocityDesired(0.);
      //handle.setKd(3.1415);
      handle.setKd(0.5);
      handle.setKp(0.);
    }
  
  }


  void BumiHW::write(const ros::Time &time, const ros::Duration &period)
  {
    int i = 0;
    double  pos_des = 0,  vel_des = 0,  ff = 0, kp = 0, kd = 0;
  

    parallel_ankle_solver_.parallel_ankle_solve_cmd(joint_data_);

    if(real_log_) {
      logRecorderHw_->WriteJointCmdPos(&joint_data_[0], actuatedDofNum_);
      logRecorderHw_->WriteJointCmdVel(&joint_data_[0], actuatedDofNum_);
      logRecorderHw_->WriteJointCmdTor(&joint_data_[0], actuatedDofNum_);

      logRecorderHw_->Close();
    }

    for (auto& actr_cfg : actuator_list_cfg_) {

      pos_des = (joint_data_[i].pos_des_ - actr_cfg.bias) * actr_cfg.direction;
      vel_des = joint_data_[i].vel_des_ * actr_cfg.direction;
      
      ff = joint_data_[i].ff_ * actr_cfg.direction;
      kp = joint_data_[i].kp_;
      kd = joint_data_[i].kd_;
      // if (actr_cfg.name == "l_ankle_pitch_joint")
      // {
      //   std::cout<< "l_ankle_pitch_joint idx : " << i << std::endl;
      // }
      // else if (actr_cfg.name == "l_ankle_roll_joint")
      // {
      //   std::cout<< "l_ankle_roll_joint idx : " << i << std::endl;
      // }else if (actr_cfg.name == "r_ankle_pitch_joint")
      // {
      //   std::cout<< "r_ankle_pitch_joint idx : " << i << std::endl;
      // }else if (actr_cfg.name == "r_ankle_roll_joint")
      // {
      //   std::cout<< "r_ankle_roll_joint idx : " << i << std::endl;
      // }
      
      
      if (actr_cfg.name == "l_ankle_pitch_joint" ||
          actr_cfg.name == "r_ankle_pitch_joint" ||
          actr_cfg.name == "l_ankle_roll_joint" ||
          actr_cfg.name == "r_ankle_roll_joint") {

        kp = 0;
        kd = 0;
        // ff = std::clamp(ff, -5.0, 5.0);
      }

    // pos_des = 0;   // 已注释：不要清零位置指令，否则 AcController 设置的目标位置无效
    // vel_des = 0;   // 已注释：不要清零速度指令
    // // ff = 0;

      hw_ctrl_->SetMitCmd(actr_cfg.name, pos_des, vel_des, ff, kp, kd);
      i++; 
    }
    
    hw_ctrl_->rt_ethercat_set_command();
    hw_ctrl_->rt_ethercat_run();
  }

  bool BumiHW::setupJoints()
  {
    int i = 0;
    for (auto& actr_cfg : actuator_list_cfg_) {
      // 创建JointStateHandle对象并注册到jointStateInterface_
      hardware_interface::JointStateHandle state_handle(actr_cfg.name, &joint_data_[i].pos_, &joint_data_[i].vel_, &joint_data_[i].tau_);
        
      jointStateInterface_.registerHandle(state_handle);

      // 创建HybridJointHandle对象并注册到hybridJointInterface_
      hybridJointInterface_.registerHandle(HybridJointHandle(state_handle, &joint_data_[i].pos_des_, &joint_data_[i].vel_des_, &joint_data_[i].kp_, &joint_data_[i].kd_, &joint_data_[i].ff_));

      i++;
    }
    return true;
  }

  bool BumiHW::setupImu()
  {
     imuSensorInterface_.registerHandle(hardware_interface::ImuSensorHandle(
        "base_imu", "base_imu", imu_data_.ori, imu_data_.ori_cov, imu_data_.angular_vel, imu_data_.angular_vel_cov,
        imu_data_.linear_acc, imu_data_.linear_acc_cov));
    imu_data_.ori_cov[0] = 0.0012;
    imu_data_.ori_cov[4] = 0.0012;
    imu_data_.ori_cov[8] = 0.0012;

    imu_data_.angular_vel_cov[0] = 0.0004;
    imu_data_.angular_vel_cov[4] = 0.0004;
    imu_data_.angular_vel_cov[8] = 0.0004;

    return true;
  }

#if 0
  void BumiHW::CalibrateImu(const bool is_sim) {
   // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
   // std::cout << "[quat] : " << robot_state->quat << std::endl;
   // std::cout << "[euler]: " << robot_state->imu_rpy_angle.transpose() << std::endl;
   // std::cout << "[gyro] : " << robot_state->imu_rpy_rate.transpose() << std::endl;
   // std::cout << "[acc]  : " << robot_state->imu_xyz_acc.transpose() << std::endl;

    // get ImuCalibrationParam
    Eigen::Quaternion<double> quat_pre;
    Eigen::Quaternion<double> quat_post;
    if (is_sim) {
        quat_pre.x() = 0;
        quat_pre.y() = 0;
        quat_pre.z() = 0;
        quat_pre.w() = 1;
        quat_post.x() = 0;
        quat_post.y() = 0;
        quat_post.z() = 0;
        quat_post.w() = 1;
    } else {
        quat_pre.x() = 0;
        quat_pre.y() = 0;
        quat_pre.z() = 0;
        quat_pre.w() = 1;
        quat_post.x() = 0;
        quat_post.y() = 0;
        quat_post.z() = 1;
        quat_post.w() = 0;
    }
    Eigen::Matrix3d mat_pre;
    Eigen::Matrix3d mat_post;
    Eigen::Matrix3d mat_output;
    mat_pre = ori_tools_.quatToRotMatrix(quat_pre);
    mat_post = ori_tools_.quatToRotMatrix(quat_post);

    // solve quat AKA orientation
    Eigen::Quaternion<double> quat_input;
    Eigen::Quaternion<double> quat_output;

    quat_input.x() = (double)imu_rc->imu_rc_data_.q1;
    quat_input.y() = (double)imu_rc->imu_rc_data_.q2;
    quat_input.z() = (double)imu_rc->imu_rc_data_.q3;
    quat_input.w() = (double)imu_rc->imu_rc_data_.q0;

    mat_output = mat_pre * ori_tools_.quatToRotMatrix(quat_input) * mat_post;

    quat_output = ori_tools_.rotationMatrixToQuaternion(mat_output);

    imu_data_.ori[0] = quat_output.x();
    imu_data_.ori[1] = quat_output.y();
    imu_data_.ori[2] = quat_output.z();
    imu_data_.ori[3] = quat_output.w();

    // rpy angle
    // robot_state->imu_rpy_angle = ori_tools_.quatToEulerAngle(quat_output);

    // solve gyro AKA angular velocity
    Eigen::Vector3d gyro_input;
    Eigen::Vector3d gyro_output;

    gyro_input << (double)imu_rc->imu_rc_data_.gyr_x,
                  (double)imu_rc->imu_rc_data_.gyr_y,
                  (double)imu_rc->imu_rc_data_.gyr_z;
      
    gyro_output = mat_post * gyro_input;

    imu_data_.angular_vel[0] = gyro_output(0);
    imu_data_.angular_vel[1] = gyro_output(1);
    imu_data_.angular_vel[2] = gyro_output(2);

    // solve acc AKA linear acceleration
    Eigen::Vector3d acc_input;
    Eigen::Vector3d acc_output;

    acc_input << (double)imu_rc->imu_rc_data_.acc_x,
                 (double)imu_rc->imu_rc_data_.acc_y,
                 (double)imu_rc->imu_rc_data_.acc_z;
    if(is_sim) {
        acc_output = mat_post * acc_input;
    } else {
        acc_output = mat_post * acc_input * 9.81;
    }

    imu_data_.linear_acc[0] = acc_output(0);
    imu_data_.linear_acc[1] = acc_output(1);
    imu_data_.linear_acc[2] = acc_output(2);
  }
#endif 
  // void BumiHW::parallel_ankle_solve_state(std::vector<BumiMotorData> &joint_data) {
  //   double q_1;
  //   double q_2;
  //   double dq_1;
  //   double dq_2;
  //   double tor_1;
  //   double tor_2;

  //   double jacob[4];

  //   double q_roll;
  //   double q_pitch;
  //   double dq_roll;
  //   double dq_pitch;
  //   double tor_roll;
  //   double tor_pitch;

  //   // std::cout << "joint_data[9 & 8].tau_" << joint_data[0 * (10) +  9].tau_ << "\t\t" << joint_data[0 * (10) +  8].tau_ << std::endl;
  //   // std::cout << "joint_data[8].ff_ & tau_" << joint_data[0 * (10) +  8].ff_ << "\t\t" << joint_data[0 * (10) +  8].tau_ << std::endl;
  //   // std::cout << "joint_data[9].ff_ & tau_" << joint_data[0 * (10) +  9].ff_ << "\t\t" << joint_data[0 * (10) +  9].tau_ << std::endl;


  //   if(fabs(joint_data[4+4].pos_) > 3.14 || fabs(joint_data[5+4].pos_) > 3.14 || fabs(joint_data[4+14].pos_) > 3.14 || fabs(joint_data[5+14].pos_) > 3.14) {
  //     ankle_motor_ready_flag_ = false;
  //   } else {
  //     ankle_motor_ready_flag_ = true;
  //   }

  //   for(int which_leg = 0; which_leg < 2; ++which_leg){
  //     q_1   = joint_data[which_leg * (10) +  9].pos_ + 0.2618;
  //     q_2   = joint_data[which_leg * (10) +  8].pos_ + 0.2618;
  //     dq_1  = joint_data[which_leg * (10) +  9].vel_;
  //     dq_2  = joint_data[which_leg * (10) +  8].vel_;
  //     tor_1 = joint_data[which_leg * (10) +  9].tau_;
  //     tor_2 = joint_data[which_leg * (10) +  8].tau_;

  //     parallel_ankle_fk(q_1, q_2, 1-which_leg, &q_roll, &q_pitch);
  //     parallel_ankle_jacobian(q_roll, q_pitch, q_1, q_2, 1-which_leg, 0, jacob);
  //     parallel_ankle_fk_vel(dq_1, dq_2, jacob, 1-which_leg, &dq_roll, &dq_pitch);
  //     parallel_ankle_fd(tor_1, tor_2, jacob, 1-which_leg, &tor_roll, &tor_pitch);

  //     joint_data[which_leg * (10) +  9].pos_ = q_roll;
  //     joint_data[which_leg * (10) +  8].pos_ = q_pitch;
  //     joint_data[which_leg * (10) +  9].vel_ = dq_roll;
  //     joint_data[which_leg * (10) +  8].vel_ = dq_pitch;
  //     joint_data[which_leg * (10) +  9].tau_ = tor_roll;
  //     joint_data[which_leg * (10) +  8].tau_ = tor_pitch;
  //   }
  // }

  // void BumiHW::parallel_ankle_solve_cmd(std::vector<BumiMotorData> &joint_data){
  //   double q_1;
  //   double q_2;
  //   double dq_1;
  //   double dq_2;
  //   double tor_1;
  //   double tor_2;

  //   double jacob[4];

  //   double q_roll;
  //   double q_pitch;
  //   double dq_roll;
  //   double dq_pitch;
  //   double tor_roll;
  //   double tor_pitch;

  //   for(int which_leg = 0; which_leg < 2; ++which_leg){
  //     q_roll    = joint_data[which_leg * (10) +  9].pos_;
  //     q_pitch   = joint_data[which_leg * (10) +  8].pos_;
  //     dq_roll   = joint_data[which_leg * (10) +  9].vel_;
  //     dq_pitch  = joint_data[which_leg * (10) +  8].vel_;
  //     tor_roll  = joint_data[which_leg * (10) +  9].kp_ * (joint_data[which_leg * (10) +  9].pos_des_ - joint_data[which_leg * (10) +  9].pos_) + joint_data[which_leg * (10) +  9].kd_ * (joint_data[which_leg * (10) +  9].vel_des_ - joint_data[which_leg * (10) +  9].vel_);
  //     tor_pitch = joint_data[which_leg * (10) +  8].kp_ * (joint_data[which_leg * (10) +  8].pos_des_ - joint_data[which_leg * (10) +  8].pos_) + joint_data[which_leg * (10) +  8].kd_ * (joint_data[which_leg * (10) +  8].vel_des_ - joint_data[which_leg * (10) +  8].vel_);

  //     // q_roll    = joint_data[which_leg * (10) +  9].pos_des_;
  //     // q_pitch   = joint_data[which_leg * (10) +  8].pos_des_;
  //     // dq_roll   = joint_data[which_leg * (10) +  9].vel_des_;
  //     // dq_pitch  = joint_data[which_leg * (10) +  8].vel_des_;
  //     // tor_roll  = joint_data[which_leg * (10) +  9].ff_;
  //     // tor_pitch = joint_data[which_leg * (10) +  8].ff_;

  //     // q_roll = std::min(0.4, std::max(-0.4, q_roll));
  //     // q_pitch = std::min(0.4, std::max(-0.4, q_pitch));
      
  //     parallel_ankle_ik(q_roll, q_pitch, 1-which_leg, 0, &q_1, &q_2);
  //     parallel_ankle_jacobian(q_roll, q_pitch, q_1, q_2, 1-which_leg, 0, jacob);
  //     parallel_ankle_ik_vel(q_roll, q_pitch, q_1, q_2, dq_roll, dq_pitch, 1-which_leg, 0, &dq_1, &dq_2);
  //     parallel_ankle_id(tor_roll, tor_pitch, jacob, 1-which_leg, &tor_1, &tor_2);

  //     // joint_data[which_leg * (10) +  9].pos_des_ = q_1 - 0.0;
  //     // joint_data[which_leg * (10) +  8].pos_des_ = q_2 - 0.0;
  //     // joint_data[which_leg * (10) +  9].vel_des_ = dq_1;
  //     // joint_data[which_leg * (10) +  8].vel_des_ = dq_2;
  //     joint_data[which_leg * (10) +  9].ff_ = tor_1;
  //     joint_data[which_leg * (10) +  8].ff_ = tor_2;
  //   }

  //   // std::cout << "joint_data[9 & 8].kp_" << joint_data[0 * (10) +  9].kp_ << "\t" << joint_data[0 * (10) +  8].kp_ << std::endl;

  //   // std::cout << "joint_data[19 & 18].ff_" << joint_data[1 * (10) +  9].ff_ << "\t" << joint_data[1 * (10) +  8].ff_ << std::endl;

  //   if(!ankle_motor_ready_flag_) {
  //     joint_data[4+4].ff_ = 0.0;
  //     joint_data[5+4].ff_ = 0.0;
  //     joint_data[4+14].ff_ = 0.0;
  //     joint_data[5+14].ff_ = 0.0;
  //   }
  // }
  

} // namespace legged
