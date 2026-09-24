#include "rl_controllers_bumi3/RLControllerBase.h"
#include <string.h>
#include <pluginlib/class_list_macros.hpp>
#include "rl_controllers_bumi3/RotationTools.h"
#include "rl_controllers_bumi3/utilities.h"
#include <kdl_parser/kdl_parser.hpp>
#include <set>

namespace legged
{

  bool RLControllerBase::init(hardware_interface::RobotHW *robotHw, ros::NodeHandle &controllerNH)
  {
    // Hardware interface
    ros::NodeHandle nh;
    if (!nh.getParam("/LeggedRobotCfg/joint_names", jointNames_))
    {
      ROS_ERROR("[Bumi3 RLControllerBase] Missing required /LeggedRobotCfg/joint_names");
      return false;
    }
    const std::set<std::string> uniqueJointNames(jointNames_.begin(), jointNames_.end());
    if (jointNames_.size() != 21 || uniqueJointNames.size() != jointNames_.size() ||
        std::any_of(jointNames_.begin(), jointNames_.end(), [](const std::string &name) { return name.empty(); }))
    {
      ROS_ERROR("[Bumi3 RLControllerBase] /LeggedRobotCfg/joint_names must contain 21 unique, non-empty physical joint names");
      return false;
    }
    actuatedDofNum_ = jointNames_.size();

    std::vector<double> lieJointPositions;
    std::vector<double> standJointPositions;
    if (!nh.getParam("/LeggedRobotCfg/lie_joint_positions", lieJointPositions) ||
        !nh.getParam("/LeggedRobotCfg/stand_joint_positions", standJointPositions) ||
        lieJointPositions.size() != actuatedDofNum_ ||
        standJointPositions.size() != actuatedDofNum_)
    {
      ROS_ERROR_STREAM("[Bumi3 RLControllerBase] Stand and lie position vectors must both have "
                       << actuatedDofNum_ << " elements in Bumi3 physical joint order");
      return false;
    }

    // Load policy model and rl cfg
    if (!loadModel(controllerNH))
    {
      ROS_ERROR_STREAM("[RLControllerBase] Failed to load the model. Ensure the path is correct and accessible.");
      return false;
    }
    if (!loadRLCfg(controllerNH))
    {
      ROS_ERROR_STREAM("[RLControllerBase] Failed to load the rl config. Ensure the yaml is correct and accessible.");
      return false;
    }
    if (!loadMotions(controllerNH))
    {
      ROS_ERROR_STREAM("[RLControllerBase] Failed to load the motions. Ensure the path is correct and accessible.");
      return false;
    }
    standJointAngles_.resize(actuatedDofNum_);
    lieJointAngles_.resize(actuatedDofNum_);
    for (size_t index = 0; index < actuatedDofNum_; ++index)
    {
      lieJointAngles_(index) = lieJointPositions[index];
      standJointAngles_(index) = standJointPositions[index];
    }

    joint_dim_ = actuatedDofNum_;
    //加载kd
    nh.getParam("/Kd_config/kd", cfg_kd);
    
    nh.getParam("/logFile", log_path_);

    if(real_log_) {
      logRecorder_ = std::make_unique<LogRecorder>(log_path_, actuatedDofNum_, jointNames_);
    }

    urdf::Model urdfModel;
    if (!urdfModel.initParam("legged_robot_description")) {
      std::cerr << "[LeggedRobotVisualizer] Could not read URDF from: \"legged_robot_description\"" << std::endl;
    } else {
      KDL::Tree kdlTree;
      kdl_parser::treeFromUrdfModel(urdfModel, kdlTree);
      robotStatePublisherPtr_.reset(new robot_state_publisher::RobotStatePublisher(kdlTree));
    }

    realJointPosPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_joint_pos", 1);
    realJointVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_joint_vel", 1);
    realTorquePublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_torque", 1);

    realImuAngularVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_angular_vel", 1);
    realImuLinearAccPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_linear_acc", 1);
    realImuEulerXyzPulbisher = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_euler_xyz", 1);

    outputPlannedJointPosPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_joint_pos", 1);
    outputPlannedJointVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_joint_vel", 1);
    outputPlannedTorquePublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_torque", 1);

    auto *hybridJointInterface = robotHw->get<HybridJointInterface>();
    for (const auto &jointName : jointNames_)
    {
      hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
    }

    imuSensorHandles_ = robotHw->get<hardware_interface::ImuSensorInterface>()->getHandle("base_imu");

    cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &RLControllerBase::cmdVelCallback, this);
    joyInfoSub_ = controllerNH.subscribe("/joy", 1000, &RLControllerBase::joyInfoCallback, this);
    switchCtrlClient_ = controllerNH.serviceClient<controller_manager_msgs::SwitchController>("/controller_manager/switch_controller");
    auto emergencyStopCallback = [this](const std_msgs::Float32::ConstPtr &msg){emergency_stop = true;ROS_INFO("Emergency Stop");};
    emgStopSub_ = controllerNH.subscribe<std_msgs::Float32>("/emergency_stop", 1, emergencyStopCallback);

    // start control
    auto startControlCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.5);
      if (ros::Time::now() - switchTime > t)
      {
        if (!start_control)
        {
          start_control = true;
          standPercent_ = 0;
          for (size_t i = 0; i < hybridJointHandles_.size(); i++)
          {
            currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
          }
          mode_ = Mode::LIE;
          ROS_INFO("Start Control");
        }
        else
        {
          start_control = false;
          mode_ = Mode::DEFAULT;
          ROS_INFO("ShutDown Control");
        }
        switchTime = ros::Time::now();
      }
    };
    startCtrlSub_ = controllerNH.subscribe<std_msgs::Float32>("/start_control", 1, startControlCallback);

    // switchMode
    auto switchModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.8);
      if (ros::Time::now() - switchTime > t)
      {
        if (start_control == true)
        {
          if (mode_ == Mode::STAND)
          {
            standPercent_ = 0;
            for (size_t i = 0; i < hybridJointHandles_.size(); i++)
            {
              currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
            }
            mode_ = Mode::LIE;
            ROS_INFO("STAND2LIE");
          }
          else if (mode_ == Mode::LIE)
          {
            standPercent_ = 0;
            mode_ = Mode::STAND;
            ROS_INFO("LIE2STAND");
          }
        }
        switchTime = ros::Time::now();
      }
    };
    switchModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/switch_mode", 1, switchModeCallback);

    // walkMode
    auto walkModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::STAND)
        {
          mode_ = Mode::WALK;
          ROS_INFO("STAND2WALK");
          isfirstRecObs_ = true;
        } 
        else if (mode_ == Mode::WALK13DOF)
        {
          mode_ = Mode::WALK;
          ROS_INFO("WALK13DOF2WALK");
          isfirstRecObs_ = true;
        }
        switchTime = ros::Time::now();
      }
    };
    walkModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/walk_mode", 1, walkModeCallback);

    // DANCEMode
    auto danceModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::WALK || mode_ == Mode::STAND || mode_ == Mode::LIE)
        {
          mode_ = Mode::DANCE;
          ROS_INFO("walk2dance");
          danceTimeStep = 0;
          isfirstRecObs_ = true;
        }
        switchTime = ros::Time::now();
      }
    };
    danceModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/dance_mode", 1, danceModeCallback);

    auto Walk13DOFModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::STAND)
        {
          mode_ = Mode::WALK13DOF;
          ROS_INFO("STAND to WALK13DOF");
          isfirstRecObs_ = true;
        }else if (mode_ == Mode::WALK)
        {
          mode_ = Mode::WALK13DOF;
          ROS_INFO("WALK to WALK13DOF");
          isfirstRecObs_ = true;
        }
        switchTime = ros::Time::now();
      }
    };
    Walk13DOFModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/walk13dof_mode", 1, Walk13DOFModeCallback);

    // positionMode
    auto positionModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::WALK)
        {
          mode_ = Mode::STAND;
          ROS_INFO("WALK2STAND");
        }
        else if (mode_ == Mode::DEFAULT)
        {
          standPercent_ = 0;
          for (size_t i = 0; i < hybridJointHandles_.size(); i++)
          {
            currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
          }
          mode_ = Mode::LIE;
          ROS_INFO("DEF2LIE");
        }
        else if (mode_ == Mode::WALK13DOF)
        {
          standPercent_ = 0;
          for (size_t i = 0; i < hybridJointHandles_.size(); i++)
          {
            currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
          }
          mode_ = Mode::LIE;
          ROS_INFO("WALK13DOF2LIE");
        }

        switchTime = ros::Time::now();
      }
    };
    positionCtrlSub_ = controllerNH.subscribe<std_msgs::Float32>("/position_control", 1, positionModeCallback);

       // vel_limit 订阅
    auto velLimitCallback = [this](const std_msgs::Float32::ConstPtr &msg) {
      static ros::Time lastUpdate(0);
      float val = msg->data;
      // 只有轴值不为0时才更新（按下时）
      if (std::abs(val) > 1e-3) {
        ros::Time now = ros::Time::now();
        // 时间保护，间隔至少0.1秒
        if ((now - lastUpdate).toSec() > 0.3f) {
          float new_limit = this->current_vel_limit_ + val;
          if (new_limit < 1.0f) new_limit = 1.0f;
          if (new_limit > 1.5f) new_limit = 1.5f;
          this->current_vel_limit_ = new_limit;
          ROS_INFO_STREAM("[RLControllerBase] vel_limit set to: " << new_limit);
          lastUpdate = now;
        }
      }
      // 弹起（val==0）时不更新
    };
    velLimitSub_ = controllerNH.subscribe<std_msgs::Float32>("/vel_limit", 1, velLimitCallback);

    return true;
  }

  std::atomic<scalar_t> kp_stance{0};
  std::atomic<scalar_t> kd_stance{3};

  // only once
  void RLControllerBase::starting(const ros::Time &time)
  {
    updateStateEstimation(time, ros::Duration(0.002));
    currentJointAngles_.resize(hybridJointHandles_.size());
    scalar_t durationSecs = 2.0;
    standDuration_ = durationSecs * 500.0;
    standPercent_ = 0;
    mode_ = Mode::DEFAULT;
    loopCount_ = 0;

    server_ptr_ = std::make_unique<dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>>(ros::NodeHandle("controller"));
    dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>::CallbackType f;
    f = boost::bind(&RLControllerBase::dynamicParamCallback, this, _1, _2);
    server_ptr_->setCallback(f);

    pos_des_output_.resize(joint_dim_);
    vel_des_output_.resize(joint_dim_);
    pos_des_output_.setZero();
    vel_des_output_.setZero();

  }

  void RLControllerBase::update(const ros::Time &time, const ros::Duration &period)
  {
    if(real_log_) {
      logRecorder_->Open();
      logRecorder_->WriteScalar(time.toSec());
    }
    
    ros::NodeHandle nh;
    updateStateEstimation(time, period);
    // ROS_WARN(mode: %d, mode_);
    switch (mode_)
    {
    case Mode::DEFAULT:
      handleDefautMode();
      break;
    case Mode::LIE:
      handleLieMode();
      break;
    case Mode::STAND:
      handleStandMode();
      break;
    case Mode::WALK:
      handleWalkMode();
      break;
    case Mode::DANCE:
      handleDanceMode();
      break;
    case Mode::WALK13DOF:
      handleWalk13DOFMode();
      break;
    default:
      ROS_ERROR_STREAM("Unexpected mode encountered: " << static_cast<int>(mode_));
      break;
    }
    if (emergency_stop)
    {
      emergency_stop = false;
      mode_ = Mode::DEFAULT;
    }

    vector_t output_torque(joint_dim_);
    for (int j = 0; j < hybridJointHandles_.size(); j++)
    {
      pos_des_output_(j) = hybridJointHandles_[j].getPositionDesired();
      vel_des_output_(j) = hybridJointHandles_[j].getVelocityDesired();
      output_torque(j) = hybridJointHandles_[j].getFeedforward() +
                          hybridJointHandles_[j].getKp() * (hybridJointHandles_[j].getPositionDesired() - hybridJointHandles_[j].getPosition()) +
                          hybridJointHandles_[j].getKd() * (hybridJointHandles_[j].getVelocityDesired() - hybridJointHandles_[j].getVelocity());
    }
    outputPlannedJointPosPublisher_.publish(createFloat64MultiArrayFromVector(pos_des_output_));
    outputPlannedJointVelPublisher_.publish(createFloat64MultiArrayFromVector(vel_des_output_));
    outputPlannedTorquePublisher_.publish(createFloat64MultiArrayFromVector(output_torque));
    
    if(real_log_) {
      logRecorder_->WriteEigenVec(pos_des_output_);
      logRecorder_->WriteEigenVec(vel_des_output_);
      logRecorder_->WriteEigenVec(output_torque);
      
      logRecorder_->Close();
    }

    loopCount_++;
  }

  void RLControllerBase::handleDefautMode()
  {
    for (int j = 0; j < hybridJointHandles_.size(); j++)
      hybridJointHandles_[j].setCommand(0, 0, 0, 0.1, 0);

    // ROS_WARN(The value of kdConfig.cfg_kd is: %f, kdConfig.cfg_kd);
  }

  void RLControllerBase::handleLieMode()
  {
    if (standPercent_ <= 1)
    {
      for (int j = 0; j < hybridJointHandles_.size(); j++)
      {
        scalar_t pos_des = currentJointAngles_[j] * (1 - standPercent_) + lieJointAngles_[j] * standPercent_;
        if (j == 15 ||  j == 19 ) hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
        else if (j == 16 || j == 20) hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
        else hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
      }
      standPercent_ += 1 / standDuration_;
      standPercent_ = std::min(standPercent_, scalar_t(1));
    }
  }

  void RLControllerBase::handleStandMode()
  {
    if (standPercent_ <= 1)
    {
      for (int j = 0; j < hybridJointHandles_.size(); j++)
      {
        scalar_t pos_des = lieJointAngles_[j] * (1 - standPercent_) + standJointAngles_[j] * standPercent_;
        if (j == 15 ||  j == 19 ) hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
        else if (j == 16 || j == 20) hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
        else hybridJointHandles_[j].setCommand(pos_des, 0, 10, 0.5, 0);
      }
      standPercent_ += 1 / standDuration_;
      standPercent_ = std::min(standPercent_, scalar_t(1));
    }
  }

  void RLControllerBase::updateStateEstimation(const ros::Time &time, const ros::Duration &period)
  {
    vector_t jointPos(actuatedDofNum_), jointVel(actuatedDofNum_), jointTor(actuatedDofNum_);

    vector_t imuEulerXyz(3);
    contact_flag_t contacts;
    quaternion_t quat;
    vector3_t angularVel, linearAccel;
    matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;

    for (size_t i = 0; i <  actuatedDofNum_; ++i)
    {
      jointPos(i) = hybridJointHandles_[i].getPosition();
      jointVel(i) = hybridJointHandles_[i].getVelocity();
      jointTor(i) = hybridJointHandles_[i].getEffort();
    }
    
    for (size_t i = 0; i < 4; ++i)
    {
      quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
    }
    for (size_t i = 0; i < 3; ++i)
    {
      angularVel(i) = imuSensorHandles_.getAngularVelocity()[i];
      linearAccel(i) = imuSensorHandles_.getLinearAcceleration()[i];
    }
    for (size_t i = 0; i < 9; ++i)
    {
      orientationCovariance(i) = imuSensorHandles_.getOrientationCovariance()[i];
      angularVelCovariance(i) = imuSensorHandles_.getAngularVelocityCovariance()[i];
      linearAccelCovariance(i) = imuSensorHandles_.getLinearAccelerationCovariance()[i];
    }

    propri_.jointPos = jointPos;
    propri_.jointVel = jointVel;
    propri_.baseAngVel = angularVel;
    propri_.robot_quat_ = quat;


    vector3_t gravityVector(0, 0, -1);
    vector3_t zyx = quatToZyx(quat);
    propri_.baseEulerXyz[0] = zyx[2];
    propri_.baseEulerXyz[1] = zyx[1];
    propri_.baseEulerXyz[2] = zyx[0];

    matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();
    propri_.projectedGravity = inverseRot * gravityVector;
    phase_ = time.toSec();
    for (size_t i = 0; i < 3; ++i)
    {
      imuEulerXyz(i) = propri_.baseEulerXyz[i];
    }

    realImuAngularVelPublisher_.publish(createFloat64MultiArrayFromVector(angularVel));
    realImuLinearAccPublisher_.publish(createFloat64MultiArrayFromVector(linearAccel));
    realImuEulerXyzPulbisher.publish(createFloat64MultiArrayFromVector(imuEulerXyz));

    realTorquePublisher_.publish(createFloat64MultiArrayFromVector(jointTor));
    realJointPosPublisher_.publish(createFloat64MultiArrayFromVector(jointPos));
    realJointVelPublisher_.publish(createFloat64MultiArrayFromVector(jointVel));

    robotStatePublisherPtr_->publishFixedTransforms(true);
    tf::Transform baseTransform;
    baseTransform.setOrigin(tf::Vector3(0.0, 0.0, 0.0)); // Origin
    baseTransform.setRotation(tf::Quaternion(quat.coeffs()(0), quat.coeffs()(1), quat.coeffs()(2), quat.coeffs()(3)));  
    tfBroadcaster_.sendTransform(tf::StampedTransform(baseTransform, time, "world", "base_link"));

  std::map<std::string, scalar_t> jointPositions;
  for (size_t i = 0; i < jointNames_.size(); ++i)
  {
    jointPositions[jointNames_[i]] = jointPos(i);
  }
   robotStatePublisherPtr_->publishTransforms(jointPositions, time);
   
   if(real_log_) {
    logRecorder_->WriteEigenVec(jointPos);
    logRecorder_->WriteEigenVec(jointVel);
    logRecorder_->WriteEigenVec(jointTor);
    
    logRecorder_->WriteEigenVec(imuEulerXyz);
    logRecorder_->WriteEigenVec(angularVel);
    logRecorder_->WriteEigenVec(linearAccel);
   }
  }

  void RLControllerBase::cmdVelCallback(const geometry_msgs::Twist &msg)
  {
    command_.x = msg.linear.x;
    command_.y = msg.linear.y;
    command_.yaw = msg.angular.z;
  }

  void RLControllerBase::dynamicParamCallback(legged_debugger::TutorialsConfig &config, uint32_t level)
  {
    kp_stance = config.kp_stance;
    kd_stance = config.kd_stance;
  }

  void RLControllerBase::joyInfoCallback(const sensor_msgs::Joy &msg)
  {
    if (msg.header.frame_id.empty())
    {
      return;
    }
    for (int i = 0; i < msg.axes.size(); i++)
    {
      joyInfo.axes[i] = msg.axes[i];
    }
    for (int i = 0; i < msg.buttons.size(); i++)
    {
      joyInfo.buttons[i] = msg.buttons[i];
    }
  }

  
} // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::RLControllerBase, controller_interface::ControllerBase)
