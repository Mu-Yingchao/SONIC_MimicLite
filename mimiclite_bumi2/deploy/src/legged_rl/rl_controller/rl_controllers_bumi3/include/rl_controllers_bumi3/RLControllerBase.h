#pragma once

#include "rl_controllers_bumi3/Types.h"
#include "rl_controllers_bumi3/log_recorder.h"
#include <robot_state_publisher/robot_state_publisher.h>

#include <controller_interface/multi_interface_controller.h>
#include <gazebo_msgs/ModelStates.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <legged_common/hardware_interface/ContactSensorInterface.h>
#include <legged_common/hardware_interface/HybridJointInterface.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <tf/transform_broadcaster.h>

#include <controller_manager_msgs/SwitchController.h>
#include <sensor_msgs/Joy.h>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "TutorialsConfig.h"
#include <dynamic_reconfigure/server.h>
#include <dynamic_reconfigure/ParamDescription.h>

#include <atomic>

namespace legged
{

  struct RLRobotCfg
  {
    struct ControlCfg
    { 
      int decimation;
      float user_torque_limit;
      float user_power_limit;
    };

    scalar_t clipActions;
    scalar_t clipObs;

    ControlCfg controlCfg;
  };

  struct JoyInfo
  {
    float axes[8];
    int buttons[12];
  };

  struct Proprioception
  {
    vector_t jointPos;
    vector_t jointVel;
    vector3_t baseAngVel;
    vector3_t baseEulerXyz;
    vector3_t projectedGravity;
    quaternion_t robot_quat_;
  };

  struct Command
  {
    std::atomic<scalar_t> x;
    std::atomic<scalar_t> y;
    std::atomic<scalar_t> yaw;
  };

  class RLControllerBase : public controller_interface::MultiInterfaceController<HybridJointInterface, hardware_interface::ImuSensorInterface,
                                                                                 ContactSensorInterface>
  {
  public:
    enum class Mode : uint8_t
    {
      LIE,
      STAND,
      WALK,
      DANCE,
      DEFAULT,
      WALK13DOF
    };

    RLControllerBase() = default;
    virtual ~RLControllerBase() = default;
    virtual bool init(hardware_interface::RobotHW *robotHw, ros::NodeHandle &controllerNH);
    virtual void starting(const ros::Time &time);
    virtual void update(const ros::Time &time, const ros::Duration &period);

    virtual bool loadModel(ros::NodeHandle &nh) { return false; };
    virtual bool loadRLCfg(ros::NodeHandle &nh) { return false; };
    virtual bool loadMotions(ros::NodeHandle &nh) { return false; };
    virtual void computeActions(){};
    virtual void computeObservation(){};
    virtual void computeActionsDance(){};
    virtual void computeObservationDance(){};
    virtual void computeActions13DOF(){};
    virtual void computeObservation13DOF(){};

    virtual void handleLieMode();
    virtual void handleStandMode();
    virtual void handleDefautMode();
    virtual void handleWalkMode(){};
    virtual void handleDanceMode(){};
    virtual void handleWalk13DOFMode(){};

    std::unique_ptr<dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>> server_ptr_;
    void dynamicParamCallback(legged_debugger::TutorialsConfig &config, uint32_t level);

  protected:
    virtual void updateStateEstimation(const ros::Time &time, const ros::Duration &period);

    virtual void cmdVelCallback(const geometry_msgs::Twist &msg);
    virtual void joyInfoCallback(const sensor_msgs::Joy &msg);
    bool isfirstRecObs_{true};
    bool isfirstRec13DOFObs_{true};

    // 当前速度上限（全局变量，初始值1.0，范围0.5~2.5）
    float current_vel_limit_ = 1.8;

    Mode mode_;
    int64_t loopCount_;
    Command command_;
    RLRobotCfg robotCfg_{};

    JoyInfo joyInfo;
    std::atomic_bool emergency_stop{false};
    std::atomic_bool start_control{false};
    std::atomic_bool position_control{false};
    ros::Time switchTime;

    vector_t rbdState_;
    vector_t measuredRbdState_;
    Proprioception propri_;

    // hardware interface
    std::vector<std::string> jointNames_;
    std::vector<HybridJointHandle> hybridJointHandles_;

    hardware_interface::ImuSensorHandle imuSensorHandles_;
    std::vector<ContactSensorHandle> contactHandles_;

    ros::Subscriber cmdVelSub_;
    ros::Subscriber joyInfoSub_;
    ros::Subscriber emgStopSub_;
    ros::Subscriber startCtrlSub_;
    ros::Subscriber switchModeSub_;
    ros::Subscriber walkModeSub_;
    ros::Subscriber danceModeSub_;
    ros::Subscriber Walk13DOFModeSub_;
    ros::Subscriber velLimitSub_;

    ros::Subscriber positionCtrlSub_;
    controller_manager_msgs::SwitchController switchCtrlSrv_;
    ros::ServiceClient switchCtrlClient_;

    int actuatedDofNum_ = 21;

    ros::Publisher realJointVelPublisher_;
    ros::Publisher realJointPosPublisher_;
    ros::Publisher realTorquePublisher_;
    std::unique_ptr<robot_state_publisher::RobotStatePublisher> robotStatePublisherPtr_;

    ros::Publisher realImuAngularVelPublisher_;
    ros::Publisher realImuLinearAccPublisher_;
    ros::Publisher realImuEulerXyzPulbisher;

    ros::Publisher outputPlannedJointVelPublisher_;
    ros::Publisher outputPlannedJointPosPublisher_;
    ros::Publisher outputPlannedTorquePublisher_;

    int walkCount_ = 0;
    int danceTimeStep = 0;
    double pd_scale = 1.0;
    float cfg_kd;
    double phase_;
    
    

  private:
    // PD stand

    vector_t pos_des_output_{};
    vector_t vel_des_output_{};

    size_t joint_dim_{0};

    std::vector<scalar_t> currentJointAngles_;
    vector_t standJointAngles_;
    vector_t lieJointAngles_;

    scalar_t standPercent_;
    scalar_t standDuration_;
    tf::TransformBroadcaster tfBroadcaster_;
    
    bool real_log_{true};
    std::unique_ptr<LogRecorder> logRecorder_;
    std::string log_path_;

  };
} // namespace legged
