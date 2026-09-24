#pragma once

#include "rl_controllers_bumi3/RLControllerBase.h"

namespace legged
{

  class AcController : public RLControllerBase
  {
    using tensor_element_t = float;

  public:
    AcController() : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){}

    ~AcController() override = default;

  protected:
    bool loadModel(ros::NodeHandle &nh) override;
    bool loadRLCfg(ros::NodeHandle &nh) override;
    bool loadMotions(ros::NodeHandle &nh) override;
    void computeActions() override;
    void computeObservation() override;
    void computeActionsDance() override;
    void computeObservationDance() override;
    void computeActions13DOF() override;
    void computeObservation13DOF() override;
    void handleWalkMode() override;
    void handleDanceMode() override;
    void handleWalk13DOFMode() override;

  private:
    void transformBaseOriToTorsoOri(const Eigen::Quaterniond& base_quat, const std::vector<double>& waist_joint_angles, Eigen::Quaterniond& torso_quat);

    // onnx policy model
    std::string policyFilePath_;
    std::string policyFilePathDance_;
    std::string policyFilePath13DOF_;
    std::shared_ptr<Ort::Env> onnxEnvPrt_;
    std::unique_ptr<Ort::Session> policySessionPtr_;
    std::unique_ptr<Ort::Session> policySessionPtrDance_;
    std::unique_ptr<Ort::Session> policySessionPtr13DOF_;

    std::vector<const char *> policyInputNames_;
    std::vector<const char *> policyOutputNames_;
    std::vector<const char *> policyInputNamesDance_;
    std::vector<const char *> policyOutputNamesDance_;
    std::vector<const char *> policyInputNames13DOF_;
    std::vector<const char *> policyOutputNames13DOF_;

    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStringsDance;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStringsDance;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings13DOF;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings13DOF;


    std::vector<std::vector<int64_t>> policyInputShapes_;
    std::vector<std::vector<int64_t>> policyOutputShapes_;
    std::vector<std::vector<int64_t>> policyInputShapesDance_;
    std::vector<std::vector<int64_t>> policyOutputShapesDance_;
    std::vector<std::vector<int64_t>> policyInputShapes13DOF_;
    std::vector<std::vector<int64_t>> policyOutputShapes13DOF_;

    vector3_t baseLinVel_;
    vector3_t basePosition_;
    vector_t lastActions_;
    vector_t defaultJointAngles_;
    vector_t defaultJointAnglesDance_;
    vector_t defaultJointAngles13DOF_;

    bool found_joint_names{false};
    bool found_default_joint_pos{false};
    bool found_action_scale{false};
    bool found_joint_stiffness{false};
    bool found_joint_damping{false};

    std::vector<double> action_scale;
    std::vector<double> joint_stiffness;
    std::vector<double> joint_damping;
    std::vector<double> default_joint_pos;
    std::vector<std::string> joint_names;
    
    std::vector<double> action_scale_dance;
    std::vector<double> joint_stiffness_dance;
    std::vector<double> joint_damping_dance;
    std::vector<double> default_joint_pos_dance;
    std::vector<std::string> joint_names_dance;

    std::vector<double> action_scale_13DOF;
    std::vector<double> joint_stiffness_13DOF;
    std::vector<double> joint_damping_13DOF;
    std::vector<double> default_joint_pos_13DOF;
    std::vector<std::string> joint_names_13DOF;

    
    int actionsSize_;
    int observationSize_;
    int observationSizeDance_;
    int actions13DOFSize_;
    int observation13DOFSize_;
    std::vector<int> jointIndices13DOF_;

    int stackSize_;
    double lastCommandX_;
    std::vector<tensor_element_t> actions_;
    std::vector<tensor_element_t> actions13DOF_;
    std::vector<tensor_element_t> policyObservations_;
    std::vector<tensor_element_t> policyObservationsDance_;
    std::vector<tensor_element_t> policyObservations13DOF_;
    Ort::MemoryInfo memoryInfo;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBufferDance_;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer13DOF_;
     
    Eigen::Quaterniond init_to_world;                // 变换矩阵
    std::vector<std::vector<double>> ref_joint_pos;  // [timestep][joint_index]
    std::vector<std::vector<double>> ref_joint_vel;  // [timestep][joint_index]
    std::vector<std::vector<double>> ref_quat;  // [timestep][joint_index]
    size_t num_timesteps;
    size_t num_joints;
    
    bool isfirstCompAct_{true};
    bool isfirstComp13DOFAct_{true};

    double lastCmdX_{0.0};
    double cmdXSlop_{0.04};
  };

} // namespace legged
