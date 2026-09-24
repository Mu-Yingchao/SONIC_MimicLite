#pragma once

#include "rl_controllers/RLControllerBase.h"

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
    void computeActions12DOF() override;
    void computeObservation12DOF() override;
    void computeActions13DOFJump() override;
    void computeObservation13DOFJump() override;
    void handleWalkMode() override;
    void handleDanceMode() override;
    void handleWalk12DOFMode() override;
    void handleJump13DOFMode() override;

  private:
    bool loadLocomotionModels(ros::NodeHandle &nh);
    bool loadLocomotionConfig(ros::NodeHandle &nh);
    bool loadMimicLiteDeployConfig(ros::NodeHandle &nh);
    void pushMimicLiteProprioHistory();
    int clampMotionFrame(int frame) const;

    // onnx policy model
    std::string policyFilePath_;
    std::string policyFilePathDance_;
    std::string policyFilePath12DOFWalk_;
    std::string policyFilePath13DOFJump_;
    std::string mimicLiteDeployConfigPath_;
    std::shared_ptr<Ort::Env> onnxEnvPrt_;
    std::unique_ptr<Ort::Session> policySessionPtr_;
    std::unique_ptr<Ort::Session> policySessionPtrDance_;
    std::unique_ptr<Ort::Session> policySessionPtr12DOF_;
    std::unique_ptr<Ort::Session> policySessionPtr13DOFJump_;

    std::vector<const char *> policyInputNames_;
    std::vector<const char *> policyOutputNames_;
    std::vector<const char *> policyInputNamesDance_;
    std::vector<const char *> policyOutputNamesDance_;
    std::vector<const char *> policyInputNames12DOF_;
    std::vector<const char *> policyOutputNames12DOF_;
    std::vector<const char *> policyInputNames13DOFJump_;
    std::vector<const char *> policyOutputNames13DOFJump_;

    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStringsDance;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStringsDance;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings12DOF;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings12DOF;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings13DOFJump;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings13DOFJump;


    std::vector<std::vector<int64_t>> policyInputShapes_;
    std::vector<std::vector<int64_t>> policyOutputShapes_;
    std::vector<std::vector<int64_t>> policyInputShapesDance_;
    std::vector<std::vector<int64_t>> policyOutputShapesDance_;
    std::vector<std::vector<int64_t>> policyInputShapes12DOF_;
    std::vector<std::vector<int64_t>> policyOutputShapes12DOF_;
    std::vector<std::vector<int64_t>> policyInputShapes13DOFJump_;
    std::vector<std::vector<int64_t>> policyOutputShapes13DOFJump_;

    vector3_t baseLinVel_;
    vector3_t basePosition_;
    vector_t lastActions_;
    vector_t defaultJointAngles_;
    vector_t defaultJointAnglesDance_;
    vector_t defaultJointAngles12DOF_;
    vector_t defaultJointAngles13DOFJump_;

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

    std::vector<double> action_scale_12DOF;
    std::vector<double> joint_stiffness_12DOF;
    std::vector<double> joint_damping_12DOF;
    std::vector<double> default_joint_pos_12DOF;
    std::vector<std::string> joint_names_12DOF;

    std::vector<double> action_scale_13DOF_jump;
    std::vector<double> joint_stiffness_13DOF_jump;
    std::vector<double> joint_damping_13DOF_jump;
    std::vector<double> default_joint_pos_13DOF_jump;
    std::vector<std::string> joint_names_13DOF_jump;

    int actionsSize_;
    int observationSize_;
    int observationSizeDance_;
    int commandSizeDance_{0};
    int actions12DOFSize_;
    int observation12DOFSize_;
    std::vector<int> jointIndices12DOF_;
    std::vector<int> fixedUpperBodyJointIndices12DOF_;
    int actions13DOFJumpSize_;
    int observation13DOFJumpSize_;
    std::vector<int> jointIndices13DOFJump_;
    std::vector<int> fixedUpperBodyJointIndices13DOFJump_;
    int stackSize_;
    double lastCommandX_;
    int64_t walkEntryLoopCount_{0};  // 进入 WALK 模式时的循环计数,用于膝盖正弦延时
    std::vector<tensor_element_t> actions_;
    std::vector<tensor_element_t> actions12DOF_;
    std::vector<tensor_element_t> actions13DOFJump_;
    std::vector<tensor_element_t> policyObservations_;
    std::vector<tensor_element_t> policyObservationsDance_;
    // mimic-lite command group: future root local pos / root ori_b / joint pos
    std::vector<tensor_element_t> commandWindowDance_;
    std::vector<tensor_element_t> policyObservations12DOF_;
    std::vector<tensor_element_t> policyObservations13DOFJump_;
    Ort::MemoryInfo memoryInfo;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBufferDance_;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer12DOF_;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer13DOFJump_;
     
    Eigen::Quaterniond init_to_world;                // 变换矩阵
    std::vector<std::vector<double>> ref_joint_pos;  // [timestep][joint_index]
    std::vector<std::vector<double>> ref_joint_vel;  // [timestep][joint_index]
    std::vector<std::vector<double>> ref_quat;  // [timestep][wxyz] root/base
    std::vector<Eigen::Vector3d> ref_root_pos_;      // [timestep]
    size_t num_timesteps;
    size_t num_joints;

    // mimic-lite sparse proprio history + prev-action buffer
    std::vector<int> mimicHistorySteps_;
    std::vector<int> mimicFutureSteps_;
    int mimicPrevActionSteps_{3};
    int mimicHistoryCapacity_{1};
    int mimicHistoryHead_{0};
    bool mimicHistoryReady_{false};
    std::vector<vector3_t> mimicAngVelHist_;
    std::vector<vector3_t> mimicGravityHist_;
    std::vector<vector_t> mimicJointPosHist_;
    std::vector<vector_t> mimicJointVelHist_;
    std::vector<vector_t> mimicPrevActions_;  // [0]=newest

    // Dance mimic playback state: hold-and-blend entry, decimated inference.
    int danceFutureFrames_{0};
    int mimicDecimation_{1};
    int64_t lastMimicInferenceLoop_{-1};
    vector_t mimicHoldJointAngles_;
    vector_t mimicHoldJointStiffness_;
    vector_t mimicHoldJointDamping_;
    ros::Time mimicBlendStartTime_;
    bool mimicPolicyOutputReady_{false};
    double mimicBlendDuration_{0.5};
    
    bool isfirstCompAct_{true};
    bool isfirstComp12DOFAct_{true};
    bool isfirstComp13DOFJumpAct_{true};
  };

} // namespace legged
