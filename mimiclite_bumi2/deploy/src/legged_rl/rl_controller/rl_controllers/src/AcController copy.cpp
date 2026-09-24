#include "rl_controllers/AcController.h"
#include <pluginlib/class_list_macros.hpp>
#include "rl_controllers/RotationTools.h"
#include <algorithm>
#include <cmath>
#include <random>
#include "nlohmann/json.hpp"  // 需要包含这个头文件
#include <fstream>  // 添加这个头文件用于 std::ifstream

using json = nlohmann::json;


namespace legged
{
  std::vector<std::string> SplitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
  }

  std::vector<float> SplitString2Float(const std::string& str, char delimiter) {
    std::vector<float> values;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            values.push_back(std::stof(token));
        }
    }
    return values;
  }

  namespace
  {
    void readPolicyMetadata(
        Ort::Session &session,
        Ort::AllocatorWithDefaultOptions &allocator,
        std::vector<std::string> &jointNames,
        std::vector<double> &jointStiffness,
        std::vector<double> &jointDamping,
        std::vector<double> &defaultJointPosition,
        std::vector<double> &actionScale)
    {
      Ort::ModelMetadata metadata = session.GetModelMetadata();
      const auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
      for (const auto &keyPtr : keys)
      {
        const std::string key(keyPtr.get());
        auto valuePtr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
        const std::string value(valuePtr.get());
        if (key == "joint_names")
        {
          jointNames = SplitString(value, ',');
        }
        else if (key == "joint_stiffness")
        {
          const auto parsed = SplitString2Float(value, ',');
          jointStiffness.assign(parsed.begin(), parsed.end());
        }
        else if (key == "joint_damping")
        {
          const auto parsed = SplitString2Float(value, ',');
          jointDamping.assign(parsed.begin(), parsed.end());
        }
        else if (key == "default_joint_pos")
        {
          const auto parsed = SplitString2Float(value, ',');
          defaultJointPosition.assign(parsed.begin(), parsed.end());
        }
        else if (key == "action_scale")
        {
          const auto parsed = SplitString2Float(value, ',');
          actionScale.assign(parsed.begin(), parsed.end());
        }
      }
    }

    void readPolicyIo(
        Ort::Session &session,
        Ort::AllocatorWithDefaultOptions &allocator,
        std::vector<Ort::AllocatedStringPtr> &inputNameStorage,
        std::vector<Ort::AllocatedStringPtr> &outputNameStorage,
        std::vector<const char *> &inputNames,
        std::vector<const char *> &outputNames,
        std::vector<std::vector<int64_t>> &inputShapes,
        std::vector<std::vector<int64_t>> &outputShapes,
        const std::string &label)
    {
      inputNameStorage.clear();
      outputNameStorage.clear();
      inputNames.clear();
      outputNames.clear();
      inputShapes.clear();
      outputShapes.clear();

      for (size_t i = 0; i < session.GetInputCount(); ++i)
      {
        auto name = session.GetInputNameAllocated(i, allocator);
        inputNameStorage.push_back(std::move(name));
        inputNames.push_back(inputNameStorage.back().get());
        inputShapes.push_back(
            session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      }
      for (size_t i = 0; i < session.GetOutputCount(); ++i)
      {
        auto name = session.GetOutputNameAllocated(i, allocator);
        outputNameStorage.push_back(std::move(name));
        outputNames.push_back(outputNameStorage.back().get());
        outputShapes.push_back(
            session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      }

      std::ostringstream shapeStream;
      shapeStream << label << " input shape:";
      if (!inputShapes.empty())
      {
        shapeStream << " [";
        for (size_t i = 0; i < inputShapes.front().size(); ++i)
        {
          if (i != 0) shapeStream << ", ";
          shapeStream << inputShapes.front()[i];
        }
        shapeStream << "]";
      }
      if (!outputShapes.empty())
      {
        shapeStream << ", output shape: [";
        for (size_t i = 0; i < outputShapes.front().size(); ++i)
        {
          if (i != 0) shapeStream << ", ";
          shapeStream << outputShapes.front()[i];
        }
        shapeStream << "]";
      }
      ROS_INFO_STREAM(shapeStream.str());
    }
  } // namespace

  void AcController::handleWalkMode()
  {
    // compute observation & actions
    if (std::cout.fail())
    {
      std::cerr << "std::cout is in a bad state!" << std::endl;
      // 可能需要清除错误状态
      std::cout.clear();
    }
    // 摔倒保护
    if (propri_.projectedGravity(2) >= -0.3)
    {
      std::cout << "摔倒保护" << std::endl;
      mode_ = Mode::DEFAULT;
    }
    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {
      computeObservation();  // 注释 policy 观测
      computeActions();      // 注释 policy 推理
      // limit action range
      scalar_t actionMin = -robotCfg_.clipActions;
      scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                     [actionMin, actionMax](scalar_t x)
                     { return std::max(actionMin, std::min(actionMax, x)); });
    }
    scalar_t pos_des;
    // set action
    for (int i = 0; i < actionsSize_; i++)
    {
      std::string partName = hybridJointHandles_[i].getName();

      // actions_[i] = 0;
      // pos_des = 0;  // 默认回0位
           if(i==15|| i==16){
        joint_stiffness[i]= 8.0;
        joint_damping[i]= 0.5;
        // vel_des = 0.3 * cos(phase_);
        // pos_des =  0.3 * sin(phase_);  // 15,16 输出正弦波
      }
      if(i==19|| i==20){
        joint_stiffness[i]= 2.0;
        joint_damping[i]= 0.2;
        // pos_des = 0;  // 19,20 回 0 位
        // vel_des = 0;
      }
      //   ||i==19||i==20

      pos_des = actions_[i] * action_scale[i] + defaultJointAngles_(i);  // 注释policy输出，使用正弦波
      // std::cout << "joint_name: " << partName << " kp:" << joint_stiffness[i] << " kd:" << joint_damping[i] << " action_scale: "<< action_scale[i] << std::endl;
      hybridJointHandles_[i].setCommand(pos_des, 0, joint_stiffness[i], joint_damping[i], 0);
      lastActions_(i, 0) = actions_[i];
    }
  
  }
  void AcController::handleDanceMode()
  {
    // compute observation & actions
    if (std::cout.fail())
    {
      std::cerr << "std::cout is in a bad state!" << std::endl;
      // 可能需要清除错误状态
      std::cout.clear();
    }
    // 摔倒保护
    // if (propri_.projectedGravity(2) >= -0.3)
    // {
    //   std::cout << "摔倒保护" << std::endl;
    //   mode_ = Mode::DEFAULT;
    // }


    // 初始化随机数引擎（使用硬件种子）
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 定义分布：[0, 1)
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float random_num = dis(gen);
    // set action

    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {

      computeObservationDance();
      computeActionsDance();
      // limit action range
      scalar_t actionMin = -robotCfg_.clipActions;
      scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                     [actionMin, actionMax](scalar_t x)
                     { return std::max(actionMin, std::min(actionMax, x)); });
    }

    // set action
    for (int i = 0; i < actionsSize_; i++)
    {
      std::string partName = hybridJointHandles_[i].getName();
      // actions_[i] = random_num * actions_[i] + (1 - random_num) * lastActions_(i, 0);
      // if(i==15||i==16||
      //   i==19||i==20){
      //     joint_stiffness_dance[i]= 30.0;
      //     joint_damping_dance[i]= 1.0;
      // }

      
      // actions_[i] = 0.5 * actions_[i] + 0.5 * lastActions_(i, 0);
      scalar_t pos_des = actions_[i] * action_scale_dance[i] + defaultJointAnglesDance_(i);
      // if(i == 4|| i== 5|| i== 10||i==11 8){
      //   actions_[i] = 0.8 * actions_[i] + (1 - 0.8) * lastActions_(i, 0);
      // }
      // std::cout << "joint_name: " << partName << " kp:" << joint_stiffness[i] << " kd:" << joint_damping[i] << " action_scale: "<< action_scale[i] << std::endl;
      hybridJointHandles_[i].setCommand(pos_des, 0, joint_stiffness_dance[i], joint_damping_dance[i], 0);
      lastActions_(i, 0) = actions_[i];
    }
  
  }

  void AcController::handleWalk12DOFMode()
  {
    if (propri_.projectedGravity(2) >= -0.3)
    {
      ROS_WARN("WALK12DOF fall protection");
      mode_ = Mode::DEFAULT;
      return;
    }

    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {
      computeObservation12DOF();
      computeActions12DOF();
      const scalar_t actionMin = -robotCfg_.clipActions;
      const scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions12DOF_.begin(), actions12DOF_.end(), actions12DOF_.begin(),
                     [actionMin, actionMax](scalar_t action)
                     { return std::max(actionMin, std::min(actionMax, action)); });
    }

    scalar_t pos_des;
    // Only the twelve leg joints are controlled by the policy.
    for (int actionIndex = 0; actionIndex < actions12DOFSize_; ++actionIndex)
    {
      const int jointIndex = jointIndices12DOF_[actionIndex];
      scalar_t kp = joint_stiffness_12DOF[jointIndex];
      scalar_t kd = joint_damping_12DOF[jointIndex];

      // Match the ankle PD gains used by the 21-DOF walking controller.
      // Keep this override in the controller so it takes precedence over the
      // gains embedded in the lower-body ONNX model metadata.
      if (actionIndex == 8 || actionIndex == 9)
      {
        kp = 5.0;
        kd = 0.5;
      }
      else if (actionIndex == 10 || actionIndex == 11)
      {
        kp = 2.0;
        kd = 0.2;
      }

      pos_des =
          actions12DOF_[actionIndex] * action_scale_12DOF[actionIndex] +
          defaultJointAngles12DOF_(actionIndex);
      hybridJointHandles_[jointIndex].setCommand(pos_des, 0, kp, kd, 0);
      lastActions_(jointIndex, 0) = actions12DOF_[actionIndex];
    }

    // Waist and arms stay at the same default hanging-arm pose as walking.
    for (const int jointIndex : fixedUpperBodyJointIndices12DOF_)
    {
      hybridJointHandles_[jointIndex].setCommand(
          default_joint_pos_12DOF[jointIndex], 0,
          joint_stiffness_12DOF[jointIndex],
          joint_damping_12DOF[jointIndex], 0);
      lastActions_(jointIndex, 0) = 0.0;
    }
  }

  void AcController::handleJump13DOFMode()
  {
    if (propri_.projectedGravity(2) >= -0.3)
    {
      ROS_WARN("JUMP13DOF fall protection");
      mode_ = Mode::DEFAULT;
      return;
    }

    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {
      computeObservation13DOFJump();
      computeActions13DOFJump();
      const scalar_t actionMin = -robotCfg_.clipActions;
      const scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions13DOFJump_.begin(), actions13DOFJump_.end(),
                     actions13DOFJump_.begin(),
                     [actionMin, actionMax](scalar_t action)
                     { return std::max(actionMin, std::min(actionMax, action)); });
    }

    for (int actionIndex = 0; actionIndex < actions13DOFJumpSize_; ++actionIndex)
    {
      const int jointIndex = jointIndices13DOFJump_[actionIndex];
      const scalar_t positionDesired =
          actions13DOFJump_[actionIndex] * action_scale_13DOF_jump[actionIndex] +
          defaultJointAngles13DOFJump_(actionIndex);
      hybridJointHandles_[jointIndex].setCommand(
          positionDesired, 0,
          joint_stiffness_13DOF_jump[jointIndex],
          joint_damping_13DOF_jump[jointIndex], 0);
      lastActions_(jointIndex, 0) = actions13DOFJump_[actionIndex];
    }

    for (const int jointIndex : fixedUpperBodyJointIndices13DOFJump_)
    {
      hybridJointHandles_[jointIndex].setCommand(
          default_joint_pos_13DOF_jump[jointIndex], 0,
          joint_stiffness_13DOF_jump[jointIndex],
          joint_damping_13DOF_jump[jointIndex], 0);
      lastActions_(jointIndex, 0) = 0.0;
    }
  }

  bool AcController::loadMotions(ros::NodeHandle &nh)
  {
      std::cout << "=== MotionLoad 初始化 ===" << std::endl;
      std::string motionFilePath;
      
      if (!nh.getParam("/motionFilePath", motionFilePath))
      {
          ROS_ERROR_STREAM("Get motion path fail from param server, some error occur!");
          return false;
      }
      
      std::cout << "运动文件路径: " << motionFilePath << std::endl;
      std::ifstream file(motionFilePath);
      if (!file.is_open()) {
          std::cerr << "无法打开文件: " << motionFilePath << std::endl;
          return false;
      }
      json motion;
      file >> motion;
       // 解析关节位置数据
      ref_joint_pos = motion["joint_pos"].get<std::vector<std::vector<double>>>();
      ref_joint_vel = motion["joint_vel"].get<std::vector<std::vector<double>>>();
      ref_quat = motion["body_quat_w"].get<std::vector<std::vector<double>>>(); // w x y z

      num_timesteps = ref_joint_pos.size();
      if (num_timesteps > 0) {
          num_joints = ref_joint_pos[0].size();
      } else {
          num_joints = 0;
      }
      
      std::cout << "成功加载数据: " << num_timesteps << " 个时间步, " 
            << num_joints << " 个关节" << std::endl;
      return true;
      
  }

  bool AcController::loadModel(ros::NodeHandle &nh)
  {
    return loadLocomotionModels(nh);
  }
  bool AcController::loadLocomotionModels(ros::NodeHandle &nh)
  {
    if (!nh.getParam("/policyFile21DOF", policyFilePath_))
    {
      ROS_ERROR("Missing /policyFile21DOF");
      return false;
    }
    if (!nh.getParam("/policyFile12DOFWalk", policyFilePath12DOFWalk_))
    {
      ROS_ERROR("Missing /policyFile12DOFWalk");
      return false;
    }
    if (!nh.getParam("/policyFile13DOFJump", policyFilePath13DOFJump_))
    {
      ROS_ERROR("Missing /policyFile13DOFJump");
      return false;
    }

    ROS_INFO_STREAM("Load 21-DOF policy: " << policyFilePath_);
    ROS_INFO_STREAM("Load 12-DOF walk policy: " << policyFilePath12DOFWalk_);
    ROS_INFO_STREAM("Load 13-DOF jump entry: " << policyFilePath13DOFJump_);

    onnxEnvPrt_ = std::make_shared<Ort::Env>(
        ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(1);
    policySessionPtr_ = std::make_unique<Ort::Session>(
        *onnxEnvPrt_, policyFilePath_.c_str(), sessionOptions);
    policySessionPtr12DOF_ = std::make_unique<Ort::Session>(
        *onnxEnvPrt_, policyFilePath12DOFWalk_.c_str(), sessionOptions);
    policySessionPtr13DOFJump_ = std::make_unique<Ort::Session>(
        *onnxEnvPrt_, policyFilePath13DOFJump_.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    readPolicyMetadata(
        *policySessionPtr_, allocator, joint_names, joint_stiffness,
        joint_damping, default_joint_pos, action_scale);
    readPolicyMetadata(
        *policySessionPtr12DOF_, allocator, joint_names_12DOF,
        joint_stiffness_12DOF, joint_damping_12DOF,
        default_joint_pos_12DOF, action_scale_12DOF);
    readPolicyMetadata(
        *policySessionPtr13DOFJump_, allocator, joint_names_13DOF_jump,
        joint_stiffness_13DOF_jump, joint_damping_13DOF_jump,
        default_joint_pos_13DOF_jump, action_scale_13DOF_jump);

    readPolicyIo(
        *policySessionPtr_, allocator,
        policyInputNodeNameAllocatedStrings,
        policyOutputNodeNameAllocatedStrings,
        policyInputNames_, policyOutputNames_,
        policyInputShapes_, policyOutputShapes_, "21-DOF");
    readPolicyIo(
        *policySessionPtr12DOF_, allocator,
        policyInputNodeNameAllocatedStrings12DOF,
        policyOutputNodeNameAllocatedStrings12DOF,
        policyInputNames12DOF_, policyOutputNames12DOF_,
        policyInputShapes12DOF_, policyOutputShapes12DOF_, "12-DOF walk");
    readPolicyIo(
        *policySessionPtr13DOFJump_, allocator,
        policyInputNodeNameAllocatedStrings13DOFJump,
        policyOutputNodeNameAllocatedStrings13DOFJump,
        policyInputNames13DOFJump_, policyOutputNames13DOFJump_,
        policyInputShapes13DOFJump_, policyOutputShapes13DOFJump_,
        "13-DOF jump entry");

    ROS_INFO("Loaded all three locomotion policies successfully");
    return true;
  }

  bool AcController::loadRLCfg(ros::NodeHandle &nh)
  {
    return loadLocomotionConfig(nh);
  }
  bool AcController::loadLocomotionConfig(ros::NodeHandle &nh)
  {
    int missing = 0;
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/control/decimation", robotCfg_.controlCfg.decimation));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/actions_size", actionsSize_));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/actions_12dof_walk_size", actions12DOFSize_));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/observations_12dof_walk_size", observation12DOFSize_));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/actions_13dof_jump_size", actions13DOFJumpSize_));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/observations_13dof_jump_size", observation13DOFJumpSize_));
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/stack_size", stackSize_));
    if (missing != 0)
    {
      ROS_ERROR_STREAM("[AcController] Missing " << missing
                       << " three-policy configuration parameter(s)");
      return false;
    }

    const auto tensorSize = [](const std::vector<std::vector<int64_t>> &shapes,
                               const std::string &label) -> int64_t
    {
      if (shapes.size() != 1 || shapes.front().empty())
      {
        ROS_ERROR_STREAM("[AcController] " << label << " must have one tensor");
        return -1;
      }
      int64_t size = 1;
      for (const int64_t dim : shapes.front())
      {
        if (dim <= 0)
        {
          ROS_ERROR_STREAM("[AcController] " << label
                           << " has unsupported dynamic dimension " << dim);
          return -1;
        }
        size *= dim;
      }
      return size;
    };

    const int64_t fullInputSize = tensorSize(policyInputShapes_, "21-DOF input");
    const int64_t fullOutputSize = tensorSize(policyOutputShapes_, "21-DOF output");
    const int64_t walkInputSize = tensorSize(policyInputShapes12DOF_, "12-DOF walk input");
    const int64_t walkOutputSize = tensorSize(policyOutputShapes12DOF_, "12-DOF walk output");
    const int64_t jumpInputSize = tensorSize(policyInputShapes13DOFJump_, "13-DOF jump input");
    const int64_t jumpOutputSize = tensorSize(policyOutputShapes13DOFJump_, "13-DOF jump output");

    if (stackSize_ <= 0 || fullInputSize <= 0 || fullInputSize % stackSize_ != 0)
    {
      ROS_ERROR("[AcController] Invalid 21-DOF history shape");
      return false;
    }
    observationSize_ = static_cast<int>(fullInputSize / stackSize_);
    if ((observationSize_ != 71 && observationSize_ != 72) ||
        fullOutputSize != actionsSize_)
    {
      ROS_ERROR_STREAM("[AcController] 21-DOF policy must be 355/360 -> 21, got "
                       << fullInputSize << " -> " << fullOutputSize);
      return false;
    }
    if (walkInputSize != observation12DOFSize_ * stackSize_ ||
        walkOutputSize != actions12DOFSize_ ||
        observation12DOFSize_ != 44 || actions12DOFSize_ != 12)
    {
      ROS_ERROR_STREAM("[AcController] 12-DOF walk must be 220 -> 12, got "
                       << walkInputSize << " -> " << walkOutputSize);
      return false;
    }
    if (jumpInputSize != observation13DOFJumpSize_ * stackSize_ ||
        jumpOutputSize != actions13DOFJumpSize_ ||
        observation13DOFJumpSize_ != 45 || actions13DOFJumpSize_ != 12)
    {
      ROS_ERROR_STREAM("[AcController] 13-DOF jump entry currently expects 225 -> 12, got "
                       << jumpInputSize << " -> " << jumpOutputSize);
      return false;
    }

    const auto validateFullMetadata = [this](
        const std::vector<std::string> &names,
        const std::vector<double> &stiffness,
        const std::vector<double> &damping,
        const std::vector<double> &defaults,
        const std::string &label)
    {
      if (names.size() != static_cast<size_t>(actionsSize_) ||
          stiffness.size() != static_cast<size_t>(actionsSize_) ||
          damping.size() != static_cast<size_t>(actionsSize_) ||
          defaults.size() != static_cast<size_t>(actionsSize_))
      {
        ROS_ERROR_STREAM("[AcController] Invalid full metadata sizes for " << label);
        return false;
      }
      for (int i = 0; i < actionsSize_; ++i)
      {
        if (names[i] != jointNames_[i])
        {
          ROS_ERROR_STREAM("[AcController] " << label
                           << " metadata joint mismatch at " << i
                           << ": model=" << names[i]
                           << ", controller=" << jointNames_[i]);
          return false;
        }
      }
      return true;
    };

    if (!validateFullMetadata(joint_names, joint_stiffness, joint_damping,
                              default_joint_pos, "21-DOF") ||
        action_scale.size() != static_cast<size_t>(actionsSize_) ||
        !validateFullMetadata(joint_names_12DOF, joint_stiffness_12DOF,
                              joint_damping_12DOF, default_joint_pos_12DOF,
                              "12-DOF walk") ||
        !validateFullMetadata(joint_names_13DOF_jump, joint_stiffness_13DOF_jump,
                              joint_damping_13DOF_jump,
                              default_joint_pos_13DOF_jump,
                              "13-DOF jump"))
    {
      return false;
    }

    const auto configureSubset = [this, &nh](
        const std::string &parameter,
        int actionCount,
        const std::vector<double> &metadataScale,
        const std::vector<double> &metadataDefaults,
        std::vector<int> &jointIndices,
        std::vector<int> &fixedJointIndices,
        vector_t &compactDefaults,
        std::vector<double> &compactScale,
        const std::string &label)
    {
      std::vector<std::string> names;
      if (!nh.getParam(parameter, names) ||
          names.size() != static_cast<size_t>(actionCount))
      {
        ROS_ERROR_STREAM("[AcController] Invalid " << parameter);
        return false;
      }
      jointIndices.clear();
      for (const auto &name : names)
      {
        const auto it = std::find(jointNames_.begin(), jointNames_.end(), name);
        if (it == jointNames_.end())
        {
          ROS_ERROR_STREAM("[AcController] Unknown " << label << " joint " << name);
          return false;
        }
        const int index = static_cast<int>(std::distance(jointNames_.begin(), it));
        if (std::find(jointIndices.begin(), jointIndices.end(), index) != jointIndices.end())
        {
          ROS_ERROR_STREAM("[AcController] Duplicate " << label << " joint " << name);
          return false;
        }
        jointIndices.push_back(index);
      }

      if (metadataScale.size() != static_cast<size_t>(actionCount) &&
          metadataScale.size() != jointNames_.size())
      {
        ROS_ERROR_STREAM("[AcController] " << label
                         << " action_scale must contain " << actionCount
                         << " or " << jointNames_.size() << " values");
        return false;
      }
      const std::vector<double> rawScale(metadataScale);
      compactScale.resize(actionCount);
      compactDefaults.resize(actionCount);
      for (int i = 0; i < actionCount; ++i)
      {
        const int jointIndex = jointIndices[i];
        compactDefaults(i) = metadataDefaults[jointIndex];
        compactScale[i] = rawScale.size() == jointNames_.size()
                              ? rawScale[jointIndex]
                              : rawScale[i];
      }

      fixedJointIndices.clear();
      for (size_t i = 0; i < jointNames_.size(); ++i)
      {
        if (std::find(jointIndices.begin(), jointIndices.end(), static_cast<int>(i)) ==
            jointIndices.end())
        {
          fixedJointIndices.push_back(static_cast<int>(i));
        }
      }
      if (fixedJointIndices.size() != 9)
      {
        ROS_ERROR_STREAM("[AcController] " << label
                         << " must leave exactly 9 fixed joints");
        return false;
      }
      return true;
    };

    if (!configureSubset(
            "/LeggedRobotCfg/joint_names_12dof_walk", actions12DOFSize_,
            action_scale_12DOF, default_joint_pos_12DOF,
            jointIndices12DOF_, fixedUpperBodyJointIndices12DOF_,
            defaultJointAngles12DOF_, action_scale_12DOF, "12-DOF walk") ||
        !configureSubset(
            "/LeggedRobotCfg/joint_names_13dof_jump", actions13DOFJumpSize_,
            action_scale_13DOF_jump, default_joint_pos_13DOF_jump,
            jointIndices13DOFJump_, fixedUpperBodyJointIndices13DOFJump_,
            defaultJointAngles13DOFJump_, action_scale_13DOF_jump,
            "13-DOF jump"))
    {
      return false;
    }

    actions_.assign(actionsSize_, 0.0f);
    actions12DOF_.assign(actions12DOFSize_, 0.0f);
    actions13DOFJump_.assign(actions13DOFJumpSize_, 0.0f);
    policyObservations_.assign(fullInputSize, 0.0f);
    policyObservations12DOF_.assign(walkInputSize, 0.0f);
    policyObservations13DOFJump_.assign(jumpInputSize, 0.0f);

    proprioHistoryBuffer_.resize(fullInputSize);
    proprioHistoryBuffer12DOF_.resize(walkInputSize);
    proprioHistoryBuffer13DOFJump_.resize(jumpInputSize);
    proprioHistoryBuffer_.setZero();
    proprioHistoryBuffer12DOF_.setZero();
    proprioHistoryBuffer13DOFJump_.setZero();

    defaultJointAngles_.resize(actionsSize_);
    for (int i = 0; i < actionsSize_; ++i)
    {
      defaultJointAngles_(i) = default_joint_pos[i];
    }
    lastActions_.resize(actionsSize_);
    lastActions_.setZero();
    command_.x = 0.0;
    command_.y = 0.0;
    command_.yaw = 0.0;

    ROS_INFO_STREAM("[AcController] policy observations: 21-DOF="
                    << observationSize_ << "x" << stackSize_
                    << ", 12-DOF walk=" << observation12DOFSize_
                    << "x" << stackSize_
                    << ", 13-DOF jump entry=" << observation13DOFJumpSize_
                    << "x" << stackSize_);
    return true;
  }

  void AcController::computeActionsDance()
  {

    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, policyObservationsDance_.data(), policyObservationsDance_.size(),
                                                                         policyInputShapesDance_[0].data(), policyInputShapesDance_[0].size()));
    // run inference
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues;
    outputValues = policySessionPtrDance_->Run(runOptions, policyInputNamesDance_.data(), policyInputValues.data(), 1, policyOutputNamesDance_.data(), 1);

    if (isfirstCompAct_){
      // for (int i = 0; i < policyObservations_.size(); ++i) {
      //   std::cout << policyObservations_[i] << " ";
      //   if ((i + 1) % observationSize_ == 0) {
      //       std::cout << std::endl;
      //   }
      // }
      isfirstCompAct_ = false;
    }

    for (int i = 0; i < actionsSize_; i++)
    {
      actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }
  }

  void AcController::computeActions()
  {

    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, policyObservations_.data(), policyObservations_.size(),
                                                                         policyInputShapes_[0].data(), policyInputShapes_[0].size()));
    // run inference
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues;
    outputValues = policySessionPtr_->Run(runOptions, policyInputNames_.data(), policyInputValues.data(), 1, policyOutputNames_.data(), 1);

    if (isfirstCompAct_){
      // for (int i = 0; i < policyObservations_.size(); ++i) {
      //   std::cout << policyObservations_[i] << " ";
      //   if ((i + 1) % observationSize_ == 0) {
      //       std::cout << std::endl;
      //   }
      // }
      isfirstCompAct_ = false;
    }

    for (int i = 0; i < actionsSize_; i++)
    {
      actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }

  }

  void AcController::computeObservation()
  {
    // Match the training command generator: forward/backward, lateral motion,
    // and in-place turning are mutually exclusive jump modes.
    constexpr scalar_t commandThreshold = 0.2;
    const scalar_t commandX = command_.x.load();
    const scalar_t commandY = command_.y.load();
    const scalar_t commandYaw = command_.yaw.load();
    const scalar_t xMagnitude = std::abs(commandX);
    const scalar_t yMagnitude = std::abs(commandY);
    const scalar_t yawMagnitude = std::abs(commandYaw);

    vector_t command(3);
    command.setZero();

    // A joystick can report more than one active axis. Keep only the dominant
    // axis so small lateral/yaw leakage cannot mix into a forward/back command.
    if (xMagnitude > commandThreshold && xMagnitude >= yMagnitude && xMagnitude >= yawMagnitude)
    {
      const scalar_t minimum = observationSize_ == 72 ? -1.2 : -0.5;
      const scalar_t maximum = observationSize_ == 72 ? 1.5 : 1.0;
      command(0) = std::max(minimum, std::min(maximum, commandX));
    }
    else if (yMagnitude > commandThreshold && yMagnitude >= yawMagnitude)
    {
      const scalar_t limit = observationSize_ == 72 ? 1.2 : 0.5;
      command(1) = std::max(-limit, std::min(limit, commandY));
    }
    else if (yawMagnitude > commandThreshold)
    {
      const scalar_t limit = observationSize_ == 72 ? 1.5 : 1.0;
      command(2) = std::max(-limit, std::min(limit, commandYaw));
    }

    // actions
    vector_t actions(lastActions_);

    vector_t proprioObs(observationSize_);

    if (observationSize_ == 72)
    {
      proprioObs << command,                           // 3
          propri_.baseAngVel,                          // 3
          propri_.projectedGravity,                    // 3
          (propri_.jointPos - defaultJointAngles_),    // 21
          propri_.jointVel,                            // 21
          actions;                                     // 21
    }
    else
    {
      proprioObs << command,                           // 3
          propri_.baseAngVel,                          // 3
          propri_.baseEulerXyz(0),                     // roll
          propri_.baseEulerXyz(1),                     // pitch
          (propri_.jointPos - defaultJointAngles_),    // 21
          propri_.jointVel,                            // 21
          actions;                                     // 21
    }
    

    if (isfirstRecObs_)
    {
      for (
         int i = observationSize_ - actionsSize_; i < observationSize_; i++)
      {
        proprioObs(i,0) = 0.0;
      }

      for (size_t i = 0; i < stackSize_; i++)
      {
        proprioHistoryBuffer_.segment(i * observationSize_, observationSize_) = proprioObs.cast<tensor_element_t>();
      }
      isfirstRecObs_ = false;
      // std::cout <<"isfirstRecObs_" << isfirstRecObs_<<std::endl;

      std::fill(policyObservations_.begin(), policyObservations_.end(), 0.0f);
      }

    proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - observationSize_) =
        proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - observationSize_);
    proprioHistoryBuffer_.tail(observationSize_) = proprioObs.cast<tensor_element_t>();

    for (size_t i = 0; i < (observationSize_ * stackSize_); i++){
      policyObservations_[i] = static_cast<tensor_element_t>(proprioHistoryBuffer_[i]);
    }

    scalar_t obsMin = -robotCfg_.clipObs;
    scalar_t obsMax = robotCfg_.clipObs;
    std::transform(policyObservations_.begin(), policyObservations_.end(), policyObservations_.begin(),
                   [obsMin, obsMax](scalar_t x)
                   { return std::max(obsMin, std::min(obsMax, x)); });

  }

void AcController::transformBaseOriToTorsoOri(const Eigen::Quaterniond& base_quat, const std::vector<double>& waist_joint_angles, Eigen::Quaterniond& torso_quat){
    Eigen::AngleAxisd rollAngle1(0, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle1(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle1(waist_joint_angles[0], Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d transform_rot_mat_1 = Eigen::Matrix3d::Identity();
    transform_rot_mat_1 = (yawAngle1 * pitchAngle1 * rollAngle1).toRotationMatrix();

    Eigen::AngleAxisd rollAngle2(waist_joint_angles[1], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle2(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle2(0, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d transform_rot_mat_2 = Eigen::Matrix3d::Identity();
    transform_rot_mat_2= (yawAngle2 * pitchAngle2 * rollAngle2).toRotationMatrix();

    Eigen::Matrix3d transform_rot_mat = transform_rot_mat_1 * transform_rot_mat_2;

    Eigen::Matrix3d torso_rot_mat = base_quat.toRotationMatrix() * transform_rot_mat;

    torso_quat = matrix_to_quaternion_eigen<double>(torso_rot_mat);
}

void AcController::computeObservationDance()
  {

    danceTimeStep += 1;
    if (danceTimeStep >= num_timesteps - 10){
        mode_ = Mode::WALK;
        ROS_INFO("other2WALK");
        isfirstRecObs_ = true;
        danceTimeStep = 0;
    }
    vector_t ref_dof_pos_(actionsSize_);
    vector_t ref_dof_vel_(actionsSize_);

    // 使用 Eigen::Map 将 std::vector<double> 转换为 Eigen 向量
    ref_dof_pos_ = Eigen::Map<const Eigen::VectorXd>(
        ref_joint_pos[danceTimeStep].data(), 
        ref_joint_pos[danceTimeStep].size()
    );

    ref_dof_vel_ = Eigen::Map<const Eigen::VectorXd>(
        ref_joint_vel[danceTimeStep].data(), 
        ref_joint_vel[danceTimeStep].size()
    );
    const auto& quat_data = ref_quat[danceTimeStep];
    const auto& ref_robot_quat = Eigen::Quaterniond(quat_data[0], quat_data[1], quat_data[2], quat_data[3]);
    std::vector<double> waist_joint_angles = {propri_.jointPos(2), 0.0}; // 假设前两个是腰部关节角度

    transformBaseOriToTorsoOri(propri_.robot_quat_, waist_joint_angles, propri_.robot_quat_);


    const auto& robot_quat = Eigen::Quaterniond(
      propri_.robot_quat_.w(), 
      propri_.robot_quat_.x(), 
      propri_.robot_quat_.y(),
      propri_.robot_quat_.z());
    if (danceTimeStep < 2){
    // 1. 获取参考运动四元数并提取偏航
    Eigen::Quaterniond yaw_motion_quat = yaw_quat<double>(ref_robot_quat);
    // 2. 将偏航四元数转换为旋转矩阵
      Eigen::Matrix3d yaw_motion_matrix = yaw_motion_quat.toRotationMatrix();
    // 3. 获取机器人当前方向并提取偏航
    Eigen::Quaterniond yaw_robot_quat = yaw_quat<double>(robot_quat);
    // 4. 将机器人偏航四元数转换为旋转矩阵
    Eigen::Matrix3d yaw_robot_matrix = yaw_robot_quat.toRotationMatrix();
    // 5. 计算变换矩阵：yaw_robot_matrix * yaw_motion_matrix^T
    init_to_world = matrix_to_quaternion_eigen<double>(yaw_robot_matrix * yaw_motion_matrix.transpose());
    }

    // 计算四元数差值
    Eigen::Quaterniond delta_q = subtract_frame_transforms<double>(robot_quat, ref_robot_quat, init_to_world);
  
    // 转换为旋转矩阵
    Eigen::Matrix3d rotation_mat = QuatToMat<double>(delta_q);
    // actions
    vector_t actions(lastActions_);

    vector_t proprioObs(observationSizeDance_);
    vector_t motion_anchor_ori_b(6);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            motion_anchor_ori_b(i * 2 + j) = rotation_mat(i, j);
        }
    }

    std::cout<<"danceTimeStep: "<< danceTimeStep<< std::endl;

    proprioObs << 
        ref_dof_pos_, // 3
        ref_dof_vel_, // 3
        motion_anchor_ori_b,
        propri_.baseAngVel,  // 3
        // rotation_mat.col(0),
        // rotation_mat.col(1),
        // propri_.projectedGravity(0),  // 1
        // propri_.projectedGravity(1),  // 1
        // propri_.projectedGravity(2),  // 1
        (propri_.jointPos - defaultJointAnglesDance_),  // 19
        propri_.jointVel,  // 19
        actions;  // 19
    

    if (isfirstRecObs_)
    {
      for (
         int i = observationSizeDance_ - actionsSize_; i < observationSizeDance_; i++)
      {
        proprioObs(i,0) = 0.0;
      }

      for (size_t i = 0; i < stackSize_; i++)
      {
        proprioHistoryBufferDance_.segment(i * observationSizeDance_, observationSizeDance_) = proprioObs.cast<tensor_element_t>();
      }
      isfirstRecObs_ = false;
      // std::cout <<"isfirstRecObs_" << isfirstRecObs_<<std::endl;

      std::fill(policyObservationsDance_.begin(), policyObservationsDance_.end(), 0.0f);
      }

    proprioHistoryBufferDance_.head(proprioHistoryBufferDance_.size() - observationSizeDance_) =
        proprioHistoryBufferDance_.tail(proprioHistoryBufferDance_.size() - observationSizeDance_);
    proprioHistoryBufferDance_.tail(observationSizeDance_) = proprioObs.cast<tensor_element_t>();

    for (size_t i = 0; i < (observationSizeDance_ * stackSize_); i++){
      policyObservationsDance_[i] = static_cast<tensor_element_t>(proprioHistoryBufferDance_[i]);
    }

    scalar_t obsMin = -robotCfg_.clipObs;
    scalar_t obsMax = robotCfg_.clipObs;
    std::transform(policyObservationsDance_.begin(), policyObservationsDance_.end(), policyObservationsDance_.begin(),
                   [obsMin, obsMax](scalar_t x)
                   { return std::max(obsMin, std::min(obsMax, x)); });

  }
  void AcController::computeActions12DOF()
  {
    std::vector<Ort::Value> inputValues;
    inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo,
        policyObservations12DOF_.data(),
        policyObservations12DOF_.size(),
        policyInputShapes12DOF_.front().data(),
        policyInputShapes12DOF_.front().size()));

    Ort::RunOptions runOptions;
    const auto outputValues = policySessionPtr12DOF_->Run(
        runOptions,
        policyInputNames12DOF_.data(), inputValues.data(), 1,
        policyOutputNames12DOF_.data(), 1);
    const auto *output = outputValues.front().GetTensorData<tensor_element_t>();
    for (int i = 0; i < actions12DOFSize_; ++i)
    {
      actions12DOF_[i] = output[i];
    }
    isfirstComp12DOFAct_ = false;
  }

  void AcController::computeObservation12DOF()
  {
    // 12-DOF walking CFG: Euler roll/pitch and walking command limits.
    constexpr scalar_t commandThreshold = 0.2;
    const scalar_t commandX = command_.x.load();
    const scalar_t commandY = command_.y.load();
    const scalar_t commandYaw = command_.yaw.load();
    const scalar_t xMagnitude = std::abs(commandX);
    const scalar_t yMagnitude = std::abs(commandY);
    const scalar_t yawMagnitude = std::abs(commandYaw);

    vector_t command(3);
    command.setZero();
    if (xMagnitude > commandThreshold && xMagnitude >= yMagnitude && xMagnitude >= yawMagnitude)
    {
      command(0) = std::max<scalar_t>(-0.5, std::min<scalar_t>(1.0, commandX));
    }
    else if (yMagnitude > commandThreshold && yMagnitude >= yawMagnitude)
    {
      command(1) = std::max<scalar_t>(-0.5, std::min<scalar_t>(0.5, commandY));
    }
    else if (yawMagnitude > commandThreshold)
    {
      command(2) = std::max<scalar_t>(-1.0, std::min<scalar_t>(1.0, commandYaw));
    }

    vector_t jointPosition(actions12DOFSize_);
    vector_t jointVelocity(actions12DOFSize_);
    vector_t previousActions(actions12DOFSize_);
    for (int i = 0; i < actions12DOFSize_; ++i)
    {
      const int jointIndex = jointIndices12DOF_[i];
      jointPosition(i) = propri_.jointPos(jointIndex);
      jointVelocity(i) = propri_.jointVel(jointIndex);
      previousActions(i) = lastActions_(jointIndex);
    }

    vector_t observation(observation12DOFSize_);
    observation << command,                                // 3
                   propri_.baseAngVel,                      // 3
                   propri_.baseEulerXyz(0),                 // roll
                   propri_.baseEulerXyz(1),                 // pitch
                   jointPosition - defaultJointAngles12DOF_,// 12
                   jointVelocity,                           // 12
                   previousActions;                         // 12

    if (isfirstRec12DOFObs_)
    {
      observation.tail(actions12DOFSize_).setZero();
      for (int historyIndex = 0; historyIndex < stackSize_; ++historyIndex)
      {
        proprioHistoryBuffer12DOF_.segment(
            historyIndex * observation12DOFSize_, observation12DOFSize_) =
            observation.cast<tensor_element_t>();
      }
      isfirstRec12DOFObs_ = false;
    }

    proprioHistoryBuffer12DOF_.head(
        proprioHistoryBuffer12DOF_.size() - observation12DOFSize_) =
        proprioHistoryBuffer12DOF_.tail(
            proprioHistoryBuffer12DOF_.size() - observation12DOFSize_);
    proprioHistoryBuffer12DOF_.tail(observation12DOFSize_) =
        observation.cast<tensor_element_t>();

    for (size_t i = 0; i < policyObservations12DOF_.size(); ++i)
    {
      policyObservations12DOF_[i] = proprioHistoryBuffer12DOF_(i);
    }

    const scalar_t observationMin = -robotCfg_.clipObs;
    const scalar_t observationMax = robotCfg_.clipObs;
    std::transform(policyObservations12DOF_.begin(), policyObservations12DOF_.end(),
                   policyObservations12DOF_.begin(),
                   [observationMin, observationMax](scalar_t value)
                   { return std::max(observationMin, std::min(observationMax, value)); });
  }

  void AcController::computeActions13DOFJump()
  {
    std::vector<Ort::Value> inputValues;
    inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo,
        policyObservations13DOFJump_.data(),
        policyObservations13DOFJump_.size(),
        policyInputShapes13DOFJump_.front().data(),
        policyInputShapes13DOFJump_.front().size()));

    Ort::RunOptions runOptions;
    const auto outputValues = policySessionPtr13DOFJump_->Run(
        runOptions,
        policyInputNames13DOFJump_.data(), inputValues.data(), 1,
        policyOutputNames13DOFJump_.data(), 1);
    const auto *output = outputValues.front().GetTensorData<tensor_element_t>();
    for (int i = 0; i < actions13DOFJumpSize_; ++i)
    {
      actions13DOFJump_[i] = output[i];
    }
    isfirstComp13DOFJumpAct_ = false;
  }

  void AcController::computeObservation13DOFJump()
  {
    constexpr scalar_t commandThreshold = 0.2;
    const scalar_t commandX = command_.x.load();
    const scalar_t commandY = command_.y.load();
    const scalar_t commandYaw = command_.yaw.load();
    const scalar_t xMagnitude = std::abs(commandX);
    const scalar_t yMagnitude = std::abs(commandY);
    const scalar_t yawMagnitude = std::abs(commandYaw);

    vector_t command(3);
    command.setZero();
    if (xMagnitude > commandThreshold && xMagnitude >= yMagnitude && xMagnitude >= yawMagnitude)
    {
      command(0) = std::max<scalar_t>(-1.2, std::min<scalar_t>(1.5, commandX));
    }
    else if (yMagnitude > commandThreshold && yMagnitude >= yawMagnitude)
    {
      command(1) = std::max<scalar_t>(-1.2, std::min<scalar_t>(1.2, commandY));
    }
    else if (yawMagnitude > commandThreshold)
    {
      command(2) = std::max<scalar_t>(-1.5, std::min<scalar_t>(1.5, commandYaw));
    }

    vector_t jointPosition(actions13DOFJumpSize_);
    vector_t jointVelocity(actions13DOFJumpSize_);
    vector_t previousActions(actions13DOFJumpSize_);
    for (int i = 0; i < actions13DOFJumpSize_; ++i)
    {
      const int jointIndex = jointIndices13DOFJump_[i];
      jointPosition(i) = propri_.jointPos(jointIndex);
      jointVelocity(i) = propri_.jointVel(jointIndex);
      previousActions(i) = lastActions_(jointIndex);
    }

    vector_t observation(observation13DOFJumpSize_);
    observation << command,                                      // 3
                   propri_.baseAngVel,                            // 3
                   propri_.projectedGravity,                      // 3
                   jointPosition - defaultJointAngles13DOFJump_,  // 12
                   jointVelocity,                                 // 12
                   previousActions;                               // 12

    if (isfirstRec13DOFJumpObs_)
    {
      observation.tail(actions13DOFJumpSize_).setZero();
      for (int historyIndex = 0; historyIndex < stackSize_; ++historyIndex)
      {
        proprioHistoryBuffer13DOFJump_.segment(
            historyIndex * observation13DOFJumpSize_,
            observation13DOFJumpSize_) = observation.cast<tensor_element_t>();
      }
      isfirstRec13DOFJumpObs_ = false;
    }

    proprioHistoryBuffer13DOFJump_.head(
        proprioHistoryBuffer13DOFJump_.size() - observation13DOFJumpSize_) =
        proprioHistoryBuffer13DOFJump_.tail(
            proprioHistoryBuffer13DOFJump_.size() - observation13DOFJumpSize_);
    proprioHistoryBuffer13DOFJump_.tail(observation13DOFJumpSize_) =
        observation.cast<tensor_element_t>();

    for (size_t i = 0; i < policyObservations13DOFJump_.size(); ++i)
    {
      policyObservations13DOFJump_[i] = proprioHistoryBuffer13DOFJump_(i);
    }

    const scalar_t observationMin = -robotCfg_.clipObs;
    const scalar_t observationMax = robotCfg_.clipObs;
    std::transform(
        policyObservations13DOFJump_.begin(), policyObservations13DOFJump_.end(),
        policyObservations13DOFJump_.begin(),
        [observationMin, observationMax](scalar_t value)
        { return std::max(observationMin, std::min(observationMax, value)); });
  }

} // namespace legged


PLUGINLIB_EXPORT_CLASS(legged::AcController, controller_interface::ControllerBase)
