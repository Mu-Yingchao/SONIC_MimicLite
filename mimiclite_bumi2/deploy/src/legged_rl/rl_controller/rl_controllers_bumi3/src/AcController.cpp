#include "rl_controllers_bumi3/AcController.h"
#include <pluginlib/class_list_macros.hpp>
#include "rl_controllers_bumi3/RotationTools.h"
#include <algorithm>
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

      computeObservation();
      computeActions();
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
      scalar_t pos_des = actions_[i] * action_scale[i] + defaultJointAngles_(i);
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

  void AcController::handleWalk13DOFMode()
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
      computeObservation13DOF();
      computeActions13DOF();
      // limit action range
      scalar_t actionMin = -robotCfg_.clipActions;
      scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions13DOF_.begin(), actions13DOF_.end(), actions13DOF_.begin(),
                     [actionMin, actionMax](scalar_t x)
                     { return std::max(actionMin, std::min(actionMax, x)); });
    }

    // set action
    for (int j = 0; j < actions13DOFSize_; j++)
    {
      int jointIdx = jointIndices13DOF_[j];
      // std::string partName = hybridJointHandles_[jointIdx].getName();
      scalar_t action_value = actions13DOF_[j] * action_scale_13DOF[jointIdx];
      scalar_t pos_des = action_value + defaultJointAngles13DOF_(j);
      // // l leg ankle pitch
      // if (j==9){
      //   hybridJointHandles_[jointIdx].setCommand(
      //       pos_des,
      //       0,
      //       30,
      //       joint_damping_13DOF[jointIdx],
      //       0);
      // }
      // // r leg ankle pitch
      // else if (j==10){
      //   hybridJointHandles_[jointIdx].setCommand(
      //       pos_des,
      //       0,
      //       30,
      //       joint_damping_13DOF[jointIdx],
      //       0);
      // }
      // // l leg knee
      // else if (j==6){
      //   hybridJointHandles_[jointIdx].setCommand(
      //       pos_des,
      //       0,
      //       80,
      //       joint_damping_13DOF[jointIdx],
      //       0);
      // }
      // // r leg knee
      // else if (j==7){
      //   hybridJointHandles_[jointIdx].setCommand(
      //       pos_des,
      //       0,
      //       80,
      //       joint_damping_13DOF[jointIdx],
      //       0);
      // }
      // else{
      //   hybridJointHandles_[jointIdx].setCommand(
      //       pos_des,
      //       0,
      //       joint_stiffness_13DOF[jointIdx],
      //       joint_damping_13DOF[jointIdx],
      //       0);

      // }

      hybridJointHandles_[jointIdx].setCommand(
          pos_des,
          0,
          joint_stiffness_13DOF[jointIdx],
          joint_damping_13DOF[jointIdx],
          0);

      lastActions_(jointIdx, 0) = actions13DOF_[j];
    }
    // waist
    // hybridJointHandles_[2].setCommand(
    // 0 + default_joint_pos_13DOF[2],
    // 0,
    // joint_stiffness_13DOF[2],
    // joint_damping_13DOF[2],
    // 0);
    // // l arm shoulder pitch
    // hybridJointHandles_[5].setCommand(
    // -1.75 + default_joint_pos_13DOF[5],
    // 0,
    // joint_stiffness_13DOF[5],
    // joint_damping_13DOF[5],
    // 0);
    // // r arm shoulder pitch
    // hybridJointHandles_[6].setCommand(
    // -1.75 + default_joint_pos_13DOF[6],
    // 0,
    // joint_stiffness_13DOF[6],
    // joint_damping_13DOF[6],
    // 0);
    // // l arm shoulder roll
    // hybridJointHandles_[9].setCommand(
    // -0.3 + default_joint_pos_13DOF[9],
    // 0,
    // joint_stiffness_13DOF[9],
    // joint_damping_13DOF[9],
    // 0);
    // // r arm shoulder roll
    // hybridJointHandles_[10].setCommand(
    // 0 + default_joint_pos_13DOF[10],
    // 0,
    // joint_stiffness_13DOF[10],
    // joint_damping_13DOF[10],
    // 0);

    // // l arm shoulder yaw
    // hybridJointHandles_[14].setCommand(
    // 1.75 + default_joint_pos_13DOF[14],
    // 0,
    // joint_stiffness_13DOF[14],
    // joint_damping_13DOF[14],
    // 0);
    // // l arm elbow
    // hybridJointHandles_[17].setCommand(
    // -1.75 + default_joint_pos_13DOF[17],
    // 0,
    // joint_stiffness_13DOF[17],
    // joint_damping_13DOF[17],
    // 0);
    // // r arm elbow
    // hybridJointHandles_[18].setCommand(
    // -1.75 + default_joint_pos_13DOF[18],
    // 0,
    // joint_stiffness_13DOF[18],
    // joint_damping_13DOF[18],
    // 0);
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
    std::string policyFilePath;
    std::string policyFilePathDance; 
    std::string policyFile13DOF;
    if (!nh.getParam("/policyFile", policyFilePath))
    {
      ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
      return false;
    }
    policyFilePath_ = policyFilePath;
    ROS_INFO_STREAM("Load Onnx model from path : " << policyFilePath);

    if (!nh.getParam("/policyFileDance", policyFilePathDance))
    {
      ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
      return false;
    }
    policyFilePathDance_ = policyFilePathDance;
    ROS_INFO_STREAM("Load Onnx model from path : " << policyFilePathDance);

    if (!nh.getParam("/policyFile13DOF", policyFile13DOF))
    {
      ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
      return false;
    }
    policyFilePath13DOF_ = policyFile13DOF;
    ROS_INFO_STREAM("Load Onnx model from path : " << policyFile13DOF);

    // create env
    onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
        
    // create session
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(1);
    policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFilePath.c_str(), sessionOptions);
    policySessionPtrDance_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFilePathDance.c_str(), sessionOptions);
    policySessionPtr13DOF_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFile13DOF.c_str(), sessionOptions);

    // get input and output info
    policyInputNames_.clear();
    policyOutputNames_.clear();    
    policyInputShapes_.clear();
    policyOutputShapes_.clear();

    policyInputNamesDance_.clear();
    policyOutputNamesDance_.clear();
    policyInputShapesDance_.clear();
    policyOutputShapesDance_.clear();

    policyInputNames13DOF_.clear();
    policyOutputNames13DOF_.clear();
    policyInputShapes13DOF_.clear();
    policyOutputShapes13DOF_.clear();


    Ort::AllocatorWithDefaultOptions allocator;

    Ort::ModelMetadata metadata = policySessionPtr_->GetModelMetadata();
    Ort::ModelMetadata metadataDance = policySessionPtrDance_->GetModelMetadata();
    Ort::ModelMetadata metadata13DOF = policySessionPtr13DOF_->GetModelMetadata();

    auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    for (size_t i = 0ul; i < keys.size(); i++) {
        auto value = metadata.LookupCustomMetadataMapAllocated(keys[i].get(), allocator);
        std::string key = std::string(keys[i].get());

        if (key == "joint_names") {
            found_joint_names = true;
            std::string joint_names_str = std::string(value.get());
            auto Joint_Names = SplitString(joint_names_str, ',');
            joint_names.resize(Joint_Names.size());
            for (size_t j = 0; j < Joint_Names.size(); j++){
              joint_names[j] = Joint_Names[j];
            }
        }

        if (key == "joint_stiffness") {
            found_joint_stiffness = true;
            std::string joint_stiffness_str = std::string(value.get());
            auto Joint_Stiffness = SplitString2Float(joint_stiffness_str, ',');
            joint_stiffness.resize(Joint_Stiffness.size());
            for (size_t j = 0; j < Joint_Stiffness.size(); j++){
              joint_stiffness[j] = Joint_Stiffness[j];
            }
        }

        if (key == "joint_damping") {
            found_joint_damping = true;
            std::string joint_damping_str = std::string(value.get());
            auto Joint_Damping = SplitString2Float(joint_damping_str, ',');
            joint_damping.resize(Joint_Damping.size());
            for (size_t j = 0; j < Joint_Damping.size(); j++){
              joint_damping[j] = Joint_Damping[j];
            }
        }

        if (key == "default_joint_pos") {
            found_default_joint_pos = true;
            std::string default_joint_pos_str = std::string(value.get());
            auto Default_Joint_Pos = SplitString2Float(default_joint_pos_str, ',');
            default_joint_pos.resize(Default_Joint_Pos.size());
            for (size_t j = 0; j < Default_Joint_Pos.size(); j++){
              default_joint_pos[j] = Default_Joint_Pos[j];
            }
        }

        if (key == "action_scale") {
            found_action_scale = true;
            std::string action_scale_str = std::string(value.get());
            auto Action_Scales = SplitString2Float(action_scale_str, ',');
            action_scale.resize(Action_Scales.size());
            for (size_t j = 0; j < Action_Scales.size(); j++){
              action_scale[j] = Action_Scales[j];
            }
        }

    }

    keys = metadataDance.GetCustomMetadataMapKeysAllocated(allocator);
    for (size_t i = 0ul; i < keys.size(); i++) {
        auto value = metadataDance.LookupCustomMetadataMapAllocated(keys[i].get(), allocator);
        std::string key = std::string(keys[i].get());

        if (key == "joint_names") {
            found_joint_names = true;
            std::string joint_names_str = std::string(value.get());
            auto Joint_Names = SplitString(joint_names_str, ',');
            joint_names_dance.resize(Joint_Names.size());
            for (size_t j = 0; j < Joint_Names.size(); j++){
              joint_names_dance[j] = Joint_Names[j];
            }
        }

        if (key == "joint_stiffness") {
            found_joint_stiffness = true;
            std::string joint_stiffness_str = std::string(value.get());
            auto Joint_Stiffness = SplitString2Float(joint_stiffness_str, ',');
            joint_stiffness_dance.resize(Joint_Stiffness.size());
            for (size_t j = 0; j < Joint_Stiffness.size(); j++){
              joint_stiffness_dance[j] = Joint_Stiffness[j];
            }
        }

        if (key == "joint_damping") {
            found_joint_damping = true;
            std::string joint_damping_str = std::string(value.get());
            auto Joint_Damping = SplitString2Float(joint_damping_str, ',');
            joint_damping_dance.resize(Joint_Damping.size());
            for (size_t j = 0; j < Joint_Damping.size(); j++){
              joint_damping_dance[j] = Joint_Damping[j];
            }
        }

        if (key == "default_joint_pos") {
            found_default_joint_pos = true;
            std::string default_joint_pos_str = std::string(value.get());
            auto Default_Joint_Pos = SplitString2Float(default_joint_pos_str, ',');
            default_joint_pos_dance.resize(Default_Joint_Pos.size());
            for (size_t j = 0; j < Default_Joint_Pos.size(); j++){
              default_joint_pos_dance[j] = Default_Joint_Pos[j];
            }
        }

        if (key == "action_scale") {
            found_action_scale = true;
            std::string action_scale_str = std::string(value.get());
            auto Action_Scales = SplitString2Float(action_scale_str, ',');
            action_scale_dance.resize(Action_Scales.size());
            for (size_t j = 0; j < Action_Scales.size(); j++){
              action_scale_dance[j] = Action_Scales[j];
            }
        }

    }

    keys = metadata13DOF.GetCustomMetadataMapKeysAllocated(allocator);
    for (size_t i = 0ul; i < keys.size(); i++) {
      auto value = metadata13DOF.LookupCustomMetadataMapAllocated(keys[i].get(), allocator);
      std::string key = std::string(keys[i].get());
      if (key == "joint_names") {
        found_joint_names = true;
        std::string joint_names_str = std::string(value.get());
        auto Joint_Names = SplitString(joint_names_str, ',');
        joint_names_13DOF.resize(Joint_Names.size());
        for (size_t j = 0; j < Joint_Names.size(); j++){
          joint_names_13DOF[j] = Joint_Names[j];
        }
      }

      if (key == "joint_stiffness") {
        found_joint_stiffness = true;
        std::string joint_stiffness_str = std::string(value.get());
        auto Joint_Stiffness = SplitString2Float(joint_stiffness_str, ',');
        joint_stiffness_13DOF.resize(Joint_Stiffness.size());
        for (size_t j = 0; j < Joint_Stiffness.size(); j++){
          joint_stiffness_13DOF[j] = Joint_Stiffness[j];
        }
      }

      if (key == "joint_damping") {
        found_joint_damping = true;
        std::string joint_damping_str = std::string(value.get());
        auto Joint_Damping = SplitString2Float(joint_damping_str, ',');
        joint_damping_13DOF.resize(Joint_Damping.size());
        for (size_t j = 0; j < Joint_Damping.size(); j++){
          joint_damping_13DOF[j] = Joint_Damping[j];
        }
      }

      if (key == "default_joint_pos") {
        found_default_joint_pos = true;
        std::string default_joint_pos_str = std::string(value.get());
        auto Default_Joint_Pos = SplitString2Float(default_joint_pos_str, ',');
        default_joint_pos_13DOF.resize(Default_Joint_Pos.size());
        for (size_t j = 0; j < Default_Joint_Pos.size(); j++){
          default_joint_pos_13DOF[j] = Default_Joint_Pos[j];
        }
      }

      if (key == "action_scale") {
        found_action_scale = true;
        std::string action_scale_str = std::string(value.get());
        auto Action_Scales = SplitString2Float(action_scale_str, ',');
        action_scale_13DOF.resize(Action_Scales.size());
        for (size_t j = 0; j < Action_Scales.size(); j++){
          action_scale_13DOF[j] = Action_Scales[j];
        }
      }
    }

    if (!found_joint_names) ROS_ERROR("Missing required metadata key: 'joint_names'");
    if (!found_joint_stiffness) ROS_ERROR("Missing required metadata key: 'joint_stiffness'");
    if (!found_joint_damping) ROS_ERROR("Missing required metadata key: 'joint_damping'");
    if (!found_default_joint_pos) ROS_ERROR("Missing required metadata key: 'default_joint_pos'");
    if (!found_action_scale) ROS_ERROR("Missing required metadata key: 'action_scale'");

    for (size_t j = 0; j < joint_names.size(); ++j) {
        ROS_INFO_STREAM(std::endl << "Joint Name: " << joint_names[j] << std::endl
          << " Joint Stiffness: " << joint_stiffness[j] << std::endl
          << " Joint Damping: " << joint_damping[j] << std::endl
          << " Default Joint Pos: " << default_joint_pos[j] << std::endl
          << " Action Scale: " << action_scale[j]);
    }

    for (size_t j = 0; j < joint_names_13DOF.size(); ++j) {
        ROS_INFO_STREAM(std::endl << "Joint Name 13DOF: " << joint_names_13DOF[j] << std::endl
          << " Joint Stiffness 13DOF: " << joint_stiffness_13DOF[j] << std::endl
          << " Joint Damping 13DOF: " << joint_damping_13DOF[j] << std::endl
          << " Default Joint Pos 13DOF: " << default_joint_pos_13DOF[j] << std::endl
          << " Action Scale 13DOF: " << action_scale_13DOF[j]);
    }

    ROS_INFO_STREAM("count: " << policySessionPtr_->GetOutputCount());

// -------------------------------- Walk --------------------------------
    for (int i = 0; i < policySessionPtr_->GetInputCount(); i++) {
      auto policyInputnamePtr = policySessionPtr_->GetInputNameAllocated(i, allocator);
      policyInputNodeNameAllocatedStrings.push_back(std::move(policyInputnamePtr));
      policyInputNames_.push_back(policyInputNodeNameAllocatedStrings.back().get());
      policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }

    for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++)
    {
      auto policyOutputnamePtr = policySessionPtr_->GetOutputNameAllocated(i, allocator);
      policyOutputNodeNameAllocatedStrings.push_back(std::move(policyOutputnamePtr));
      policyOutputNames_.push_back(policyOutputNodeNameAllocatedStrings.back().get());
      std::cout << policySessionPtr_->GetOutputNameAllocated(i, allocator).get() << std::endl;
      policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    // -------------------------------- Dance --------------------------------
    for (int i = 0; i < policySessionPtrDance_->GetInputCount(); i++) {
      auto policyInputnamePtr = policySessionPtrDance_->GetInputNameAllocated(i, allocator);
      policyInputNodeNameAllocatedStringsDance.push_back(std::move(policyInputnamePtr));
      policyInputNamesDance_.push_back(policyInputNodeNameAllocatedStringsDance.back().get());
      policyInputShapesDance_.push_back(policySessionPtrDance_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtrDance_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }

    for (int i = 0; i < policySessionPtrDance_->GetOutputCount(); i++)
    {
      auto policyOutputnamePtr = policySessionPtrDance_->GetOutputNameAllocated(i, allocator);
      policyOutputNodeNameAllocatedStringsDance.push_back(std::move(policyOutputnamePtr));
      policyOutputNamesDance_.push_back(policyOutputNodeNameAllocatedStringsDance.back().get());
      std::cout << policySessionPtrDance_->GetOutputNameAllocated(i, allocator).get() << std::endl;
      policyOutputShapesDance_.push_back(policySessionPtrDance_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtrDance_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
    
    // -------------------------------- Walk 13DOF --------------------------------
    for (int i = 0; i < policySessionPtr13DOF_->GetInputCount(); i++) {
      auto policyInputnamePtr = policySessionPtr13DOF_->GetInputNameAllocated(i, allocator);
      policyInputNodeNameAllocatedStrings13DOF.push_back(std::move(policyInputnamePtr));
      policyInputNames13DOF_.push_back(policyInputNodeNameAllocatedStrings13DOF.back().get());
      policyInputShapes13DOF_.push_back(policySessionPtr13DOF_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr13DOF_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Walk 13dof Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }

    for (int i = 0; i < policySessionPtr13DOF_->GetOutputCount(); i++)
    {
      auto policyOutputnamePtr = policySessionPtr13DOF_->GetOutputNameAllocated(i, allocator);
      policyOutputNodeNameAllocatedStrings13DOF.push_back(std::move(policyOutputnamePtr));
      policyOutputNames13DOF_.push_back(policyOutputNodeNameAllocatedStrings13DOF.back().get());
      std::cout << policySessionPtr13DOF_->GetOutputNameAllocated(i, allocator).get() << std::endl;
      policyOutputShapes13DOF_.push_back(policySessionPtr13DOF_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr13DOF_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    ROS_INFO_STREAM("Load Onnx model successfully !!!");
    return true;
  }

  bool AcController::loadRLCfg(ros::NodeHandle &nh)
  {
    RLRobotCfg::ControlCfg &controlCfg = robotCfg_.controlCfg;

    int error = 0;

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/decimation", controlCfg.decimation));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationSize_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_dance_size", observationSizeDance_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/stack_size", stackSize_));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_13dof_size", actions13DOFSize_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_13dof_size", observation13DOFSize_));

    actions_.resize(actionsSize_);
    actions13DOF_.resize(actions13DOFSize_);
    policyObservations_.resize(observationSize_ * stackSize_);
    policyObservationsDance_.resize(observationSizeDance_ * stackSize_);
    policyObservations13DOF_.resize(observation13DOFSize_ * stackSize_);

    std::fill(policyObservations_.begin(), policyObservations_.end(), 0.0f);
    std::fill(policyObservationsDance_.begin(), policyObservationsDance_.end(), 0.0f);
    std::fill(policyObservations13DOF_.begin(), policyObservations13DOF_.end(), 0.0f);

    command_.x = 0;
    command_.y = 0;
    command_.yaw = 0;

    lastActions_.resize(actionsSize_);
    lastActions_.setZero();

    const int inputSize = observationSize_ * stackSize_;
    const int inputSizeDance = observationSizeDance_ * stackSize_;
    const int inputSize13DOF = observation13DOFSize_ * stackSize_;
    proprioHistoryBuffer_.resize(inputSize);
    proprioHistoryBufferDance_.resize(inputSizeDance);
    proprioHistoryBuffer13DOF_.resize(inputSize13DOF);
    defaultJointAngles_.resize(actionsSize_);
    
    for (int i = 0; i < actionsSize_; i++)
    {
      defaultJointAngles_(i) = default_joint_pos[i];
    }

    defaultJointAnglesDance_.resize(actionsSize_);
    
    for (int i = 0; i < actionsSize_; i++)
    {
      defaultJointAnglesDance_(i) = default_joint_pos_dance[i];
    }

    defaultJointAngles13DOF_.resize(actions13DOFSize_);
    
    std::vector<std::string> jointNames13DOF;
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/joint_names_13dof", jointNames13DOF));
    if (error != 0 || actionsSize_ != static_cast<int>(jointNames_.size()) ||
        actions13DOFSize_ != 13 || jointNames13DOF.size() != static_cast<size_t>(actions13DOFSize_))
    {
      ROS_ERROR_STREAM("[Bumi3 AcController] Invalid configured dimensions or /LeggedRobotCfg/joint_names_13dof");
      return false;
    }

    jointIndices13DOF_.clear();
    for (const auto &jointName : jointNames13DOF)
    {
      const auto jointIt = std::find(jointNames_.begin(), jointNames_.end(), jointName);
      const int jointIndex = static_cast<int>(std::distance(jointNames_.begin(), jointIt));
      if (jointIt == jointNames_.end() ||
          std::find(jointIndices13DOF_.begin(), jointIndices13DOF_.end(), jointIndex) != jointIndices13DOF_.end())
      {
        ROS_ERROR_STREAM("[Bumi3 AcController] Invalid or duplicate 13-DOF physical joint: " << jointName);
        return false;
      }
      jointIndices13DOF_.push_back(jointIndex);
    }

    if (default_joint_pos_13DOF.size() != static_cast<size_t>(actionsSize_) ||
        action_scale_13DOF.size() != static_cast<size_t>(actionsSize_) ||
        joint_stiffness_13DOF.size() != static_cast<size_t>(actionsSize_) ||
        joint_damping_13DOF.size() != static_cast<size_t>(actionsSize_))
    {
      ROS_ERROR_STREAM("[Bumi3 AcController] 13-DOF policy metadata must provide " << actionsSize_
                       << " full-body defaults, scales, stiffness, and damping values");
      return false;
    }

    for (int i = 0; i < actions13DOFSize_; i++)
    {
      defaultJointAngles13DOF_(i) = default_joint_pos_13DOF[jointIndices13DOF_[i]];
    }

    return (error == 0);
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
    // command
    vector_t command(3);
    // 绝对值小于0.3的都置0
    if (abs(command_.x) < 0.3) command_.x = 0.0;
    if (abs(command_.y) < 0.3) command_.y = 0.0;
    if (abs(command_.yaw) < 0.3) command_.yaw = 0.0;

    // x的command限定在-0.5-0.8
     if (command_.x < -0.0) command_.x = -0.5;
    if (command_.x > current_vel_limit_) command_.x = current_vel_limit_;

    // 速度大于 0.6,限制原地转向的速度
    if (command_.x > 0.6 && command_.yaw > 1.0) command_.yaw = 1.0;
    if (command_.x > 0.6 && command_.yaw < -1.0) command_.yaw = -1.0;

    // slop
    double command_x = command_.x;
    {
      // slop
      double delta = command_.x - lastCmdX_;
      if (delta > cmdXSlop_) delta = cmdXSlop_;
      if (delta < -cmdXSlop_) delta = -cmdXSlop_;

      // update
      command_x = lastCmdX_ + delta;
      lastCmdX_ = command_x;
    }

    command[0] = command_x;
    command[1] = command_.y;
    command[2] = command_.yaw;
    // std::cout << "Bumi walk command: " << command[0] << " " << command[1] << " " << command[2] << std::endl;

    // actions
    vector_t actions(lastActions_);

    vector_t proprioObs(observationSize_);

    proprioObs << command, // 3
        propri_.baseAngVel,  // 3
        propri_.baseEulerXyz(0),     // 1
        propri_.baseEulerXyz(1),     // 1
        (propri_.jointPos - defaultJointAngles_),  // 23
        propri_.jointVel,  // 23
        actions;  // 23
    

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

  void AcController::computeActions13DOF()
  {
    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, policyObservations13DOF_.data(), policyObservations13DOF_.size(),
                                                                         policyInputShapes13DOF_[0].data(), policyInputShapes13DOF_[0].size()));
    // run inference
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues;
    outputValues = policySessionPtr13DOF_->Run(runOptions, policyInputNames13DOF_.data(), policyInputValues.data(), 1, policyOutputNames13DOF_.data(), 1);

    if (isfirstComp13DOFAct_){
      std::cout << "first policyObservations13DOF_: " << std::endl;
      for (int i = 0; i < policyObservations13DOF_.size(); ++i) {
        std::cout << policyObservations13DOF_[i] << " ";
        if ((i + 1) % observation13DOFSize_ == 0) {
            std::cout << std::endl;
        }
      }
      isfirstComp13DOFAct_ = false;
    }

    for (int i = 0; i < actions13DOFSize_; i++)
    {
      actions13DOF_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }
  }

  void AcController::computeObservation13DOF()
  {
    // command
    vector_t command(3);
    // 绝对值小于0.3的都置0
    if (abs(command_.x) < 0.3) command_.x = 0.0;
    if (abs(command_.y) < 0.3) command_.y = 0.0;
    if (abs(command_.yaw) < 0.3) command_.yaw = 0.0;

    // 如果y的command大于0.3，其余command为0
    if (abs(command_.y) > 0.3) {
        command_.x = 0.0;
        command_.yaw = 0.0;
    }

    // 如果x和yaw的command都大于0.3，则y不管如何都置0
    if (abs(command_.x) > 0.3 && abs(command_.yaw) > 0.3) {
        command_.y = 0.0;
    }

    // x的command限定在-0.6-0.8
    if (command_.x < -0.0) command_.x = -0.6;
    if (command_.x > current_vel_limit_) command_.x = current_vel_limit_;
    if (command_.x > 0.8) command_.x = 0.8;

    // 速度大于 0.6,限制原地转向的速度
    if (command_.x > 0.6 && command_.yaw > 1.0) command_.yaw = 1.0;
    if (command_.x > 0.6 && command_.yaw < -1.0) command_.yaw = -1.0;

    command[0] = command_.x;
    command[1] = command_.y;
    command[2] = command_.yaw;
    // std::cout << "Bumi walk command: " << command[0] << " " << command[1] << " " << command[2] << std::endl;

    // 从全量状态中筛取对应索引
    vector_t jointPosSel(jointIndices13DOF_.size());
    vector_t jointVelSel(jointIndices13DOF_.size());
    vector_t actionsSel(jointIndices13DOF_.size());

    for (size_t i = 0; i < jointIndices13DOF_.size(); i++) {
        int idx = jointIndices13DOF_[i];
        jointPosSel[i] = propri_.jointPos[idx];
        jointVelSel[i] = propri_.jointVel[idx];
        actionsSel[i] = lastActions_[idx];
    }

    vector_t proprioObs(observation13DOFSize_);
    proprioObs << command,                         // 3
                   propri_.baseAngVel,              // 3
                   propri_.baseEulerXyz(0),     // 1
                   propri_.baseEulerXyz(1),     // 1
                   (jointPosSel - defaultJointAngles13DOF_), // 13
                   jointVelSel,                     // 13
                   actionsSel;                       // 13
    // TODO
    if (isfirstRec13DOFObs_)
    {
      for (int i = observation13DOFSize_ - actions13DOFSize_; i < observation13DOFSize_; i++)
      {
        proprioObs(i,0) = 0.0;
      }

      for (size_t i = 0; i < stackSize_; i++)
      {
        proprioHistoryBuffer13DOF_.segment(i * observation13DOFSize_, observation13DOFSize_) = proprioObs.cast<tensor_element_t>();
      }
      isfirstRec13DOFObs_ = false;
      // std::cout <<"isfirstRec13DOFObs_" << isfirstRec13DOFObs_<<std::endl;

      std::fill(policyObservations13DOF_.begin(), policyObservations13DOF_.end(), 0.0f);
    }

    proprioHistoryBuffer13DOF_.head(proprioHistoryBuffer13DOF_.size() - observation13DOFSize_) =
        proprioHistoryBuffer13DOF_.tail(proprioHistoryBuffer13DOF_.size() - observation13DOFSize_);
    proprioHistoryBuffer13DOF_.tail(observation13DOFSize_) = proprioObs.cast<tensor_element_t>();

    for (size_t i = 0; i < (observation13DOFSize_ * stackSize_); i++){
      policyObservations13DOF_[i] = static_cast<tensor_element_t>(proprioHistoryBuffer13DOF_[i]);
    }

    scalar_t obsMin = -robotCfg_.clipObs;
    scalar_t obsMax = robotCfg_.clipObs;
    std::transform(policyObservations13DOF_.begin(), policyObservations13DOF_.end(), policyObservations13DOF_.begin(),
                   [obsMin, obsMax](scalar_t x)
                   { return std::max(obsMin, std::min(obsMax, x)); });

  }

} // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::AcController, controller_interface::ControllerBase)
