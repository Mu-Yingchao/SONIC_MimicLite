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

  // ===================== 台架正弦测试（全关节成对扫描）=====================
  // 仅本文件生效，AcController.cpp 未改动。切换见 CMakeLists 第 56 行。
  //
  // 按 kSteps 顺序逐对测试: 每对 缓升2s -> 静置1s -> 5个周期, 约 19.7s。
  // 11 步全部跑完约 3.6 分钟。左右关节同相位驱动, 响应差异即故障特征。
  // 中心取各关节 URDF 限位中点; 幅值 0.3, 行程不够的(踝roll)用 0.15。
  // kp/kd 取 policy metadata, 每步开始时打印到 rosout(自动存日志)。
  namespace
  {
    constexpr bool kBenchEnable = true;       // false 即退回正常策略行走

    // 以下为默认值; 每步开始时会从 ROS 参数 /bench/* 重新读取。
    // 参数文件: rl_controllers/config/bench_params.yaml, 启动时由
    // load_ac_controller.launch 自动加载; 改完 rosparam load 即生效, 不用重编。
    scalar_t gFrequency  = 0.30;              // 频率 [Hz]
    scalar_t gRampTime   = 2.0;               // 缓升到摆动底端 [s]
    scalar_t gSettleTime = 1.0;               // 静置读基线 [s]
    int      gCycles     = 5;                 // 每对测几个周期

    // 关节名 -> yaml 键: 去掉 l_/r_ 前缀和 _joint 后缀, 左右共用一个键。
    // 例如 l_knee_pitch_joint 和 r_knee_pitch_joint 都查 knee_pitch。
    inline std::string gainKey(const std::string &jointName)
    {
      std::string k = jointName;
      if (k.rfind("l_", 0) == 0 || k.rfind("r_", 0) == 0) k = k.substr(2);
      const std::string suffix = "_joint";
      if (k.size() > suffix.size() &&
          k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0)
      {
        k = k.substr(0, k.size() - suffix.size());
      }
      return k;
    }

    // 查 /bench/<kp|kd>/<键>; 没配置或为负则回退到 policy metadata。
    // 只在每步开始调用一次, 结果缓存起来给控制环用。
    inline scalar_t benchGain(const char *which, const std::string &jointName, scalar_t metaVal)
    {
      ros::NodeHandle pnh;
      double v = -1.0;
      pnh.param(std::string("/bench/") + which + "/" + gainKey(jointName), v, -1.0);
      return v >= 0.0 ? static_cast<scalar_t>(v) : metaVal;
    }

    // 从 /bench/* 重新读参数。改 config/bench_params.yaml 后 rosparam load 即生效。
    inline void loadBenchParams()
    {
      ros::NodeHandle pnh;
      pnh.param("/bench/frequency",   gFrequency,  gFrequency);
      pnh.param("/bench/ramp_time",   gRampTime,   gRampTime);
      pnh.param("/bench/settle_time", gSettleTime, gSettleTime);
      pnh.param("/bench/cycles",      gCycles,     gCycles);
    }

    struct BenchStep
    {
      const char *left;
      const char *right;      // "" 表示无配对
      scalar_t    centerL;
      scalar_t    centerR;
      scalar_t    amplitude;
    };

    // 左关节, 右关节, 左中心, 右中心, 幅值   (中心=URDF限位中点)
    const std::vector<BenchStep> kSteps = {
        {"l_leg_pitch_joint",   "r_leg_pitch_joint",    0.0000,  0.0000, 0.30},
        {"l_leg_roll_joint",    "r_leg_roll_joint",     0.4364, -0.4364, 0.30},
        {"l_leg_yaw_joint",     "r_leg_yaw_joint",      0.0000,  0.0000, 0.30},
        {"l_knee_pitch_joint",  "r_knee_pitch_joint",   1.1344,  1.1344, 0.30},
        {"l_ankle_pitch_joint", "r_ankle_pitch_joint",  0.0087,  0.0087, 0.30},
        {"l_ankle_roll_joint",  "r_ankle_roll_joint",   0.0000,  0.0000, 0.15},
        {"l_arm_pitch_joint",   "r_arm_pitch_joint",   -0.7854, -0.7854, 0.30},
        {"l_arm_roll_joint",    "r_arm_roll_joint",     1.0472, -1.0472, 0.30},
        {"l_arm_yaw_joint",     "r_arm_yaw_joint",      0.0000,  0.0000, 0.30},
        {"l_elbow_pitch_joint", "r_elbow_pitch_joint", -1.1344, -1.1344, 0.30},
        {"waist_yaw_joint",     "",                     0.0000,  0.0000, 0.30},
    };

    inline scalar_t stepSeconds() { return gRampTime + gSettleTime + gCycles / gFrequency; }
  }  // namespace

  void AcController::handleWalkMode()
  {
    if (kBenchEnable)
    {
      // 台架上机器人被固定，摔倒判据会误触发并中断测试，故本分支不做摔倒保护。
      // 急停按 button 7 (/start_control)。恢复行走前把 kBenchEnable 改回 false。
      ROS_WARN_ONCE("[bench] bench mode: fall protection DISABLED, policy skipped");

      static ros::Time      start;
      static ros::Publisher stepPub;
      static vector_t       holdPos;      // 非待测关节的冻结目标位
      static int            lastStep = -1;
      static int            idxL = -1, idxR = -1;
      static scalar_t       rampL = 0.0, rampR = 0.0;
      static scalar_t       kpL = 0.0, kdL = 0.0, kpR = 0.0, kdR = 0.0;

      if (isfirstRecObs_)
      {
        loadBenchParams();   // 先读参数, 横幅才显示真实值
        start = ros::Time::now();
        lastStep = -1;
        idxL = idxR = -1;
        holdPos.resize(actionsSize_);
        for (int i = 0; i < actionsSize_; ++i)
        {
          const scalar_t p = hybridJointHandles_[i].getPositionDesired();
          holdPos(i) = std::isfinite(p) ? p : hybridJointHandles_[i].getPosition();
        }
        if (!stepPub)
        {
          // 发布当前步号, 供记录脚本切分数据; 结束后发 -1
          stepPub = ros::NodeHandle().advertise<std_msgs::Float32>("/bench/step", 1, true);
        }
        ROS_WARN_STREAM("[bench] scan start: " << kSteps.size() << " steps, "
                        << stepSeconds() << " s/step, total "
                        << kSteps.size() * stepSeconds() / 60.0 << " min");
        isfirstRecObs_ = false;
      }

      const scalar_t t    = (ros::Time::now() - start).toSec();
      const int      step = static_cast<int>(t / stepSeconds());

      auto findJoint = [this](const char *name) -> int
      {
        if (name == nullptr || name[0] == '\0') return -1;
        for (int i = 0; i < actionsSize_; ++i)
        {
          if (hybridJointHandles_[i].getName() == name) return i;
        }
        ROS_ERROR_STREAM("[bench] joint not found: " << name);
        return -1;
      };

      if (step != lastStep)
      {
        loadBenchParams();   // 每步开始重新读, 下一步即生效

        // 上一步的关节: 冻结位改成它正弦结束的位置(底端), 避免切步时跳变
        if (lastStep >= 0 && lastStep < static_cast<int>(kSteps.size()))
        {
          const BenchStep &prev = kSteps[lastStep];
          if (idxL >= 0) holdPos(idxL) = prev.centerL - prev.amplitude;
          if (idxR >= 0) holdPos(idxR) = prev.centerR - prev.amplitude;
        }
        idxL = idxR = -1;
        if (step < static_cast<int>(kSteps.size()))
        {
          const BenchStep &s = kSteps[step];
          idxL = findJoint(s.left);
          idxR = findJoint(s.right);
          if (idxL >= 0)
          {
            rampL = hybridJointHandles_[idxL].getPosition();
            kpL = benchGain("kp", s.left, joint_stiffness[idxL]);
            kdL = benchGain("kd", s.left, joint_damping[idxL]);
          }
          if (idxR >= 0)
          {
            rampR = hybridJointHandles_[idxR].getPosition();
            kpR = benchGain("kp", s.right, joint_stiffness[idxR]);
            kdR = benchGain("kd", s.right, joint_damping[idxR]);
          }
          // 关键参数打印到 rosout, 自动落到 ~/.ros/log/<run_id>/rosout.log
          if (idxL >= 0)
          {
            ROS_WARN_STREAM("[bench] step" << step << "/" << kSteps.size() - 1 << " " << s.left
                            << " idx=" << idxL
                            << " kp=" << kpL << " kd=" << kdL
                            << " key=" << gainKey(s.left)
                            << (kpL != joint_stiffness[idxL] || kdL != joint_damping[idxL]
                                    ? " [yaml override]" : " [policy metadata]")
                            << " range[" << s.centerL - s.amplitude << ", "
                            << s.centerL + s.amplitude << "] start=" << rampL);
          }
          if (idxR >= 0)
          {
            ROS_WARN_STREAM("[bench] step" << step << "/" << kSteps.size() - 1 << " " << s.right
                            << " idx=" << idxR
                            << " kp=" << kpR << " kd=" << kdR
                            << " key=" << gainKey(s.right)
                            << (kpR != joint_stiffness[idxR] || kdR != joint_damping[idxR]
                                    ? " [yaml override]" : " [policy metadata]")
                            << " range[" << s.centerR - s.amplitude << ", "
                            << s.centerR + s.amplitude << "] start=" << rampR);
          }
        }
        else
        {
          ROS_WARN("[bench] all steps done, holding pose");
        }
        std_msgs::Float32 msg;
        msg.data = (step < static_cast<int>(kSteps.size())) ? step : -1;
        stepPub.publish(msg);
        lastStep = step;
      }

      // 非待测关节: 保持冻结目标位
      for (int i = 0; i < actionsSize_; ++i)
      {
        if (i != idxL && i != idxR)
        {
          hybridJointHandles_[i].setCommand(holdPos(i), 0, joint_stiffness[i], joint_damping[i], 0);
        }
      }
      if (step >= static_cast<int>(kSteps.size()))
      {
        return;
      }

      // 待测关节: 缓升到摆动底端 -> 静置 -> 正弦(cos 起振, 位置与速度都连续)
      const scalar_t ts  = t - step * stepSeconds();
      const scalar_t amp = kSteps[step].amplitude;
      auto drive = [&](int j, scalar_t center, scalar_t rampFrom, scalar_t kp, scalar_t kd)
      {
        if (j < 0) return;
        const scalar_t bottom = center - amp;
        scalar_t pos = bottom;
        scalar_t vel = 0.0;
        if (ts < gRampTime)
        {
          const scalar_t ph = M_PI * ts / gRampTime;
          pos = rampFrom + (bottom - rampFrom) * 0.5 * (1.0 - std::cos(ph));
          vel = (bottom - rampFrom) * 0.5 * (M_PI / gRampTime) * std::sin(ph);
        }
        else if (ts >= gRampTime + gSettleTime)
        {
          const scalar_t tc = ts - gRampTime - gSettleTime;
          const scalar_t w  = 2.0 * M_PI * gFrequency;
          pos = center - amp * std::cos(w * tc);
          vel = amp * w * std::sin(w * tc);
        }
        hybridJointHandles_[j].setCommand(pos, vel, kp, kd, 0);
      };
      drive(idxL, kSteps[step].centerL, rampL, kpL, kdL);
      drive(idxR, kSteps[step].centerR, rampR, kpR, kdR);
      return;
    }


    // ===================== 以下为原策略行走路径, 逐行未改动 =====================

    // ===== hip pitch 电机台架测试(已停用,需要时取消注释)=====
    // constexpr scalar_t hipSineFrequency = 0.6;   // 正弦频率 Hz
    // constexpr scalar_t hipSineAmplitude = 0.3;   // 正弦幅值 rad
    // constexpr scalar_t hipSineStartDelay = 1.0;  // 进入 WALK 后延时 s 再开始正弦
    constexpr scalar_t controlLoopRate = 500.0;  // 控制环频率 Hz

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
    // 进入 WALK 模式时(walkModeCallback 置 isfirstRecObs_ = true)记录起始循环数
    if (isfirstRecObs_)
    {
      walkEntryLoopCount_ = loopCount_;
      isfirstRecObs_ = false;
    }
    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {
      computeObservation();  // policy 观测
      computeActions();      // policy 推理
      // limit action range
      scalar_t actionMin = -robotCfg_.clipActions;
      scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                     [actionMin, actionMax](scalar_t x)
                     { return std::max(actionMin, std::min(actionMax, x)); });
    }
    // 进入 WALK 后经过的时间(台架正弦测试用,停用中)
    const scalar_t sineTime =
        static_cast<scalar_t>(loopCount_ - walkEntryLoopCount_) / controlLoopRate -
        1.0;

    scalar_t pos_des;
    // set action
    for (int i = 0; i < actionsSize_; i++)
    {
      std::string partName = hybridJointHandles_[i].getName();

      scalar_t vel_des = 0;
      pos_des = actions_[i] * action_scale[i] + defaultJointAngles_(i);

      // ===== hip pitch 电机测试(已停用):两个 leg pitch 关节覆盖为正弦波,
      // 其余关节全部硬置 0。台架测试时把上面的 pos_des 行注释掉并启用下面的块。 =====
      // pos_des = 0;
      // vel_des = 0;
      // if (partName.find("leg_pitch") != std::string::npos && sineTime >= 0.0)
      // {
      //   const scalar_t sinePhase = 2.0 * M_PI * 0.6 * sineTime;
      //   pos_des = 0.3 * std::sin(sinePhase);
      //   vel_des = 0.3 * 2.0 * M_PI * 0.6 * std::cos(sinePhase);
      // }

      hybridJointHandles_[i].setCommand(pos_des, vel_des, joint_stiffness[i], joint_damping[i], 0);
      lastActions_(i, 0) = actions_[i];
    }

  }
  void AcController::handleDanceMode()
  {
    const bool enteringDance = isfirstRecObs_;
    if (enteringDance)
    {
      // Hold the current joint targets/gains until the first policy output,
      // then blend in over mimicBlendDuration_ to avoid a pose jump.
      mimicHoldJointAngles_.resize(actionsSize_);
      mimicHoldJointStiffness_.resize(actionsSize_);
      mimicHoldJointDamping_.resize(actionsSize_);
      for (int i = 0; i < actionsSize_; ++i)
      {
        double holdPosition = hybridJointHandles_[i].getPositionDesired();
        if (!std::isfinite(holdPosition))
        {
          holdPosition = hybridJointHandles_[i].getPosition();
        }
        mimicHoldJointAngles_(i) = holdPosition;
        mimicHoldJointStiffness_(i) = hybridJointHandles_[i].getKp();
        mimicHoldJointDamping_(i) = hybridJointHandles_[i].getKd();
      }
      mimicPolicyOutputReady_ = false;
      lastMimicInferenceLoop_ = -1;
    }

    if (danceTimeStep >= static_cast<int>(num_timesteps))
    {
      command_.x = 0.0;
      command_.y = 0.0;
      command_.yaw = 0.0;
      mode_ = Mode::WALK;
      isfirstRecObs_ = true;
      danceTimeStep = 0;
      lastActions_.setZero();
      std::fill(actions_.begin(), actions_.end(), 0.0f);
      ROS_INFO("Dance mimic finished; return to zero-velocity WALK");
      return;
    }

    // 摔倒保护: breaking 的 floorwork 段 base 大幅倾斜,绝对重力判断会误触发,
    // 这里保持注释(与原 dance 实现一致);训练侧倒地判断是相对参考姿态的。
    // if (propri_.projectedGravity(2) >= -0.3)
    // {
    //   ROS_ERROR("Dance mimic fall protection");
    //   mode_ = Mode::DEFAULT;
    //   return;
    // }

    const bool mimicInferenceDue =
        enteringDance ||
        lastMimicInferenceLoop_ < 0 ||
        loopCount_ - lastMimicInferenceLoop_ >= mimicDecimation_;
    if (mimicInferenceDue)
    {
      computeObservationDance();
      computeActionsDance();
      lastMimicInferenceLoop_ = loopCount_;

      const bool outputIsFinite =
          std::all_of(actions_.begin(), actions_.end(),
                      [](tensor_element_t action)
                      { return std::isfinite(action); });
      if (!outputIsFinite)
      {
        ROS_ERROR_THROTTLE(
            1.0,
            "Dance mimic policy produced a non-finite action; holding the previous joint targets");
      }
      else
      {
        const scalar_t actionMin = -robotCfg_.clipActions;
        const scalar_t actionMax = robotCfg_.clipActions;
        std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                       [actionMin, actionMax](scalar_t x)
                       { return std::max(actionMin, std::min(actionMax, x)); });

        if (!mimicPolicyOutputReady_)
        {
          mimicPolicyOutputReady_ = true;
          mimicBlendStartTime_ = ros::Time::now();
          ROS_INFO("Dance mimic policy output ready; smoothly taking over from the current pose");
        }
      }
    }

    if (!mimicPolicyOutputReady_)
    {
      for (int i = 0; i < actionsSize_; ++i)
      {
        hybridJointHandles_[i].setCommand(
            mimicHoldJointAngles_(i), 0,
            mimicHoldJointStiffness_(i), mimicHoldJointDamping_(i), 0);
      }
      return;
    }

    const scalar_t blend =
        std::max<scalar_t>(
            0.0,
            std::min<scalar_t>(
                1.0,
                (ros::Time::now() - mimicBlendStartTime_).toSec() /
                    mimicBlendDuration_));
    for (int i = 0; i < actionsSize_; i++)
    {
      const scalar_t policyPosition =
          actions_[i] * action_scale_dance[i] + defaultJointAnglesDance_(i);
      const scalar_t pos_des =
          mimicHoldJointAngles_(i) * (1.0 - blend) + policyPosition * blend;
      const scalar_t kp =
          mimicHoldJointStiffness_(i) * (1.0 - blend) +
          joint_stiffness_dance[i] * blend;
      const scalar_t kd =
          mimicHoldJointDamping_(i) * (1.0 - blend) +
          joint_damping_dance[i] * blend;
      hybridJointHandles_[i].setCommand(pos_des, 0, kp, kd, 0);
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
        kp = 8.0;
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

    scalar_t pos_des;
    for (int actionIndex = 0; actionIndex < actions13DOFJumpSize_; ++actionIndex)
    {
      const int jointIndex = jointIndices13DOFJump_[actionIndex];
      scalar_t kp = joint_stiffness_13DOF_jump[jointIndex];
      scalar_t kd = joint_damping_13DOF_jump[jointIndex];

      // Match the ankle PD gains used by the 21-DOF walking controller.
      // Jump action indices 4/10 are ankle pitch; 5/11 are ankle roll.
      if (actionIndex == 4 || actionIndex == 10)
      {
        kp = 8.0;
        kd = 0.5;
      }
      else if (actionIndex == 5 || actionIndex == 11)
      {
        kp = 2.0;
        kd = 0.2;
      }

      pos_des =
          actions13DOFJump_[actionIndex] * action_scale_13DOF_jump[actionIndex] +
          defaultJointAngles13DOFJump_(actionIndex);
      hybridJointHandles_[jointIndex].setCommand(pos_des, 0, kp, kd, 0);
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
      std::string motionFilePath;
      if (!nh.getParam("/motionFilePath", motionFilePath))
      {
          ROS_ERROR_STREAM("Get motion path fail from param server, some error occur!");
          return false;
      }

      std::ifstream file(motionFilePath);
      if (!file.is_open())
      {
          ROS_ERROR_STREAM("Cannot open dance motion file: " << motionFilePath);
          return false;
      }

      try
      {
          json motion;
          file >> motion;
          // NoetixBumi motion JSON v2: metadata + joint_pos/joint_vel per frame
          // and body_quat_w as the reference root quaternion (wxyz).
          if (!motion.contains("metadata") ||
              !motion.contains("joint_pos") ||
              !motion.contains("joint_vel") ||
              !motion.contains("body_quat_w"))
          {
              ROS_ERROR_STREAM("Dance motion JSON must contain metadata, joint_pos, "
                               "joint_vel and body_quat_w: " << motionFilePath);
              return false;
          }
          const auto &metadata = motion.at("metadata");
          if (!metadata.contains("joint_names") || !metadata.contains("fps"))
          {
              ROS_ERROR_STREAM("Dance motion metadata must contain joint_names and fps: "
                               << motionFilePath);
              return false;
          }
          const auto motionJointNames =
              metadata.at("joint_names").get<std::vector<std::string>>();
          if (motionJointNames != joint_names_dance)
          {
              ROS_ERROR_STREAM("Dance motion joint_names do not match the mimic ONNX "
                               "metadata: " << motionFilePath);
              return false;
          }
          constexpr double expectedFps = 50.0;
          const double motionFps = metadata.at("fps").get<double>();
          if (std::abs(motionFps - expectedFps) > 1.0e-6)
          {
              ROS_ERROR_STREAM("Dance motion must be 50 Hz; got " << motionFps);
              return false;
          }

          ref_joint_pos =
              motion.at("joint_pos").get<std::vector<std::vector<double>>>();
          ref_joint_vel =
              motion.at("joint_vel").get<std::vector<std::vector<double>>>();
          ref_quat =
              motion.at("body_quat_w").get<std::vector<std::vector<double>>>();
      }
      catch (const std::exception &exception)
      {
          ROS_ERROR_STREAM("Failed to parse dance motion " << motionFilePath
                           << ": " << exception.what());
          return false;
      }

      const size_t frameCount = ref_joint_pos.size();
      if (frameCount == 0 ||
          ref_joint_vel.size() != frameCount ||
          ref_quat.size() != frameCount)
      {
          ROS_ERROR_STREAM("Dance motion contains inconsistent frame arrays: "
                           << motionFilePath);
          return false;
      }
      for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
      {
          if (ref_joint_pos[frameIndex].size() != static_cast<size_t>(actionsSize_) ||
              ref_joint_vel[frameIndex].size() != static_cast<size_t>(actionsSize_) ||
              ref_quat[frameIndex].size() != 4)
          {
              ROS_ERROR_STREAM("Invalid dance motion frame " << frameIndex
                               << ": expected 21 joint positions, 21 joint velocities "
                               "and one wxyz quaternion");
              return false;
          }
      }

      num_timesteps = frameCount;
      num_joints = frameCount > 0 ? ref_joint_pos.front().size() : 0;
      danceTimeStep = 0;
      ROS_INFO_STREAM("Loaded fixed dance motion with " << frameCount
                      << " frames at 50 Hz: " << motionFilePath);
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
    if (!nh.getParam("/policyFileDance", policyFilePathDance_))
    {
      ROS_ERROR("Missing /policyFileDance");
      return false;
    }

    ROS_INFO_STREAM("Load 21-DOF policy: " << policyFilePath_);
    ROS_INFO_STREAM("Load 12-DOF walk policy: " << policyFilePath12DOFWalk_);
    ROS_INFO_STREAM("Load 13-DOF jump entry: " << policyFilePath13DOFJump_);
    ROS_INFO_STREAM("Load dance mimic policy: " << policyFilePathDance_);

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
    policySessionPtrDance_ = std::make_unique<Ort::Session>(
        *onnxEnvPrt_, policyFilePathDance_.c_str(), sessionOptions);

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
    readPolicyMetadata(
        *policySessionPtrDance_, allocator, joint_names_dance,
        joint_stiffness_dance, joint_damping_dance,
        default_joint_pos_dance, action_scale_dance);

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
    readPolicyIo(
        *policySessionPtrDance_, allocator,
        policyInputNodeNameAllocatedStringsDance,
        policyOutputNodeNameAllocatedStringsDance,
        policyInputNamesDance_, policyOutputNamesDance_,
        policyInputShapesDance_, policyOutputShapesDance_, "dance mimic");

    ROS_INFO("Loaded all locomotion + dance mimic policies successfully");
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

    // Dance mimic policy: obs history [1, obs*stack] + future command window
    // [1, futureFrames*2*actions]; both outputs/actions are full 21 joints.
    missing += static_cast<int>(!nh.getParam(
        "/LeggedRobotCfg/size/observations_dance_size", observationSizeDance_));
    if (missing != 0)
    {
      ROS_ERROR("[AcController] Missing observations_dance_size parameter");
      return false;
    }
    nh.param("/controllers/ac_controller/mimic_decimation",
             mimicDecimation_, robotCfg_.controlCfg.decimation);
    if (mimicDecimation_ <= 0)
    {
      ROS_ERROR("[AcController] mimic_decimation must be positive");
      return false;
    }
    ROS_INFO_STREAM("[AcController] dance mimic decimation: " << mimicDecimation_);

    const auto tensorSizeAt = [this](size_t index, const std::string &label) -> int64_t
    {
      if (index >= policyInputShapesDance_.size() || policyInputShapesDance_[index].empty())
      {
        ROS_ERROR_STREAM("[AcController] " << label << " tensor " << index << " missing");
        return -1;
      }
      int64_t size = 1;
      for (const int64_t dim : policyInputShapesDance_[index])
      {
        if (dim <= 0)
        {
          ROS_ERROR_STREAM("[AcController] " << label
                           << " tensor " << index << " has dynamic dimension " << dim);
          return -1;
        }
        size *= dim;
      }
      return size;
    };
    const int64_t danceObsSize = tensorSizeAt(0, "dance mimic obs");
    const int64_t danceWindowRaw = tensorSizeAt(1, "dance command window");
    const int64_t danceOutputSize = tensorSize(policyOutputShapesDance_, "dance mimic output");
    if (danceObsSize != observationSizeDance_ * stackSize_ ||
        danceOutputSize != actionsSize_ ||
        observationSizeDance_ != 5 * actionsSize_ + 9 ||
        danceWindowRaw % (2 * actionsSize_) != 0)
    {
      ROS_ERROR_STREAM("[AcController] dance mimic must be [1, "
                       << 5 * actionsSize_ + 9 << "x" << stackSize_
                       << "] + [1, N x " << 2 * actionsSize_ << "] -> [1, "
                       << actionsSize_ << "], got [1, " << danceObsSize
                       << "] + [1, " << danceWindowRaw << "] -> [1, " << danceOutputSize);
      return false;
    }
    danceFutureFrames_ = static_cast<int>(danceWindowRaw / (2 * actionsSize_));
    ROS_INFO_STREAM("[AcController] dance mimic command window: "
                    << danceFutureFrames_ << " future frames");

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
                              "13-DOF jump") ||
        !validateFullMetadata(joint_names_dance, joint_stiffness_dance,
                              joint_damping_dance, default_joint_pos_dance,
                              "dance mimic") ||
        action_scale_dance.size() != static_cast<size_t>(actionsSize_))
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
    policyObservationsDance_.assign(danceObsSize, 0.0f);
    commandWindowDance_.assign(danceWindowRaw, 0.0f);

    proprioHistoryBuffer_.resize(fullInputSize);
    proprioHistoryBuffer12DOF_.resize(walkInputSize);
    proprioHistoryBuffer13DOFJump_.resize(jumpInputSize);
    proprioHistoryBufferDance_.resize(danceObsSize);
    proprioHistoryBuffer_.setZero();
    proprioHistoryBuffer12DOF_.setZero();
    proprioHistoryBuffer13DOFJump_.setZero();
    proprioHistoryBufferDance_.setZero();

    defaultJointAngles_.resize(actionsSize_);
    for (int i = 0; i < actionsSize_; ++i)
    {
      defaultJointAngles_(i) = default_joint_pos[i];
    }
    defaultJointAnglesDance_.resize(actionsSize_);
    for (int i = 0; i < actionsSize_; ++i)
    {
      defaultJointAnglesDance_(i) = default_joint_pos_dance[i];
    }
    mimicHoldJointAngles_.resize(actionsSize_);
    mimicHoldJointStiffness_.resize(actionsSize_);
    mimicHoldJointDamping_.resize(actionsSize_);
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
                    << "x" << stackSize_
                    << ", dance mimic=" << observationSizeDance_
                    << "x" << stackSize_
                    << " + " << danceFutureFrames_ << " future frames");
    return true;
  }

  void AcController::computeActionsDance()
  {
    // The mimic policy takes both the proprioceptive history and the future
    // reference command window (MHA K/V tokens), in that input order.
    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, policyObservationsDance_.data(), policyObservationsDance_.size(),
        policyInputShapesDance_[0].data(), policyInputShapesDance_[0].size()));
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, commandWindowDance_.data(), commandWindowDance_.size(),
        policyInputShapesDance_[1].data(), policyInputShapesDance_[1].size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues;
    outputValues = policySessionPtrDance_->Run(
        runOptions, policyInputNamesDance_.data(), policyInputValues.data(),
        policyInputNamesDance_.size(), policyOutputNamesDance_.data(), 1);

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

void AcController::computeObservationDance()
  {
    if (danceTimeStep < 0 || danceTimeStep >= static_cast<int>(num_timesteps))
    {
      return;
    }

    // The motion loader has already verified that the JSON joint order matches
    // the ONNX metadata. Training constructs the command as
    // concat(joint_pos, joint_vel) in that same robot joint order.
    vector_t ref_dof_pos_(actionsSize_);
    vector_t ref_dof_vel_(actionsSize_);
    for (int i = 0; i < actionsSize_; ++i)
    {
      ref_dof_pos_(i) = ref_joint_pos[danceTimeStep][i];
      ref_dof_vel_(i) = ref_joint_vel[danceTimeStep][i];
    }

    // Anchor body is waist_yaw_link (training anchor_body_name). Its world
    // orientation is the base orientation composed with the waist yaw joint.
    const auto& quat_data = ref_quat[danceTimeStep];
    Eigen::Quaterniond ref_robot_quat(
        quat_data[0], quat_data[1], quat_data[2], quat_data[3]);
    ref_robot_quat.normalize();
    const Eigen::Quaterniond ref_waist_yaw(
        Eigen::AngleAxisd(ref_joint_pos[danceTimeStep][2], Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond ref_anchor_quat = ref_robot_quat * ref_waist_yaw;
    ref_anchor_quat.normalize();

    Eigen::Quaterniond robot_base_quat(
        propri_.robot_quat_.w(), propri_.robot_quat_.x(),
        propri_.robot_quat_.y(), propri_.robot_quat_.z());
    robot_base_quat.normalize();
    const Eigen::Quaterniond robot_waist_yaw(
        Eigen::AngleAxisd(propri_.jointPos(2), Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond robot_anchor_quat = robot_base_quat * robot_waist_yaw;
    robot_anchor_quat.normalize();

    if (isfirstRecObs_)
    {
      // Align the reference heading with the robot's yaw once at playback
      // start, so the motion is replayed in the robot's current facing.
      init_to_world =
          yaw_quat<double>(robot_anchor_quat) *
          yaw_quat<double>(ref_anchor_quat).conjugate();
      init_to_world.normalize();
    }

    // motion_anchor_ori_b: reference anchor orientation relative to the real
    // robot's anchor orientation, as the first two rotation-matrix columns.
    const Eigen::Quaterniond motion_anchor_quat_w =
        init_to_world * ref_anchor_quat;
    const Eigen::Quaterniond motion_anchor_quat_b =
        robot_anchor_quat.conjugate() * motion_anchor_quat_w;
    const Eigen::Matrix3d motion_anchor_rotation_b =
        motion_anchor_quat_b.normalized().toRotationMatrix();
    vector_t motion_anchor_ori_b(6);
    for (int row = 0; row < 3; ++row)
    {
      for (int column = 0; column < 2; ++column)
      {
        motion_anchor_ori_b(row * 2 + column) =
            motion_anchor_rotation_b(row, column);
      }
    }

    vector_t actions(lastActions_);
    vector_t proprioObs(observationSizeDance_);
    proprioObs <<
        ref_dof_pos_,
        ref_dof_vel_,
        motion_anchor_ori_b,
        propri_.baseAngVel,
        (propri_.jointPos - defaultJointAnglesDance_),
        propri_.jointVel,
        actions;

    // Future reference command window: frames +1..+N after the current one,
    // matching the training FutureCommandCfg frame_offsets. Clamp at the end
    // of the clip so the policy keeps receiving a valid window.
    const int lastFrame = static_cast<int>(num_timesteps) - 1;
    const int commandDim = 2 * actionsSize_;
    for (int frame = 0; frame < danceFutureFrames_; ++frame)
    {
      const int refFrame = std::min(danceTimeStep + 1 + frame, lastFrame);
      for (int i = 0; i < actionsSize_; ++i)
      {
        commandWindowDance_[frame * commandDim + i] =
            static_cast<tensor_element_t>(ref_joint_pos[refFrame][i]);
        commandWindowDance_[frame * commandDim + actionsSize_ + i] =
            static_cast<tensor_element_t>(ref_joint_vel[refFrame][i]);
      }
    }

    if (isfirstRecObs_)
    {
      // The action part of the first observation must be zero, matching the
      // training reset state where last_action starts at zero.
      for (int i = observationSizeDance_ - actionsSize_;
           i < observationSizeDance_; ++i)
      {
        proprioObs(i, 0) = 0.0;
      }

      for (size_t i = 0; i < stackSize_; i++)
      {
        proprioHistoryBufferDance_.segment(i * observationSizeDance_, observationSizeDance_) = proprioObs.cast<tensor_element_t>();
      }
      isfirstRecObs_ = false;

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
    std::transform(commandWindowDance_.begin(), commandWindowDance_.end(), commandWindowDance_.begin(),
                   [obsMin, obsMax](tensor_element_t x)
                   { return std::max<tensor_element_t>(
                       std::min<tensor_element_t>(x, static_cast<tensor_element_t>(obsMax)),
                       static_cast<tensor_element_t>(obsMin)); });

    ++danceTimeStep;
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


// ● 这些 topic 不需要额外开启——控制器一跑起来(./simulation.sh 加载 ac_controller 后)就自动在 500 Hz 发布，你要做的只是“订阅+画图”。完整操作步骤：
   
//   第 0 步：修 GUI 显示(root 跑 launch 的前提)

//   在桌面用户的终端执行一次(每次重启 X 后需重做)：
//   xhost +local:

//   第 1 步：启动仿真

//   ./simulation.sh
//   等日志出现 Loaded controllers 即表示 topic 已在发布。验证一下：
//   rostopic list | grep data_analysis     # 应列出 9 个
//   rostopic hz /data_analysis/rl_planned_joint_pos   # 应约 500 Hz

//   第 2 步：开曲线窗口(另开终端，source 环境)

//   方式 A:rqt_plot 一条命令拉全(左膝示例，右膝把 11 换 12)：
//   source ./devel/setup.bash
//   rqt_plot \
//     /data_analysis/rl_planned_joint_pos/data[11] \
//     /data_analysis/real_joint_pos/data[11] \
//     /data_analysis/rl_planned_joint_vel/data[11] \
//     /data_analysis/real_joint_vel/data[11] \
//     /data_analysis/rl_planned_torque/data[11] \
//     /data_analysis/real_torque/data[11] \
//     /data_analysis/imu_euler_xyz/data[0] \
//     /data_analysis/imu_euler_xyz/data[1] \
//     /data_analysis/imu_euler_xyz/data[2]
//   曲线太多会挤，rqt_plot 窗口左上角可以用 -/+ 按钮把曲线拆到多个子图。

//   方式 B:PlotJuggler(推荐，9 个全看最舒服)：
//   sudo apt install ros-noetic-plotjuggler-ros   # 只需装一次
//   rosrun plotjuggler plotjuggler
//   然后：左侧 Streaming → ROS Topic Subscriber → 勾选全部 9 个 /data_analysis/* → Start;把 rl_planned_joint_pos/data[11]、real_joint_pos/data[11]
//   拖进一张图，rl_planned_torque/data[11]、real_torque/data[11] 拖进另一张，IMU 的 3 维拖第三张。缩放、暂停、导出 PNG/CSV 都在工具栏。

//   第 3 步(可选)：同时录 bag

//   如果既要实时看又要留数据：
//   rosbag record /data_analysis/* -O knee_test.bag
//   或启动时直接 ./simulation.sh record_data:=true(自动录到 rl_controllers/data/)。

//   之后手动 /start_control → /switch_mode → /walk_mode,进