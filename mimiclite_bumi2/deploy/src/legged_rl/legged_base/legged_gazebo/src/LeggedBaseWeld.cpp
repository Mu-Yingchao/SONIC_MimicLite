#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <std_msgs/Float32.h>

namespace gazebo
{
/**
 * 电机测试台架插件:
 * - 收到 /walk_mode 时,把模型抬高 lift_z(双脚离地),然后动态创建
 *   world->base_link 的 fixed joint,物理级焊接机身,关节电机怎么摆都不会晃;
 * - 收到 /base_unfreeze / /emergency_stop / /position_control / /switch_mode
 *   时删除该关节,恢复自由基座。
 * 挂载方式(bumi.urdf):
 *   <gazebo><plugin name="legged_base_weld" filename="liblegged_base_weld.so">
 *     <link>base_link</link><lift_z>0.08</lift_z>
 *   </plugin></gazebo>
 */
class LeggedBaseWeld : public ModelPlugin
{
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    linkName_ = "base_link";
    liftZ_ = 0.08;
    if (sdf->HasElement("link"))
      linkName_ = sdf->Get<std::string>("link");
    if (sdf->HasElement("lift_z"))
      liftZ_ = sdf->Get<double>("lift_z");

    if (!ros::isInitialized())
    {
      int argc = 0;
      char** argv = nullptr;
      ros::init(argc, argv, "legged_base_weld");
    }
    ros::NodeHandle nh;
    freezeSub_ = nh.subscribe("/walk_mode", 1, &LeggedBaseWeld::onFreeze, this);
    unfreezeSubs_.push_back(
        nh.subscribe("/base_unfreeze", 1, &LeggedBaseWeld::onUnfreeze, this));
    unfreezeSubs_.push_back(
        nh.subscribe("/emergency_stop", 1, &LeggedBaseWeld::onUnfreeze, this));
    unfreezeSubs_.push_back(
        nh.subscribe("/position_control", 1, &LeggedBaseWeld::onUnfreeze, this));
    unfreezeSubs_.push_back(
        nh.subscribe("/switch_mode", 1, &LeggedBaseWeld::onUnfreeze, this));

    // 物理操作统一放到物理线程的世界更新回调里做
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(
        [this](const common::UpdateInfo&) { this->onWorldUpdate(); });

    gzmsg << "[legged_base_weld] ready: weld on /walk_mode, unweld on /base_unfreeze"
          << " (link=" << linkName_ << ", lift_z=" << liftZ_ << ")\n";
  }

private:
  void onFreeze(const std_msgs::Float32ConstPtr&)
  {
    freezeRequested_ = true;
    unfreezeRequested_ = false;
  }

  void onUnfreeze(const std_msgs::Float32ConstPtr&)
  {
    unfreezeRequested_ = true;
    freezeRequested_ = false;
  }

  void onWorldUpdate()
  {
    if (freezeRequested_)
    {
      freezeRequested_ = false;
      if (joint_)
        return;
      physics::LinkPtr link = model_->GetLink(linkName_);
      if (!link)
      {
        gzwarn << "[legged_base_weld] link " << linkName_ << " not found\n";
        return;
      }
      // 抬高模型让双脚离地,避免焊接后脚掌仍与地面接触顶撞
      ignition::math::Pose3d pose = model_->WorldPose();
      pose.Pos().Z() += liftZ_;
      model_->SetWorldPose(pose);
      // 动态创建 world->link 的 fixed joint(父 link 为空即世界)
      joint_ = model_->GetWorld()->Physics()->CreateJoint("fixed", model_);
      joint_->SetName("legged_base_weld_joint");
      joint_->Load(physics::LinkPtr(), link, link->WorldPose());
      joint_->Init();
      gzmsg << "[legged_base_weld] base_link welded to world at z="
            << pose.Pos().Z() << "\n";
    }
    else if (unfreezeRequested_)
    {
      unfreezeRequested_ = false;
      if (!joint_)
        return;
      model_->RemoveJoint(joint_->GetName());
      joint_.reset();
      gzmsg << "[legged_base_weld] base_link unwelded\n";
    }
  }

  physics::ModelPtr model_;
  physics::JointPtr joint_;
  std::string linkName_;
  double liftZ_{ 0.08 };
  ros::Subscriber freezeSub_;
  std::vector<ros::Subscriber> unfreezeSubs_;
  event::ConnectionPtr updateConnection_;
  bool freezeRequested_{ false };
  bool unfreezeRequested_{ false };
};

GZ_REGISTER_MODEL_PLUGIN(LeggedBaseWeld)
}  // namespace gazebo
