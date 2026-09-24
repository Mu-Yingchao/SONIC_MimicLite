#include "log_recorder_hw.h"
#include <sstream>

namespace legged {

LogRecorderHw::LogRecorderHw(const std::string &log_path, const int dof, const std::vector<std::string> &joint_names, const int decimation) : log_path_(log_path), dof_(dof), joint_names_(joint_names), decimation_(decimation) {
    std::filesystem::path path(log_path_);
    std::filesystem::path backup_path = path.parent_path() / (path.stem().string() + "_autobackup" + path.extension().string());
    fs_log_path_ = path;
    fs_log_backup_path_ = backup_path;

    if (std::filesystem::exists(fs_log_backup_path_)) {
        std::filesystem::remove(fs_log_backup_path_);
    }

    // 获取上层目录
    std::filesystem::path fs_log_dir = fs_log_path_.parent_path();

    // 如果路径非空且目录不存在，则创建
    if (!fs_log_dir.empty() && !std::filesystem::exists(fs_log_dir)) {
        std::filesystem::create_directories(fs_log_dir);
    }

    log_file_.open(log_path_, std::ios::trunc);

    auto titles = GenerateTitles();
    
    // std::vector<std::string> title = {
    //     "time", 
    //     "PosState_0", "PosState_1", "PosState_2", "PosState_3", "PosState_4", "PosState_5", "PosState_6", "PosState_7", "PosState_8", "PosState_9", "PosState_10", "PosState_11", "PosState_12", "PosState_13", "PosState_14", "PosState_15", "PosState_16", "PosState_17",
    //     "VelState_0", "VelState_1", "VelState_2", "VelState_3", "VelState_4", "VelState_5", "VelState_6", "VelState_7", "VelState_8", "VelState_9", "VelState_10", "VelState_11", "VelState_12", "VelState_13", "VelState_14", "VelState_15", "VelState_16", "VelState_17",
    //     "TorState_0", "TorState_1", "TorState_2", "TorState_3", "TorState_4", "TorState_5", "TorState_6", "TorState_7", "TorState_8", "TorState_9", "TorState_10", "TorState_11", "TorState_12", "TorState_13", "TorState_14", "TorState_15", "TorState_16", "TorState_17", 
    //     // "propri_.baseAngVel", "propri_.projectedGrav",
    //     "imuEulerXyz_0", "imuEulerXyz_1", "imuEulerXyz_2",
    //     "angularVel_0",  "angularVel_1",  "angularVel_2",
    //     "linearAccel_0", "linearAccel_1", "linearAccel_2",

    //     "PosCmd_0", "PosCmd_1", "PosCmd_2", "PosCmd_3", "PosCmd_4", "PosCmd_5", "PosCmd_6", "PosCmd_7", "PosCmd_8", "PosCmd_9", "PosCmd_10", "PosCmd_11", "PosCmd_12", "PosCmd_13", "PosCmd_14", "PosCmd_15", "PosCmd_16", "PosCmd_17",
    //     "VelCmd_0", "VelCmd_1", "VelCmd_2", "VelCmd_3", "VelCmd_4", "VelCmd_5", "VelCmd_6", "VelCmd_7", "VelCmd_8", "VelCmd_9", "VelCmd_10", "VelCmd_11", "VelCmd_12", "VelCmd_13", "VelCmd_14", "VelCmd_15", "VelCmd_16", "VelCmd_17",
    //     "TorCmd_0", "TorCmd_1", "TorCmd_2", "TorCmd_3", "TorCmd_4", "TorCmd_5", "TorCmd_6", "TorCmd_7", "TorCmd_8", "TorCmd_9", "TorCmd_10", "TorCmd_11", "TorCmd_12", "TorCmd_13", "TorCmd_14", "TorCmd_15", "TorCmd_16", "TorCmd_17", 
    // };

    for (int i = 0; i < titles.size(); ++i) {
        log_file_ << titles[i] << ",";
    }
    log_file_ << std::endl;
    log_file_.close();
}

LogRecorderHw::~LogRecorderHw() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void LogRecorderHw::Open() {
    if (cycle_count_ % decimation_ == 0) {
        if (cycle_count_ >= size_check_decimation_) {
            auto file_size = std::filesystem::file_size(fs_log_path_);
            constexpr std::uintmax_t MAX_LOG_SIZE = 1 * 1024 * 1024 * 1024;  // 1ull << 30; // 1GB
            if (file_size > MAX_LOG_SIZE) {
                if (std::filesystem::exists(fs_log_backup_path_)) {
                    std::filesystem::remove(fs_log_backup_path_);
                }
                std::filesystem::rename(fs_log_path_, fs_log_backup_path_);
                log_file_.open(log_path_, std::ios::trunc);

                cycle_count_ = 0; // 重置计数器
            } else {
                log_file_.open(log_path_, std::ios::app);
            }
        } else {
            log_file_.open(log_path_, std::ios::app);
        }
    }
    ++cycle_count_;
}

void LogRecorderHw::Close() {
    if (!log_file_.is_open()) {
        return;
    }
    log_file_ << std::endl;
    log_file_.close();
}

void LogRecorderHw::WriteScalar(const double& data) {
    if (!log_file_.is_open()) {
        return;
    }
    log_file_ << data << "," << "|" << ",";
}

template<typename T> void LogRecorderHw::WriteStdVec(const std::vector<T>& data) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < data.size()-1; ++i) {
        log_file_ << data[i] << ",";
    }
    log_file_ << data[data.size()-1] << "," << "|" << ",";
}

void LogRecorderHw::WriteEigenVec(const Eigen::VectorXd& data) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < data.size()-1; ++i) {
        log_file_ << data[i] << ",";
    }
    log_file_ << data[data.size()-1] << "," << "|" << ",";
}

void LogRecorderHw::WriteEigenVec(const Eigen::Vector3d& data) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        log_file_ << data[i] << ",";
    }
    log_file_ << data[2] << "," << "|" << ",";
}

void LogRecorderHw::WriteJointStatePos(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].pos_ << ",";
    }
    log_file_ << joint_data[size-1].pos_ << "," << "|" << ",";
}

void LogRecorderHw::WriteJointStateVel(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].vel_ << ",";
    }
    log_file_ << joint_data[size-1].vel_ << "," << "|" << ",";
}

void LogRecorderHw::WriteJointStateTor(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].tau_ << ",";
    }
    log_file_ << joint_data[size-1].tau_ << "," << "|" << ",";
}

void LogRecorderHw::WriteJointCmdPos(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].pos_des_ << ",";
    }
    log_file_ << joint_data[size-1].pos_des_ << "," << "|" << ",";
}

void LogRecorderHw::WriteJointCmdVel(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].vel_des_ << ",";
    }
    log_file_ << joint_data[size-1].vel_des_ << "," << "|" << ",";
}

void LogRecorderHw::WriteJointCmdTor(const BumiMotorData* joint_data, const int size) {
    if (!log_file_.is_open()) {
        return;
    }
    for (int i = 0; i < size-1; ++i) {
        log_file_ << joint_data[i].ff_ << ",";
    }
    log_file_ << joint_data[size-1].ff_ << "," << "|" << ",";
}

std::vector<std::string> LogRecorderHw::GenerateTitles() {
    std::vector<std::string> titles;
    titles.push_back("time");

    // 通用生成函数
    auto appendWithPrefix = [&](const std::string& prefix, int dof) {
        for (int i = 0; i < dof; ++i) {
            std::ostringstream oss;
            oss << prefix << "_" << i;
            titles.push_back(oss.str());
        }
    };

    auto appendWithJointNames = [&](const std::string& prefix, const std::vector<std::string>& joint_names) {
        for (int i = 0; i < joint_names.size(); ++i) {
            std::ostringstream oss;
            oss << prefix << "_" << joint_names[i];
            titles.push_back(oss.str());
        }
    };

    auto appendWithoutPrefix = [&](const std::string& prefix, int dof) {
        for (int i = 0; i < dof; ++i) {
            std::ostringstream oss;
            oss << prefix;
            titles.push_back(oss.str());
        }
    };

    // appendWithPrefix("PosState", dof_);
    // appendWithPrefix("VelState", dof_);
    // appendWithPrefix("TorState", dof_);

    appendWithoutPrefix("|", 1); // 分隔符

    appendWithJointNames("PosState", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    appendWithJointNames("VelState", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    appendWithJointNames("TorState", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    // appendWithPrefix("imuEulerXyz", 3);
    // appendWithoutPrefix("|", 1); // 分隔符

    // appendWithPrefix("angularVel", 3);
    // appendWithoutPrefix("|", 1); // 分隔符

    // appendWithPrefix("linearAccel", 3);
    // appendWithoutPrefix("|", 1); // 分隔符

    // appendWithPrefix("PosCmd", dof_);
    // appendWithPrefix("VelCmd", dof_);
    // appendWithPrefix("TorCmd", dof_);

    appendWithJointNames("PosCmd", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    appendWithJointNames("VelCmd", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    appendWithJointNames("TorCmd", joint_names_);
    appendWithoutPrefix("|", 1); // 分隔符

    return titles;
}

} // namespace legged