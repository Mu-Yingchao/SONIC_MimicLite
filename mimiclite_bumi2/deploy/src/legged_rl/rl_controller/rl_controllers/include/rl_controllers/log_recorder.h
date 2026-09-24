#pragma once

#include "rl_controllers/Types.h"
#include <fstream>
#include <filesystem>
#include <vector>

namespace legged {

class LogRecorder {
public:
    LogRecorder(const std::string &log_path, const int dof, const std::vector<std::string> &joint_names, const int decimation = 10);

    ~LogRecorder();

    void Open();

    void Close();

    void WriteScalar(const double& data);

    template<typename T> void WriteStdVec(const std::vector<T>& data);

    void WriteEigenVec(const Eigen::VectorXd& data);

    void WriteEigenVec(const Eigen::Vector3d& data);

    std::vector<std::string> GenerateTitles();

private:
    std::ofstream log_file_;
    std::string log_path_;
    std::filesystem::path fs_log_path_;
    std::filesystem::path fs_log_backup_path_;
    int dof_{20};
    std::vector<std::string> joint_names_;
    int decimation_{10};
    int size_check_decimation_{60000};
    int cycle_count_{0};
};

} // namespace legged