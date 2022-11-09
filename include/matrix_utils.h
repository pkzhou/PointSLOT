//
// Created by liuyuzhen on 2020/5/24.
//

#ifndef ORB_SLAM2_MATRIX_UTILS_H
#define ORB_SLAM2_MATRIX_UTILS_H

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T &roll, const T &pitch, const T &yaw);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in);




#endif //ORB_SLAM2_MATRIX_UTILS_H
