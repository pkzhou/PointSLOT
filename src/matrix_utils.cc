//
// Created by liuyuzhen on 2020/5/24.
//
#include "matrix_utils.h"

// std c
#include <math.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

using namespace Eigen;

template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T &roll, const T &pitch, const T &yaw)
{
    T sy = sin(yaw * 0.5);
    T cy = cos(yaw * 0.5);
    T sp = sin(pitch * 0.5);
    T cp = cos(pitch * 0.5);
    T sr = sin(roll * 0.5);
    T cr = cos(roll * 0.5);
    T w = cr * cp * cy + sr * sp * sy;
    T x = sr * cp * cy - cr * sp * sy;
    T y = cr * sp * cy + sr * cp * sy;
    T z = cr * cp * sy - sr * sp * cy;
    return Eigen::Quaternion<T>(w, x, y, z);
}
template Eigen::Quaterniond zyx_euler_to_quat<double>(const double &, const double &, const double &);

//  ，模板t还是不太理解，应该是函数不定义类型，对于什么数据类型参数都可以
/// 任一矩阵转换成齐次， 在底部加以行全1
template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in)
{
    // 定义动态矩阵
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_homo_out;
    // 行数
    int raw_rows = pts_in.rows();
    // 列数
    int raw_cols = pts_in.cols();

    // 增加一行， 增加的是 全1的一行
    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
            Matrix<T, 1, Dynamic>::Ones(raw_cols);
    return pts_homo_out;
}
template MatrixXd real_to_homo_coord<double>(const MatrixXd &);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_out(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N

    return pts_out;
}
template MatrixXd homo_to_real_coord<double>(const MatrixXd &);
