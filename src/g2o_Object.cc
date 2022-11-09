//
// Created by liuyuzhen on 2020/5/31.
//
/**
* This file is part of CubeSLAM
*
* Copyright (C) 2018  Shichao Yang (Carnegie Mellon Univ)
*/

//#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "matrix_utils.h"
#include "g2o_Object.h"
#include "Parameters.h"
//#include <Eigen/Core>
//#include <Eigen/Geometry>
//#include <Eigen/Dense>
//#include <math.h>
//#include <algorithm> // std::swap
int jishu_temp = 0;
namespace g2o
{

using namespace Eigen;
using namespace std;

SE3Quat exptwist_norollpitch(const Vector6d &update)
{
    Vector3d omega;
    for (int i = 0; i < 3; i++)
        omega[i] = update[i];
    Vector3d upsilon;
    for (int i = 0; i < 3; i++)
        upsilon[i] = update[i + 3];

    double theta = omega.norm();
    Matrix3d Omega = skew(omega);

    Matrix3d R;
    R << cos(omega(2)), -sin(omega(2)), 0,
            sin(omega(2)), cos(omega(2)), 0,
            0, 0, 1;

    Matrix3d V;
    if (theta < 0.00001)
    {
        V = R;
    }
    else
    {
        Matrix3d Omega2 = Omega * Omega;

        V = (Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * Omega + (theta - sin(theta)) / (pow(theta, 3)) * Omega2);
    }

    return SE3Quat(Quaterniond(R), V * upsilon);
}

void ObjectState::UsingVelocitySetPredictPos(const Vector6d& LastVel, const double &delta_t)
{
    Eigen::Vector3d rotation;
    rotation(0) = LastVel(0);
    rotation(1) = LastVel(1);
    rotation(2) = LastVel(2);
    rotation = rotation*delta_t;
    Eigen::Vector3d translation;
    translation(0) = LastVel(3) * delta_t;
    translation(1) = LastVel(4) * delta_t;
    translation(2) = LastVel(5) * delta_t;
    Vector6d delta_pos = Eigen::Matrix<double, 6,1>::Zero();
    delta_pos(0) = rotation(0);
    delta_pos(1) = rotation(1);
    delta_pos(2) = rotation(2);
    g2o::SE3Quat Tlc = g2o::SE3Quat::exp(delta_pos);
    Tlc.setTranslation(translation);
    g2o::SE3Quat vSE3CurrentPose = pose * Tlc;
    pose = vSE3CurrentPose;

}

ObjectState ObjectState::transform_from(const SE3Quat &Twc) const
{
    ObjectState res;
    res.pose = Twc * this->pose;
    res.scale = this->scale;
    return res;
}


/// 计算： [R*Diagonal_scale t; 0 1]， 这里的pose是指世界系
Matrix4d ObjectState::similarityTransform() const
{
    Matrix4d res = pose.to_homogeneous_matrix();
    Matrix3d scale_mat = scale.asDiagonal()*0.5; /// 请注意是half
    //std::cout<<"scale: "<<scale<<std::endl;
    res.topLeftCorner<3, 3>() = res.topLeftCorner<3, 3>() * scale_mat;
    return res;
}

/// 得到8个角点在世界系下的x，y，z坐标。 要求： 目标系原点在方块中心
Matrix3Xd ObjectState::compute3D_BoxCorner() const
{
    Matrix3Xd corners_body;
    corners_body.resize(3, 8);


    switch(ORB_SLAM2::EnObjectCenter)
    {
        case 0: // 几何中心
        {
            /// 3×8(矩阵)， 注意中心点在方体的中心不是在底面
            corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    -1, -1, -1, -1, 1, 1, 1, 1;
            break;
        }

        case 1: // 底面中心, 只有virtual kitti
        {
            corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                    0, 0, 0, 0, -2, -2, -2, -2,
                    1, -1, -1, 1, 1, -1, -1, 1;
            break;
        }

        default:
            assert(0);
    }

    //std::cout<<real_to_homo_coord<double>(corners_body)<<std::endl;

    /// 对于8个顶点，将其从目标系转换到世界系，计算方法： R * Diagonal_scale * corner + t
    /// 解释： similarityTransform() ： [R*Diagonal_scale t; 0 1]
    /// real_to_homo_coord: 在原矩阵增加一行全1
    /// 得到8个顶点在世界系下的坐标
    Matrix3Xd corners_world = homo_to_real_coord<double>(similarityTransform() * real_to_homo_coord<double>(corners_body));
    return corners_world;
}

/// 得到图像平面bbox左上角顶点和右下角顶点坐标：[u_min v_min u_max v_max]
Vector4d ObjectState::projectOntoImageRect(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
{
    /// 得到8顶点在世界系下的坐标
    Matrix3Xd corners_3d_world = compute3D_BoxCorner();
    /// 得到8个顶点在图像平面的坐标
    /// 计算方法： K * [R t; 0 1]（相机pose） * [8个顶点在世界系的齐次坐标]， 结果再转为非齐次就是图像坐标u v
    Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(campose_cw.to_homogeneous_matrix() * real_to_homo_coord<double>(corners_3d_world)));
    /// 得到u_max v_max
    Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // 找出每一行最大的数
    /// 得到u_min v_min
    Vector2d topleft = corner_2d.rowwise().minCoeff();//找出每一行最小的数
    /// 返回[u_min v_min u_max v_max]
    return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
}

/// 得到图像平面bbox左上角顶点和右下角顶点坐标：[u_min v_min u_max v_max]
Vector4d ObjectState::projectOntoImageRectFromCamera(const Matrix3d &Kalib) const
{
    /// 得到8顶点在相机系下的坐标
    Matrix3Xd corners_3d_camera = compute3D_BoxCorner();
    /// 得到8个顶点在图像平面的坐标
    /// 计算方法： K * 相机pose） * [8个顶点在相机系的齐次坐标]， 结果再转为非齐次就是图像坐标u v
    Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(real_to_homo_coord<double>(corners_3d_camera)));
    /// 得到u_max v_max
    Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // 找出每一行最大的数
    /// 得到u_min v_min
    Vector2d topleft = corner_2d.rowwise().minCoeff();//找出每一行最小的数
    /// 返回[u_min v_min u_max v_max]
    return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
}

/// 返回3D object投影过来的bbox信息: [center.x center.y width height]
Vector4d ObjectState::projectOntoImageBbox(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
{
    /// 得到投影过来的bbox的左上角顶点和右下角顶点坐标： [u_min v_min u_max v_max]
    Vector4d rect_project = projectOntoImageRect(campose_cw, Kalib);
    /// 求中心点坐标
    Vector2d rect_center = (rect_project.tail<2>() + rect_project.head<2>()) / 2;
    /// 求bbox的长宽
    Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
    /// 返回[center.u center.v width heigh]
    return Vector4d(rect_center(0), rect_center(1), widthheight(0), widthheight(1));
}




//TODO vertex
// similar as above

void VertexSE3Fix::oplusImpl(const double *update_) {
    Eigen::Map<const Vector6d> update(update_);

    SE3Quat objPose;
    if(whether_fixrollpitch)
    {
        Vector6d update2 = update;
        update2(0) = 0;
        update2(1) = 0;
        objPose = exptwist_norollpitch(update2) * _estimate;
    }
    else{
        objPose = SE3Quat::exp(update) * estimate();
    }

    if(whether_fixheight)
    {
        objPose.setTranslation(Vector3d(objPose.translation()(0), _estimate.translation()(1), objPose.translation()(2))); // 注意这里是Tco, 所以y轴是高度
    }


    //setEstimate(SE3Quat::exp(update)*estimate());
    setEstimate(objPose);
}




void VertexCuboidFixScale::oplusImpl(const double *update_)
{
    Eigen::Map<const Vector6d> update(update_);

    g2o::ObjectState newcube;
    if (whether_fixrotation)
    {
        newcube.pose.setRotation(_estimate.pose.rotation());
        newcube.pose.setTranslation(_estimate.pose.translation() + update.tail<3>());
    }
    else if (whether_fixrollpitch)
    {
        Vector6d update2 = update;
        update2(0) = 0;
        update2(1) = 0;
        newcube.pose =  exptwist_norollpitch(update2) * _estimate.pose;
    }
    else
        newcube.pose =  SE3Quat::exp(update) * _estimate.pose;

    if (whether_fixheight)//TODO 固定高度难道不是动z轴吗，为什么是动y轴，不能理解
        //newcube.setTranslation(Vector3d(newcube.translation()(0), _estimate.translation()(1), newcube.translation()(2)));
        newcube.setTranslation(Vector3d(newcube.translation()(0), newcube.translation()(1), _estimate.translation()(2)));


    if (fixedscale(0) > 0)
        newcube.scale = fixedscale; /// 如果使用统一固定尺度，则每次更新完的尺度都是统一的固定尺度fixedscale
    else
        newcube.scale = _estimate.scale; /// 不使用统一固定尺度， 则每次更新完的尺度都不变(为原3D object的尺度)

    setEstimate(newcube);
}



//TODO edge
void EdgeSE3OnlyCuboidFixScaleProj::computeError()
{
    //const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              //  world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]); //  object pose to world

    //SE3Quat cam_pose_Tcw = SE3Vertex->estimate();
    ObjectState global_cube = cuboidVertex->estimate();

    // 测试 c^T_o = c^T_w * w^T_o 与我离线读的结果是否基本相同
    //cout<< "object in cam: "<<Tcw * global_cube.pose <<endl;


    /// 返回3D object投影过来的bbox信息: [center.x center.y width height]
    Vector4d rect_project = global_cube.projectOntoImageBbox(Tcw, Kalib); // center, width, height
    //Vector2d left_corner(rect_project(0)-1/2*rect_project();
    //Vector2d right_corner;

    // 没有用过
    //Vector4d rect_project_new(rect_project(0), rect_project(1), 1.1* rect_project(2), 1.1*rect_project(3));

    //cout<<rect_project<<endl<<endl<<_measurement<<endl;
    //_error = rect_project_new - _measurement;
    Vector4d compensation(-0.144723, -0.146526, 2.47732, 2.40297);
    //_error = rect_project - _measurement + compensation;
    _error = rect_project - _measurement;
}

Vector4d EdgeSE3OnlyCuboidFixScaleProj::computeError_debug(double &chi)
{
    computeError();
    chi = chi2();
    return _error;
}

double EdgeSE3OnlyCuboidFixScaleProj::get_error_norm()
{
    computeError();
    return _error.norm();
}

void EdgeSE3CuboidFixScaleProj::computeError()
{
    const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              //  world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[1]); //  object pose to world

    SE3Quat cam_pose_Tcw = SE3Vertex->estimate();
    ObjectState global_cube = cuboidVertex->estimate();

    Vector4d rect_project = global_cube.projectOntoImageBbox(cam_pose_Tcw, Kalib); // center, width, height
    //_error = rect_project - _measurement;
    /// 补偿
    Vector4d compensation(-0.144723, -0.146526, 2.47732, 2.40297);
    _error = rect_project - _measurement + compensation;
}

Vector4d EdgeSE3CuboidFixScaleProj::computeError_debug(double &chitmp)
{
    computeError();
    chitmp = chi2();
    return _error;
}

double EdgeSE3CuboidFixScaleProj::get_error_norm()
{
    computeError();
    return _error.norm();
}


void EdgeDynamicPointOnlyCuboid::computeError()
{
    //const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              // world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[1]);        // point to object pose

    Vector3d localpt = Tcw* (cuboidVertex->estimate().pose * pointVertex->estimate());

    Vector2d projected = Vector2d(Kalib(0, 2) + Kalib(0, 0) * localpt(0) / localpt(2), Kalib(1, 2) + Kalib(1, 1) * localpt(1) / localpt(2));
    _error = _measurement - projected;

}


void EdgeDynamicPointOnlyCuboid::linearizeOplus()
{
    //const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              // world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[1]);        // point to object pose

    Vector3d objectpt = pointVertex->estimate();
    SE3Quat combinedT = Tcw * cuboidVertex->estimate().pose;//c^T_o
    Vector3d camerapt = combinedT * objectpt;//c^p_f

    double fx = Kalib(0, 0);
    double fy = Kalib(1, 1);

    double x = camerapt[0];
    double y = camerapt[1];
    double z = camerapt[2];
    double z_2 = z * z;

    Matrix<double, 2, 3> projptVscamerapt;
    projptVscamerapt(0, 0) = fx / z;
    projptVscamerapt(0, 1) = 0;
    projptVscamerapt(0, 2) = -x * fx / z_2;

    projptVscamerapt(1, 0) = 0;
    projptVscamerapt(1, 1) = fy / z;
    projptVscamerapt(1, 2) = -y * fy / z_2;

    // jacobian of point
    _jacobianOplusXj = -projptVscamerapt * combinedT.rotation().toRotationMatrix();



    // jacobian of object pose.   obj twist  [angle position]
    Matrix<double, 3, 6> temp;
    Vector3d Pwf = cuboidVertex->estimate().pose *objectpt;
    Matrix3d R = Tcw.rotation().toRotationMatrix();
    temp.leftCols<3>() = - R*skew(Pwf);
    temp.rightCols<3>() = R;
    _jacobianOplusXi = -projptVscamerapt * temp;



    if (cuboidVertex->whether_fixrollpitch)
    {
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = 0;
    }
    if (cuboidVertex->whether_fixrotation)
    {
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(0, 2) = 0;
        _jacobianOplusXi(1, 2) = 0;
    }
}

Vector2d EdgeDynamicPointOnlyCuboid::computeError_debug(double &chitmp)
{
    computeError();
    chitmp = chi2();
    return _error;
}

void EdgeStereoDynamicPointAndCuboid::computeError()
{
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[1]);        // point to object pose

    Vector3d localpt = Tcw* (cuboidVertex->estimate().pose * pointVertex->estimate());
    Vector3d obs(_measurement);
    _error = obs - cam_project(localpt);
}

Vector3d EdgeStereoDynamicPointAndCuboid::computeError_debug(double &chitmp) {
    computeError();
    chitmp = chi2();
    return _error;
}

Vector3d EdgeStereoDynamicPointAndCuboid::cam_project(const Vector3d &trans_xyz) const{
    const float invz = 1.0f/trans_xyz[2];
    Vector3d res;
    res[0] = trans_xyz[0] * invz * Kalib(0,0) + Kalib(0,2);
    res[1] = trans_xyz[1] * invz * Kalib(1,1) + Kalib(1,2);
    res[2] = res[0] - bf*invz;
    return res;
}

void EdgeStereoDynamicPointAndCuboid::linearizeOplus()
{
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[1]);        // point to object pose

    Vector3d objectpt = pointVertex->estimate();
    SE3Quat combinedT = Tcw * cuboidVertex->estimate().pose;//c^T_o
    Vector3d camerapt = combinedT * objectpt;//c^p_f

    double fx = Kalib(0, 0);
    double fy = Kalib(1, 1);

    double x = camerapt[0];
    double y = camerapt[1];
    double z = camerapt[2];
    double z_2 = z * z;

    Matrix<double, 3, 3> projptVscamerapt; //2d projected pixel / 3D local camera pt
    projptVscamerapt(0, 0) = fx / z;
    projptVscamerapt(0, 1) = 0;
    projptVscamerapt(0, 2) = -x * fx / z_2;

    projptVscamerapt(1, 0) = 0;
    projptVscamerapt(1, 1) = fy / z;
    projptVscamerapt(1, 2) = -y * fy / z_2;

    projptVscamerapt(2, 0) = fx/z;
    projptVscamerapt(2,1) = 0;
    projptVscamerapt(2, 2) = (-fx*x+bf)/z_2;


    Matrix<double, 3, 6> temp;
    Vector3d Pwf = cuboidVertex->estimate().pose *objectpt;
    Matrix3d R = Tcw.rotation().toRotationMatrix();
    temp.leftCols<3>() = - R*skew(Pwf);
    temp.rightCols<3>() = R;

    // object Jacobian
    _jacobianOplusXi = -projptVscamerapt * temp;
    // landmark Jacobian
    _jacobianOplusXj = -projptVscamerapt * combinedT.rotation().toRotationMatrix();

    if (cuboidVertex->whether_fixrotation)
    {
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(0, 2) = 0;
        _jacobianOplusXi(1, 2) = 0;
    }
}





void EdgeDynamicPointCuboidCamera::computeError()
{
    const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              // world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[1]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[2]);        // point to object pose
    Vector3d localpt = SE3Vertex->estimate()* (cuboidVertex->estimate().pose * pointVertex->estimate());
    Vector2d projected = Vector2d(Kalib(0, 2) + Kalib(0, 0) * localpt(0) / localpt(2), Kalib(1, 2) + Kalib(1, 1) * localpt(1) / localpt(2));
    _error = _measurement - projected;

    if(0)
    {
        jishu_temp++;
        cout << "jishu:  " << jishu_temp << endl;
        cout << "cameraid:  " << _vertices[0]->id() << "  objectid:  " << _vertices[1]->id() << "  point_id:  "
             << _vertices[2]->id() << endl;
        cout << "Twc:  " << SE3Vertex->estimate().inverse() << "   Two:  " << cuboidVertex->estimate().pose
             << "  po:  " << pointVertex->estimate() << "  obs:  " << _measurement << " proj:  " << projected
             << endl;
        cout << "error:  " << _error << endl;
        if (_error.norm() > 6)
            exit(0);
    }

}

void EdgeDynamicPointCuboidCamera::linearizeOplus()
{
    const VertexSE3Expmap *SE3Vertex = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);              // world to camera pose
    const VertexCuboidFixScale *cuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[1]); // object to world pose
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[2]);        // point to object pose

    Vector3d objectpt = pointVertex->estimate();
    SE3Quat combinedT = SE3Vertex->estimate() * cuboidVertex->estimate().pose;
    Vector3d camerapt = combinedT * objectpt;

    double fx = Kalib(0, 0);
    double fy = Kalib(1, 1);

    double x = camerapt[0];
    double y = camerapt[1];
    double z = camerapt[2];
    double z_2 = z * z;

    Matrix<double, 2, 3> projptVscamerapt; //2d projected pixel / 3D local camera pt
    projptVscamerapt(0, 0) = fx / z;
    projptVscamerapt(0, 1) = 0;
    projptVscamerapt(0, 2) = -x * fx / z_2;

    projptVscamerapt(1, 0) = 0;
    projptVscamerapt(1, 1) = fy / z;
    projptVscamerapt(1, 2) = -y * fy / z_2;

    // jacobian of point
    _jacobianOplus[2] = -projptVscamerapt * combinedT.rotation().toRotationMatrix();

    // jacobian of camera
    _jacobianOplus[0](0, 0) = x * y / z_2 * fx;
    _jacobianOplus[0](0, 1) = -(1 + (x * x / z_2)) * fx;
    _jacobianOplus[0](0, 2) = y / z * fx;
    _jacobianOplus[0](0, 3) = -1. / z * fx;
    _jacobianOplus[0](0, 4) = 0;
    _jacobianOplus[0](0, 5) = x / z_2 * fx;

    _jacobianOplus[0](1, 0) = (1 + y * y / z_2) * fy;
    _jacobianOplus[0](1, 1) = -x * y / z_2 * fy;
    _jacobianOplus[0](1, 2) = -x / z * fy;
    _jacobianOplus[0](1, 3) = 0;
    _jacobianOplus[0](1, 4) = -1. / z * fy;
    _jacobianOplus[0](1, 5) = y / z_2 * fy;

    // jacobian of object pose.   obj twist  [angle position]
    ///注意这是右导数
    Matrix<double, 3, 6> skewjaco;
    skewjaco.leftCols<3>() = -skew(objectpt);
    skewjaco.rightCols<3>() = Matrix3d::Identity();
    _jacobianOplus[1] = _jacobianOplus[2] * skewjaco; //2*6

    ///左导数
    Matrix<double, 3, 6> temp;
    Vector3d Pwf = cuboidVertex->estimate().pose *objectpt;
    Matrix3d R = SE3Vertex->estimate().rotation().toRotationMatrix();
    temp.leftCols<3>() = - R*skew(Pwf);
    temp.rightCols<3>() = R;
    //_jacobianOplus[1] = -projptVscamerapt * temp;


    if (cuboidVertex->whether_fixrollpitch)           //zero gradient for roll/pitch
    {
        _jacobianOplus[1](0, 0) = 0;
        _jacobianOplus[1](0, 1) = 0;
        _jacobianOplus[1](1, 0) = 0;
        _jacobianOplus[1](1, 1) = 0;
    }
    if (cuboidVertex->whether_fixrotation)
    {
        _jacobianOplus[1](0, 0) = 0;
        _jacobianOplus[1](0, 1) = 0;
        _jacobianOplus[1](1, 0) = 0;
        _jacobianOplus[1](1, 1) = 0;
        _jacobianOplus[1](0, 2) = 0;
        _jacobianOplus[1](1, 2) = 0;
    }
}

Vector2d EdgeDynamicPointCuboidCamera::computeError_debug(double &chi2tmp)
{
    computeError();
    chi2tmp = chi2();
    return _error;
}

void EdgeStereoDynamicPointAndCuboidAndCamera::computeError()
{
    const VertexSE3Expmap *vertexCamera = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);
    const VertexCuboidFixScale *vertexCuboid = dynamic_cast<const VertexCuboidFixScale*>(_vertices[1]);
    const VertexSBAPointXYZ *vertexPoint = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[2]);

    //cout<<vertexCamera->estimate().translation()<<endl;
    Vector3d ePointPosInObj = vertexPoint->estimate();
    Vector3d ePointPosInCam = vertexCamera->estimate() * vertexCuboid->estimate().pose * ePointPosInObj;
    Vector3d eProjected;
    const float invz = 1.0/ePointPosInCam[2];
    eProjected[0] = Kalib(0, 2) + ePointPosInCam[0] * invz * Kalib(0, 0);
    eProjected[1] = Kalib(1, 2) + ePointPosInCam[1] * invz * Kalib(1, 1);
    eProjected[2] = eProjected[0] - bf * invz;
    Vector3d obs(_measurement);
    _error = obs - eProjected;
    //cout<<"error: "<<_error<<endl;
}

Vector3d EdgeStereoDynamicPointAndCuboidAndCamera::computeError_debug(double &chi2tmp)
{
    computeError();
    chi2tmp = chi2();
    return _error;
}


/*
void EdgeStereoDynamicPointAndCuboidAndCamera::linearizeOplus()
{
    const VertexSE3Expmap *vertexCamera = dynamic_cast<const VertexSE3Expmap *>(_vertices[0]);
    const VertexCuboidFixScale *vertexCuboid = dynamic_cast<const VertexCuboidFixScale*>(_vertices[1]);
    const VertexSBAPointXYZ *vertexPoint = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[2]);

    Vector3d ePointPosInObj = vertexPoint->estimate();
    Vector3d ePointPosInCam = vertexCamera->estimate() * vertexCuboid->estimate().pose * ePointPosInObj;
    Vector3d ePointPosInWolrd = vertexCuboid->estimate().pose * ePointPosInObj;
    double x = ePointPosInCam[0];
    double y = ePointPosInCam[1];
    double z = ePointPosInCam[2];
    double z_2 = z * z;
    double fx = Kalib(0, 0);
    double fy = Kalib(1, 1);

    Matrix<double, 3, 3> ProjectionJacobian;
    ProjectionJacobian(0, 0) = fx / z;
    ProjectionJacobian(0, 1) = 0;
    ProjectionJacobian(0, 2) = -fx * x /z_2;
    ProjectionJacobian(1, 0) = 0;
    ProjectionJacobian(1, 1) = fy/z;
    ProjectionJacobian(1, 2) = -fy*y/z_2;
    ProjectionJacobian(2, 0) = fx/z;
    ProjectionJacobian(2, 1) = 0;
    ProjectionJacobian(2, 2) = (bf - fx*x)/z_2;

    Matrix<double, 3, 6> PartJacobianForCamera;
    PartJacobianForCamera.leftCols<3>() = -skew(ePointPosInCam);
    PartJacobianForCamera.rightCols<3>() = Matrix3d::Identity();
    _jacobianOplus[0] = -ProjectionJacobian * PartJacobianForCamera;

    Matrix<double, 3, 6> PartJacobianForObject;
    PartJacobianForObject.leftCols<3>() = -skew(ePointPosInWolrd);
    PartJacobianForObject.rightCols<3>() = Matrix3d::Identity();
    _jacobianOplus[1] = -ProjectionJacobian * vertexCamera->estimate().rotation().toRotationMatrix() * PartJacobianForObject;

    _jacobianOplus[2] = -ProjectionJacobian * vertexCamera->estimate().rotation().toRotationMatrix() * vertexCuboid->estimate().pose.rotation().toRotationMatrix();

    if(vertexCuboid->whether_fixrollpitch)
    {
        _jacobianOplus[1](0, 0) = 0;
        _jacobianOplus[1](0, 1) = 0;
        _jacobianOplus[1](1, 0) = 0;
        _jacobianOplus[1](1, 1) = 0;
        _jacobianOplus[1](2, 0) = 0;
        _jacobianOplus[1](2, 1) = 0;
    }
    else if(vertexCuboid->whether_fixrotation)
    {
        _jacobianOplus[1](0, 0) = 0;
        _jacobianOplus[1](0, 1) = 0;
        _jacobianOplus[1](0, 2) = 0;
        _jacobianOplus[1](1, 0) = 0;
        _jacobianOplus[1](1, 1) = 0;
        _jacobianOplus[1](1, 2) = 0;
        _jacobianOplus[1](2, 0) = 0;
        _jacobianOplus[1](2, 1) = 0;
        _jacobianOplus[1](2, 2) = 0;
    }
}
*/



void UnaryLocalPoint::computeError()
{
    // 该点的目标系位置
    const VertexSBAPointXYZ *pointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    Vector3d local_pt = pointVertex->estimate().cwiseAbs();// 应该是每个元素求绝对值
    Vector3d point_edge_error;

    // 遍历每一维
    for (int i = 0; i < 3; i++)
    {
        // 注意每一维是否与objectscale相对应？
        // 若在尺度范围内， 则误差为0
        if (local_pt(i) < objectscale(i))
            point_edge_error(i) = 0;
        // 若在(超出max_outside_margin_ratio)倍尺度内则误差直接为超出部分的长度
        else if (local_pt(i) < (max_outside_margin_ratio + 1) * objectscale(i))
            point_edge_error(i) = local_pt(i) - objectscale(i);
        // 若在(超出max_outside_margin_ratio)倍尺度范围之外，则误差为max_outside_margin_ratio倍长度
        // 注意这是一个常数误差， 就不会去优化object point在目标系下的位置
        else
            point_edge_error(i) = max_outside_margin_ratio * objectscale(i);
    }

    _error = point_edge_error.array() / objectscale.array();
    //cout<<"error: "<<_error<<endl;
}

Vector3d UnaryLocalPoint::computeError_debug()
{
    computeError();
    return _error;
}


void EdgeCurrentObjectMotion::computeError()
{
    //const VertexCuboidFixScale *cuboidVertexfrom = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]);   // object to world pose
    const VertexCuboidFixScale *cuboidVertexto = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]);     // object to world pose
    const VelocityPlanarVelocity *velocityVertex = dynamic_cast<const VelocityPlanarVelocity *>(_vertices[1]); // object to world pose

    if (cuboidVertexto == nullptr || velocityVertex == nullptr)
        cout << "Bad casting when compute Edge motion error!!!!!!!!!!!!!" << endl;

    // predict motion x y yaw and compute measurement.
    SE3Quat posefrom = lastpose;


    double yaw_from = posefrom.toXYZPRYVector()(5);

    SE3Quat poseto = cuboidVertexto->estimate().pose;
    double yaw_to = poseto.toXYZPRYVector()(5);

    Vector2d velocity = velocityVertex->estimate(); //v w   linear velocity and steer angle

    const double vehicle_length = 0.15; // front and back wheels distance,0.25,2.71
    // vehicle motion model is applied to back wheel center
    Vector3d trans_back_pred = posefrom.translation() + (velocity(0) * delta_t - vehicle_length * 0.5) * Vector3d(cos(yaw_from), sin(yaw_from), 0);
    double yaw_pred = yaw_from + tan(velocity(1)) * delta_t / vehicle_length * velocity(0);

    // as mentioned in paper: my object frame is at the center. the motion model applies to back wheen center. have offset.
    Vector3d trans_pred = trans_back_pred + vehicle_length * 0.5 * Vector3d(cos(yaw_pred), sin(yaw_pred), 0);

    _error = Vector3d(poseto.translation()[0], poseto.translation()[1], yaw_to) - Vector3d(trans_pred(0), trans_pred(1), yaw_pred);
    if (_error[2] > 2.0 * M_PI)
        _error[2] -= 2.0 * M_PI;
    if (_error[2] < -2.0 * M_PI)
        _error[2] += 2.0 * M_PI;
}

Vector3d EdgeCurrentObjectMotion::computeError_debug()
{
    computeError();
    return _error;
}

void EdgeMotionModel::computeError()
{
    const VertexCuboidFixScale *LastCuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[0]);
    const VertexCuboidFixScale *CurrentCuboidVertex = dynamic_cast<const VertexCuboidFixScale *>(_vertices[1]);
    const VertexVelocitySixDofForObject *LastVelVertex = dynamic_cast<const VertexVelocitySixDofForObject *>(_vertices[2]);
    //const VertexSBAPointXYZ *PointVertex = dynamic_cast<const VertexSBAPointXYZ *>(_vertices[3]);

    // 用速度算两帧object pose变换 Tlc
    Vector3d rotation;
    rotation(0) = LastVelVertex->estimate()(0);
    rotation(1) = LastVelVertex->estimate()(1);
    rotation(2) = LastVelVertex->estimate()(2);
    rotation = rotation*delta_t;
    Vector3d translation;
    translation(0) = LastVelVertex->estimate()(3) * delta_t;
    translation(1) = LastVelVertex->estimate()(4) * delta_t;
    translation(2) = LastVelVertex->estimate()(5) * delta_t;
    Vector6d delta_pos = Eigen::Matrix<double, 6,1>::Zero();
    delta_pos(0) = rotation(0);
    delta_pos(1) = rotation(1);
    delta_pos(2) = rotation(2);
    SE3Quat Tlc = SE3Quat::exp(delta_pos);
    Tlc.setTranslation(translation);

    //_error = CurrentCuboidVertex->estimate().pose * PointVertex->estimate() - LastCuboidVertex->estimate().pose * Tlc * PointVertex->estimate();

    _error = (CurrentCuboidVertex->estimate().pose.inverse() * LastCuboidVertex->estimate().pose * Tlc).toMinimalVector();
}
//void EdgeTransConstraintFromDetction::linearizeOplus()
//{
//
//}


} // namespace g2o
