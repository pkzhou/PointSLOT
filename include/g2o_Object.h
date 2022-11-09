//
// Created by liuyuzhen on 2020/5/24.
//

#ifndef ORB_SLAM2_G2O_OBJECT_H
#define ORB_SLAM2_G2O_OBJECT_H

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "matrix_utils.h"
//#include "Parameters.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm> // std::swap

/// 这个文件相当于是对目标 g2o优化定义顶点和边

namespace g2o
{
    typedef Eigen::Matrix<double, 9, 1> Vector9d;
    typedef Eigen::Matrix<double, 9, 9> Matrix9d;
    typedef Eigen::Matrix<double, 10, 1> Vector10d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    using namespace Eigen;
    class ObjectState
    {
    public:

        SE3Quat pose;
        Vector3d scale; /// 是真正的scale，不是half！！！, 按照xyz存的即： length height width
        ObjectState()
        {
            pose =SE3Quat();
            scale.setZero();
        }
        ObjectState(const g2o::SE3Quat &se3Pose, const Vector3d &eigScale)
        {
            pose = se3Pose;
            scale = eigScale;
        }
        // xyz roll pitch yaw half_scale
        inline void fromMinimalVector(const Vector9d &v)
        {
            Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
            pose = SE3Quat(posequat, v.head<3>());
            scale = v.tail<3>();
        }
        inline const Vector3d &translation() const { return pose.translation(); }
        inline void setPose(const g2o::SE3Quat &se3Pose){pose = se3Pose;}
        inline void setScale(const Vector3d &scale_) { scale = scale_; }
        inline void setTranslation(const Vector3d &t_) { pose.setTranslation(t_); }
        inline void setRotation(const Quaterniond &r_) { pose.setRotation(r_); }
        inline void setRotation(const Matrix3d &R) { pose.setRotation(Quaterniond(R)); }
        inline void setRotation(const Vector3d &v)
        {
            Eigen::Quaterniond rotquat = zyx_euler_to_quat(v(0), v(1), v(2));

            pose.setRotation(rotquat);
            pose.normalizeRotation();
        }

        ObjectState transform_from(const SE3Quat &Twc) const;

        void UsingVelocitySetPredictPos(const Vector6d& LastVel, const double &delta_t);


        /// 计算： [R*Diagonal_scale t; 0 1]， 这里的pose是指世界系
        Matrix4d similarityTransform() const;


        /// 得到8个角点在世界系下的x，y，z坐标。 要求： 目标系原点在方块中心
        Matrix3Xd compute3D_BoxCorner() const;


        /// 得到图像平面bbox左上角顶点和右下角顶点坐标：[u_min v_min u_max v_max]
        Vector4d projectOntoImageRect(const SE3Quat &campose_cw, const Matrix3d &Kalib) const;

        /// 得到图像平面bbox左上角顶点和右下角顶点坐标：[u_min v_min u_max v_max]
        Vector4d projectOntoImageRectFromCamera(const Matrix3d &Kalib) const;


        /// 返回3D object投影过来的bbox信息: [center.x center.y width height]
        Vector4d projectOntoImageBbox(const SE3Quat &campose_cw, const Matrix3d &Kalib) const;


    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };


    //TODO 顶点!!!!
    class VertexCuboidFixScale : public BaseVertex<6, ObjectState> // less variables. should be faster  fixed scale should be set
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexCuboidFixScale()
        {
            fixedscale.setZero();
            whether_fixrollpitch = false;
            whether_fixrotation = false;
            whether_fixheight = false;
        };

        virtual void setToOriginImpl()
        {
            _estimate = ObjectState();
            if (fixedscale(0) > 0)
                _estimate.scale = fixedscale;
        }

        virtual void oplusImpl(const double *update_);

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        Vector3d fixedscale;
        bool whether_fixrollpitch;
        bool whether_fixrotation;
        bool whether_fixheight;
    };

    class  VertexSE3Fix : public BaseVertex<6, SE3Quat>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSE3Fix(){
            whether_fixrollpitch = false;
            whether_fixheight = false;
        };

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        virtual void setToOriginImpl() {
            _estimate = SE3Quat();
        }

        virtual void oplusImpl(const double* update_);

        bool whether_fixrollpitch;
        bool whether_fixheight;
    };


    /// 新的速度顶点， 3维角速度， 3维线速度
    class VertexVelocitySixDofForObject : public BaseVertex<6, Vector6d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVelocitySixDofForObject()
        {
            bWhetherFixHeight = true;
            bWhetherFixRollPitch = true;
            bWhetherFixRotation = false;
        };

        virtual void setToOriginImpl()
        {
            _estimate.fill(0);
        }

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        //  位姿为什么要用T SEQ3来表示， 为什么不直接用6自由度的向量？？？， 像这里的速度一样
        virtual void oplusImpl(const double *update)
        {
            Eigen::Map<const Vector6d> v(update);
            Vector6d update2 = v;

            if(bWhetherFixHeight)
                update2(1) = 0; // y轴方向

            if(bWhetherFixRotation)
            {
                update2(3) = 0;
                update2(4) = 0;
                update2(5) = 0;

            }
            else if(bWhetherFixRollPitch)
            {
                update2(3) = 0;
                update2(4) = 0;
            }

            _estimate += update2;
        }

        bool bWhetherFixRollPitch;
        bool bWhetherFixRotation;
        bool bWhetherFixHeight;
    };


    // vehicle planar velocity  2Dof   [linear_velocity, steer_angle]
    class VelocityPlanarVelocity : public BaseVertex<2, Vector2d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VelocityPlanarVelocity(){};

        virtual void setToOriginImpl()
        {
            _estimate.fill(0.);
        }

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        virtual void oplusImpl(const double *update)
        {
            Eigen::Map<const Vector2d> v(update);
            _estimate += v;
        }
    };




    //TODO 边!!!!!!!!!!!!!
    class EdgeSE3OnlyCuboidFixScaleProj : public BaseUnaryEdge <4, Vector4d, VertexCuboidFixScale>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeSE3OnlyCuboidFixScaleProj(){};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector4d computeError_debug(double& chi);
        double get_error_norm();
        Matrix3d Kalib;

        SE3Quat Tcw;
    };

    // camera -fixscale_object 2D projection error, rectangle, could also change to iou
    class EdgeSE3CuboidFixScaleProj : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexCuboidFixScale>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeSE3CuboidFixScaleProj(){};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector4d computeError_debug(double &chitmp);
        double get_error_norm();
        Matrix3d Kalib;
    };



    /// 单目，仅优化object和动态landmark
    class EdgeDynamicPointOnlyCuboid : public BaseBinaryEdge<2, Vector2d, VertexCuboidFixScale, VertexSBAPointXYZ>//,multiedge多元边，2是测量值的维度，vector是测量值的数据类型
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeDynamicPointOnlyCuboid() {};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector2d computeError_debug(double &chitmp);
        virtual void linearizeOplus(); // override linearizeOplus to compute jacobians

        double get_error_norm(bool print_details = false);

        Matrix3d Kalib;
        SE3Quat Tcw;

    };

    /// 双目，仅优化object和动态landmark
    class EdgeStereoDynamicPointAndCuboid : public BaseBinaryEdge<3, Vector3d, VertexCuboidFixScale, VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeStereoDynamicPointAndCuboid() {};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector3d computeError_debug(double &chitmp);
        virtual void linearizeOplus();

        Vector3d cam_project(const Vector3d & trans_xyz) const;

        Matrix3d Kalib;
        double bf;
        SE3Quat Tcw;
    };



    ///point-object-camera, 同时优化相机,动态点和object
    class EdgeDynamicPointCuboidCamera : public BaseMultiEdge<2, Vector2d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeDynamicPointCuboidCamera() { resize(3); };

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector2d computeError_debug(double &chi2tmp);
        virtual void linearizeOplus(); // override linearizeOplus to compute jacobians

        double get_error_norm(bool print_details = false);

        Matrix3d Kalib;
    };

    // 双目： point-object-camera, 同时优化相机objectpoint
    class EdgeStereoDynamicPointAndCuboidAndCamera : public BaseMultiEdge<3, Vector3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeStereoDynamicPointAndCuboidAndCamera() {resize(3);}

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };
        void computeError();
        Vector3d computeError_debug(double &chi2tmp);
        //virtual void linearizeOplus();
        Matrix3d Kalib;
        double bf;
    };



    // a local point in object frame. want it to lie inside cuboid. only point    object dimension is fixed. basically project the point onto surface
    class UnaryLocalPoint : public BaseUnaryEdge<3, Vector3d, VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        UnaryLocalPoint(){};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();

        Vector3d computeError_debug();

        Vector3d objectscale;			 // fixed object dimensions
        double max_outside_margin_ratio; // truncate the error if point is too far from object
    };

    class EdgeCurrentObjectMotion : public BaseBinaryEdge<3, Vector3d, VertexCuboidFixScale, VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeCurrentObjectMotion(){};

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError();
        Vector3d computeError_debug();

        double delta_t; // to_time - from_time   positive   velocity needs time
        double get_error_norm(bool print_details = false);
        SE3Quat lastpose;

    };

    class EdgeSmoothTerm : public BaseBinaryEdge<6, Vector6d, VertexVelocitySixDofForObject, VertexVelocitySixDofForObject>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeSmoothTerm(){};
        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };
        void computeError(){
            const VertexVelocitySixDofForObject *LastVeclocity = dynamic_cast<const VertexVelocitySixDofForObject *>(_vertices[0]);
            const VertexVelocitySixDofForObject *CurrentVeclocity = dynamic_cast<const VertexVelocitySixDofForObject *>(_vertices[1]);
            _error = LastVeclocity->estimate() - CurrentVeclocity->estimate();
        };
        // Jacobian 怎么计算， 需要写吗
    };

    // 上一帧pose， 当前帧pose， 上一帧速度，landmark position
    // 我觉得landmark 这个顶点可以不作为顶点， 作为一个定值？
    class EdgeMotionModel : public BaseMultiEdge<6, Vector6d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeMotionModel(){resize(3);}; // 如果一条边有多个顶点，要在这里说明
        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };
        void computeError();
        double delta_t;
    };

    class EdgeTransConstraintFromDetction: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeTransConstraintFromDetction(){}

        virtual bool read(std::istream &is) { return true; };
        virtual bool write(std::ostream &os) const { return os.good(); };

        void computeError()  {
            const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
            Vector3d obs(_measurement);
            _error = obs-v1->estimate().translation();
        }
        //virtual void linearizeOplus();
    };

}



#endif //ORB_SLAM2_G2O_OBJECT_H
