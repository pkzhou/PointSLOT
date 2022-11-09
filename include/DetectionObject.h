//
// Created by liuyuzhen on 2020/5/23.
//
#pragma once
#ifndef ORB_SLAM2_CUBOID3D_H
#define ORB_SLAM2_CUBOID3D_H

//#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
//#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "g2o_Object.h"
//#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <mutex>
using namespace  std;
namespace ORB_SLAM2 {
    class MapObject;
    class MapObjectPoint;
    class DetectionObject {
    public:
        DetectionObject(const int &pFrameId, const int &nObjectID, const bool &bDynamicFlag, const double &dMonoDynaVal,
                        const double &dStereoDynaVal, const double &dTruncated, const double &dOcculuded, const double &dAlpha, const cv::Rect &rectBBox, const Eigen::Vector4d & eigBBox,
                        const Eigen::Vector3d &eigScale, const Eigen::Vector3d &Position, const double &dRotY, const double &dMeasQuality);

        DetectionObject(const long unsigned int &pFrameId, const Eigen::Matrix<double, 1, 24> &eigDetection);

        DetectionObject(const long unsigned int &pFrameId, const int &object_id, const cv::Rect &BBox, const Eigen::Vector3d &scale, const Eigen::Vector3d &position, const Eigen::Vector3d& rotation);

        // 变量: 基本属性
        long unsigned int mnFrameID;
        int mnObjectID;
        bool mbDynamicFlag;
        double mdMonoDynaVal;
        double mdStereoDynaVal;
        double mdTruncated; // 如果是virtual kitti， 则是存的truncation_ratio[0, 1] 0代表no truncation, 1代表完全truncated
        double mdOcculuded; // 如果是virtual kitti, 则存的是occupany_ration,[0, 1] 1代表完全没有遮挡， 0代表完全遮挡
        double mdAlpha;
        cv::Rect mrectBBox;// 左上角坐标 width height
        //Eigen::Vector4d mBBox;// center, width, height
        //Eigen::Matrix2Xi box_corners_2d;       // 2*8, 这个怎么赋值， 用处是什么
        Eigen::Vector3d mScale; // length, height, width 就是直接对应目标系的xyz
        Eigen::Vector3d mPos;
        double mdRotY;
        double mdRotX;
        double mdRotZ;
        double mdMeasQuality;            // [0,1] the higher, the better
        bool mbGoodDetFlag;
        int IsRecovered = 0;
        bool TooFar;



        // 优化使用
        bool mbBBoxOutlier;
        int mnBALocalForTemporalWindow;
        int mnOptimizeVertexId;
        int mnVelOptimizeVertexId;

        // groundtruth
        g2o::ObjectState mTruthPosInWolrdFrame; // debug使用
        g2o::ObjectState mTruthPosInCameraFrame;

        // 估计值:
        g2o::ObjectState mInWPos;
        g2o::ObjectState mInCPos;


        // 关联属性, 好坏属性
        std::vector<MapObjectPoint*> mvMapPoints;
        MapObject* mBelongObject = nullptr;
        bool mbFewMOPsFlag;
        int mnMatchesInliers; //Current matches in frame

        bool mbTrackOK; // 跟踪标志
        bool mbNeedCreateNewOKFFlag;
        bool mInitflag = 0;

        void SetFewMOPsFlag(bool bFewMOPsFlag);
        MapObject* GetMapObject();
        void SetMapObject(MapObject* pMO);
        cv::Rect GetBBoxRect();
        void InitInFrameMapObjectPointsOrder(const size_t &nNum);
        void AddMapObjectPoint(const int &nInDetObjMOPOrder, MapObjectPoint* pMP);
        void EraseMapObjectPoint(const int &nInDetObjMOPOrder);
        //vector<MapObjectPoint*> GetInFrameMapObjectPoints();
        void SetDynamicFlag(const double &dMonoDynVal, const double &dStereoDynVal);
        void SetDynamicFlag(bool bDynamicFlag);
        bool GetDynamicFlag();

    protected:
        std::mutex mMutexMapObject;
        std::mutex mMutexKeyPoints;
        std::mutex mMutexMapPoints;
        std::mutex mMutexDynVal;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    };
}

#endif //ORB_SLAM2_CUBOID3D_H
