/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "deepsort.h"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
#define Max_OBJ_NUM 50


class MapPoint;
class MapObjectPoint;
class KeyFrame;
class MapObject;
class DetectionObject;
class ObjectKeyFrame;
class Detector;
class Frame
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    // SLAM模式
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // 语义动态SLAM模式
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat& imColor,
          const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
          Detector* YOLODetector);

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
          const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
          Detector* YOLODetector, cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers);


    // 目标跟踪模式
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
                 const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
                 cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers);

    // 自动驾驶模式
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat& imColor,
          const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
          Detector* YOLODetector, DS::DeepSort* deepSort);

    // 终极测试模式
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat& imColor,
          const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
          const int& SLOTMode);


    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Extract ORB on the Object
    void ExtractObjORB(cv::Mat left, cv::Mat right, vector<DetectionObject*> &vDetectionObjects);

    // Yolo detection
    void DetectYOLO(cv::Mat &imColor,vector<DetectionObject*> &vDetectionObjects, Detector* YOLODetector, DS::DeepSort* deepSort, cv::Mat left, cv::Mat right);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);


    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;


    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    void OfflineObjectPoseInit(cv::Rect YOLODet, Eigen::Vector3d *initPosition,Eigen::Vector3d *initRotation,Eigen::Vector3d *scale);


    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

    // Camera pose.
    cv::Mat mTwc;

    //g2o::SE3Quat mSETwc;
    g2o::SE3Quat mSETcw;


public:
    // object  参数
    std::vector<DetectionObject*> mvDetectionObjects;
    std::vector<std::vector<cv::KeyPoint>> mvObjKeysUn;
    std::vector<std::vector<cv::KeyPoint>> mvObjKeys;
    std::vector<std::vector<cv::KeyPoint>> mvObjKeysRight;
    std::vector<std::vector<MapObjectPoint *>> mvpMapObjectPoints;
    vector<cv::Mat> mvObjPointsDescriptors;
    vector<cv::Mat> mvObjPointsDescriptorsRight;
    std::vector<std::vector<float>> mvuObjKeysRight;
    std::vector<std::vector<float>> mvObjPointDepth;
    std::vector<MapObject *> mvMapObjects;
    std::vector<std::vector<bool>> mvbObjKeysOutlier;
    size_t mnDetObj;
    vector<vector<vector<std::vector<std::size_t>>>> mvObjKeysGrid; // 每帧最多观测到50个object
    vector<vector<bool>> mvbObjKeysMatchedFlag;

    // 稠密目标特征点临时容器
    std::vector<cv::KeyPoint> mvTempObjKeys;
    std::vector<cv::KeyPoint> mvTempObjKeysUn;
    std::vector<cv::KeyPoint> mvTempObjKeysRight;
    vector<float> mvTempObjDepth;
    vector<float> mvuTempObjKeysRight;
    cv::Mat mTempObjPointsDescriptors;
    cv::Mat mTempObjPointsDescriptorsRight;

    // 稀疏目标特征点容器(用于静态时作为相机路标)
    std::vector<std::vector<cv::KeyPoint>> mvOriKeys;
    std::vector<std::vector<cv::KeyPoint>> mvOriKeysUn;
    vector<cv::Mat> mOriDescriptors;
    vector<std::vector<float>> mvuOriRight;
    vector<std::vector<float>> mvOriDepth;

    // 跟踪目标相关
    vector<size_t> mvnNewConstructedObjOrders;
    vector<pair<size_t, size_t>> mvInLastFrameTrackedObjOrders; // 在上一帧的序号
    vector<size_t> mvTotalTrackedObjOrders; // 在当前帧的序号

    // 图像相关
    cv::Mat mRawImg;
    cv::Mat mObjMask;
    cv::Mat mMaskImg;    // imgMask: 目标的id是从1开始的， 而object的id是从0开始 也就是说imgMask灰度值为1的区域代表object0
    cv::Mat mMaskImgRight;
    cv::Mat mForwardOpticalImg;


    // 函数
    cv::Mat ReadKittiSegmentationImage(const string &strFolder, const int &nFrameId);
    cv::Mat ReadKittiSegmentationImage(const string &strFolder, const int &nFrameId, bool rightseg);
    cv::Mat ReadVirtualKittiSegmentationImage(const string &strFolder, const int &nFrameId);
    cv::Mat ReadVirtualKittiForwardOpticalFlow(const string &strFolder, const int &nFrameId);

    void AssignDetObjFeasToGrid(const size_t &objOrder, const size_t &objFeaNum);
    void EraseMapObjectPoint(MapObjectPoint* pMP);
    int GetCloestFeaturesInArea(const int& nDetObjOrder, const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const;


    vector<size_t> GetObjectFeaturesInArea(const int& nDetObjOrder, const size_t& nDetObjFeaNum, const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;
    cv::Mat UnprojectStereodynamic(const size_t& nDetObjOrder, const int &i, bool flag_world_frame); // flag_world_frame=1则返回landmark世界系位置，若为0则返回在camera系的位置

    void AssignFeatures(vector<DetectionObject*> &vDetectionObjects); /// 如果是双目frame函数调用该函数则为true
    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    void UndistortObjKeyPoints();
    void ComputeObjStereoMatches();



    void OfflineDetectObject(vector<DetectionObject*>& vDetectionObjects);
    void Online2DObjectTracking(cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers, vector<DetectionObject*>& vDetectionObjects);
    int FindDetectionObject(DetectionObject* cCuboidTmp);
    bool isInFrustum(MapObjectPoint *pMP, const size_t &nOrder, float viewingCosLimit);
    bool isInBBox(const size_t &nOrder, const float &u, const float &v);


    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;



    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];




    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;


    // Scale pyramid info., 金字塔相关信息
    int mnScaleLevels; // 8
    float mfScaleFactor; // 1.2
    float mfLogScaleFactor; // log1.2, 注意对数为e
    vector<float> mvScaleFactors; // 1. 1.2 1.2^2 ... 1.2^7
    vector<float> mvInvScaleFactors; // 1 1/1.2 1/(1.2)^2 ... 1/(1.2)^7
    vector<float> mvLevelSigma2; // 1 1.2^2 1.2^4 ... 1.2^14
    vector<float> mvInvLevelSigma2; // 1 1/(1.2^2) ... 1/(1.2^14)

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();


    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);



    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
