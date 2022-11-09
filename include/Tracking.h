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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>


#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "logging.h"
#include "deepsort.h"
#include "g2o_Object.h"
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>



namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class Frame;
class MapDrawer;
class System;
class MapObject;
class MapObjectPoint;
class ObjectLocalMapping;
class ObjectKeyFrame;
class Detector;
//class DeepSort;
class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);


    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);



public:
    /// variables by yuzhen
    std::vector<MapObject *> AllObjects;
    Eigen::Matrix3d mdCamProjMatrix;
    Eigen::Matrix3f mfCamProjMatrix;
    Eigen::Matrix3d mdInvCamProjMatrix;
    Eigen::Matrix3f mfInvCamProjMatrix;
    int mnFrameCounter;
    cv::Mat mTc0w, mTwc0;

    // Tracker 相关
    cv::MultiTracker* mMultiTracker;
    vector<cv::Ptr<cv::Tracker>> mvTrackers;

    // YOLO detector 相关
    Detector* mYOLODetector;
    DS::DeepSort* mDeepSort;


    /// function
    void YoloInit(const cv::FileStorage &fSettings);
    bool TrackMapObject(); // 返回成功跟踪或建立的3D object数量
    void MapObjectInit(const int &order);
    void MapObjectReInit(const int &order);
    void TrackLastFrameObjectPoint(const bool &bUseTruthObjPoseFlag);
    void CheckReplacedMapObjectPointsInLastFrame();
    void TrackObjectLocalMap();
    void UpdateObjectLocalKeyFrames(const size_t &nInCurrentFrameOrder);
    void UpdateObjectLocalPoints(const size_t &nOrder);
    int SearchObjectLocalPoints(const size_t &n);

    void InheritObjFromLastFrame();
    void DynamicStaticDiscrimination();
    void StaticPointRecoveryFromObj();
    bool InitializeCurrentObjPose(const int &i, g2o::ObjectState &Pose);
    void FineTuningUsing2dBox(const int &i, g2o::ObjectState &Pose);
    //void CharacterizeObjectTrackingQuality();
    void ReadKittiPoseInfo(const std::string &PoseFile);
    void ReadKittiObjectInfo(const std::string &objFile);
    void ReadVirtualKittiObjectInfo(const std::string &objPoseFile, const std::string &objBBoxFile);
    void ReadVirtualKittiCameraGT(const std::string &camFile);
    void ReadMynteyeObjectInfo(const std::string &objFile);

    // YOLO load class names
    std::vector<std::string> LoadNames(const std::string& path);

    bool NeedNewObjectKeyFrame(const size_t &nOrder);
    void CreateNewObjectKeyFrame(const size_t &nOrder);
    void SetObjectLocalMapper(ObjectLocalMapping *pObjectLocalMapper);
private:
    ObjectLocalMapping* mpObjectLocalMapper;




public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;


    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;
    cv::Mat mImGrayRight;
    //cv::Mat mImColor;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();


    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;



    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;


    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    std::vector<Frame> mvTemporalLocalWindow;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
