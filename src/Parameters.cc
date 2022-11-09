//
// Created by liuyuzhen on 2020/5/23.
//
#include "Parameters.h"

namespace  ORB_SLAM2
{
    // 整体相关
    bool EbSLOTFlag = true;
    std::string EstrDatasetFolder;
    int EnDataSetNameNum;
    size_t EnImgTotalNum;
    int EnStartFrameId = 0;//97
    bool EbSetWorldFrameOnGroundFlag = false;
    double EdInitX = 0;
    double EdInitY = 0;
    double EdInitZ = 1.1;
    double EdInit_qx = -0.7071;
    double EdInit_qy = 0;
    double EdInit_qz = 0;
    double EdInit_qw = 0.7071;
    double EdDrawFrameWaiKeyTime = 50;
    float EfStereoThDepth;
    size_t EnSlidingWindowSize = 5;
    Eigen::Matrix3d EdCamProjMatrix;
    float Efbf;
    cv::Mat EK;
    double EdT;
    // 离线数据
    vector<cv::Mat> EvLeftCamGTPose, EvRightCamGTPose;
    std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> EvOfflineAllObjectDetections;
    vector<cv::Mat> OfflineFramePoses;


    // 小觅相关
    bool EbUseMynteyeCameraFlag= false;
    std::string EstrStereoRectifyFile = "/home/liuyuzhen/SLOT/cubeslam_ws/src/cube_slam/orb_object_slam/Examples/Monocular/by_jiacheng/mynteye_stereo.yaml";

    // 目标相关
    int EnSLOTMode = 1;
    int EnDynaSLAMMode = 0;
    bool EbYoloActive = true;
    int EnOnlineDetectionMode = 0;
    int EnSelectTrackedObjId = 65;
    int EnNarrowBBoxPixelValue=10;
    double EdMaxObjMissingDt = 0.5;//0.5 秒钟消失不见
    bool EbManualSetPointMaxDistance;
    float EfInObjFramePointMaxDistance; // 0.4
    int EnInitDetObjORBFeaturesNum = 40; // 70
    int EnInitMapObjectPointsNum = 17;
    int EnMinTrackedMOPsNUM = 15;
    size_t EnTrackObjectMinFeatureNum = 30;
    bool EbObjStateOptimizationFlag = true;
    bool EbUseOfflineAllObjectDetectionPosesFlag = 0;
    bool EbUseUniformObjScaleFlag = 0;
    Eigen::Vector3d EeigUniformObjScale;
    bool EbSetInitPositionByPoints = true;
    Eigen::Vector3d EeigInitPosition;
    Eigen::Vector3d EeigInitRotation(0, 0, 0);
    int EnObjectCenter; // 0 代表在几何中心, 1 代表在底面中心(只有virtual kitti提供的目标坐标系是在底部中心)
    double EdVehicleFrontAndRearDistance = 0.15;//前轮到后轮
    unordered_set<int> EobjTrackEndHash{-1};

    // YOLO
    float EfConfThres = 0.4;
    float EfIouThres = 0.5;
    std::vector<std::string> EvClassNames;



    // 显示相关
    bool EbStartViewerWith2DTracking = false;
    bool EbViewCurrentObject = true;
    bool EbUseSegementation = true;

    // 目标BA 的各项权重与ThHuber值
    int EnObjBBoxBAWeight = 2;
    double EdSmoothTermBAWeight = 2;
    double EdObjectMotionModelBAWeight = 1;
    double EdAngularVelThanLinearVelBAWeightTimes = 2;
    double EdBBoxBAWeight = 2;
    double EdSmoothTermThHuber = sqrt(10);
    double EdObjectMotionModelThHuber = sqrt(10);
    double EdBBoxThHuber = sqrt(900);
    double EdMonoThHuber = sqrt(5.991);
    double EdStereoThHuber = sqrt(7.815);

    // Debug 用
    bool EbDrawStaticFeatureMatches = 0;

    // 目前版本可能未怎么用到的Flag
    std::vector<Eigen::MatrixXd> EvOfflineAllObjectDetectionsNotUsed;
    bool EbMonoUseStereoImageInitFlag =1;
    int EnGivenTrackingObjectID=1;//object序号
    bool EbUseDynamicORBFeatureFlag=1;
    bool EbLocalMappingWithObjectFlag = 0;
}

