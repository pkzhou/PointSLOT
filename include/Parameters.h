//
// Created by liuyuzhen on 2020/5/23.
//
#pragma once

#ifndef ORB_SLAM2_PARAMETERS_H
#define ORB_SLAM2_PARAMETERS_H
#include <string>
#include<mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <unordered_set>
using namespace std;

//#include "Tracking.h"

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
namespace ORB_SLAM2{

    // 整体相关
    extern bool EbSLOTFlag; // 0代表无动态对象，1代表有动态对象
    extern std::string EstrDatasetFolder; // 数据集路径
    extern int EnDataSetNameNum;// 数据集名称代号, 0代表kitti, 1代表virtual kitti, 2代表小觅相机
    extern size_t EnImgTotalNum; // 图像总数
    extern int EnStartFrameId;
    extern bool EbSetWorldFrameOnGroundFlag;// 人为设定世界系
    extern double EdInitX;
    extern double EdInitY;
    extern double EdInitZ;
    extern double EdInit_qx;
    extern double EdInit_qy;
    extern double EdInit_qz;
    extern double EdInit_qw;
    extern double EdDrawFrameWaiKeyTime;
    extern float EfStereoThDepth;
    extern size_t EnSlidingWindowSize;
    // 相机参数
    extern Eigen::Matrix3d EdCamProjMatrix;
    extern float Efbf;
    extern cv::Mat EK;
    extern double EdT; // 图像周期

    // 离线数据
    extern vector<cv::Mat> EvLeftCamGTPose, EvRightCamGTPose; // 只有virtual kitti 才读取了验证精度
    extern std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> EvOfflineAllObjectDetections; // 目标离线检测结果
    extern vector<cv::Mat> OfflineFramePoses;

    // 小觅相关
    extern bool EbUseMynteyeCameraFlag;//使用小觅相机,如果是kitti则用了图像分割mask比较准确, 如果是小觅其包围框大小来生成maskimg可能需要调整
    extern std::string EstrStereoRectifyFile; // 双目校正文件,只有小觅才需要,kitti已经校正好了

    // 目标相关
    // 0 就是纯SLAM模式，
    // 1 是动态SLAM的模式（检测语义动态目标， 去除做动态SLAM）
    // 2 是目标跟踪模式（人为指定目标进行跟踪）
    // 3 为自动驾驶模式： 需要估计出离自己比较近的动态目标的pose，对比较近的动态目标进行tracking
    extern int EnSLOTMode;
    extern int EnDynaSLAMMode;
    extern bool EbYoloActive;
    extern int EnOnlineDetectionMode;
    extern int EnSelectTrackedObjId;
    extern int  EnNarrowBBoxPixelValue; //包围框向里收缩的像素点,来产生更紧密的maskimg,提取特征点,只针对使用了2D检测器的情况
    extern double  EdMaxObjMissingDt; // 目标丢失最大允许间隔时间
    extern bool EbManualSetPointMaxDistance;
    extern float EfInObjFramePointMaxDistance; // 允许目标点离目标中心的最大距离
    extern int EnInitDetObjORBFeaturesNum;
    extern int EnInitMapObjectPointsNum;
    extern int EnMinTrackedMOPsNUM;
    extern size_t EnTrackObjectMinFeatureNum;
    extern bool EbObjStateOptimizationFlag; // 采用目标跟踪优化
    extern bool EbUseOfflineAllObjectDetectionPosesFlag; // 每帧目标优化初值采用离线检测结果
    extern bool EbUseUniformObjScaleFlag;// 优化采用预设统一目标尺度
    extern Eigen::Vector3d EeigUniformObjScale;
    extern bool EbSetInitPositionByPoints;
    extern Eigen::Vector3d EeigInitPosition;
    extern Eigen::Vector3d EeigInitRotation;
    extern int EnObjectCenter;
    extern double EdVehicleFrontAndRearDistance; // 针对车辆,前后车轮距离
    extern unordered_set<int> EobjTrackEndHash;

    extern float EfConfThres;
    extern float EfIouThres;
    extern std::vector<std::string> EvClassNames;


    // 显示相关
    extern bool EbStartViewerWith2DTracking;
    extern bool EbViewCurrentObject;
    extern bool EbUseSegementation;


    // 目标BA 的各项权重与ThHuber值
    extern int EnObjBBoxBAWeight; // 用在目标跟踪优化
    extern double EdSmoothTermBAWeight; // 用在滑窗优化
    extern double EdObjectMotionModelBAWeight;
    extern double EdAngularVelThanLinearVelBAWeightTimes;
    extern double EdBBoxBAWeight;
    extern double EdSmoothTermThHuber;
    extern double EdObjectMotionModelThHuber;
    extern double EdBBoxThHuber;
    extern double EdMonoThHuber;
    extern double EdStereoThHuber;


    // Debug 用
    extern bool EbDrawStaticFeatureMatches;

    // 目前版本可能未怎么用到的Flag
    extern std::vector<Eigen::MatrixXd> EvOfflineAllObjectDetectionsNotUsed;
    extern bool EbMonoUseStereoImageInitFlag; // 单目配置采用双目图像初始化
    extern int  EnGivenTrackingObjectID; // 指定目标跟踪ID
    extern bool EbUseDynamicORBFeatureFlag;
    extern bool EbLocalMappingWithObjectFlag;
}


#endif //ORB_SLAM2_PARAMETERS_H
