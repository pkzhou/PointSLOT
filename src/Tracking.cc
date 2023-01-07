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
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Tracking.h"
#include "Parameters.h"
#include"ORBmatcher.h"
#include"Converter.h"
#include"Optimizer.h"
#include"PnPsolver.h"
#include "g2o_Object.h"
#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include "MapDrawer.h"
#include "System.h"
#include "KeyFrame.h"
#include "MapObject.h"
#include "MapObjectPoint.h"
#include "MapPoint.h"
#include "DetectionObject.h"
#include "ObjectKeyFrame.h"
#include "ObjectLocalMapping.h"
#include "YOLOdetector.h"
#include "cxxopts.hpp"
#include<iostream>
#include<mutex>
#include <unistd.h>
#include <string>
#include <unordered_set>
#include <numeric>

using namespace std;
static DS::Logger mgLogger;

namespace ORB_SLAM2 {
    typedef Eigen::Matrix<double, 9, 1> Vector9d;

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{



    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    // 决定了算法模式，
    EnSLOTMode = fSettings["SLOT.MODE"];
    if(EnSLOTMode == 1)
        EnDynaSLAMMode = fSettings["DynaSLAM.MODE"];
    // 是否在线检测
    EnOnlineDetectionMode = fSettings["Object.EnOnlineDetectionMode"];
    // 是否手动统一设置目标点在目标系范围
    int tmp1 = fSettings["Object.EbManualSetPointMaxDistance"];
    EbManualSetPointMaxDistance = tmp1 > 0 ? 1:0;
    if(EbManualSetPointMaxDistance)
        EfInObjFramePointMaxDistance = fSettings["Object.EfInObjFramePointMaxDistance"];


    // 算法模式
    switch(EnSLOTMode)
    {
        case 0: // SLAM 模式， 什么也不需要干
        {
            break;
        }
        case 1: // 动态SLAM模式
        {
            if(EnDynaSLAMMode == 0)
            {
                // 语义动态slam模式
                if(EnOnlineDetectionMode)// 这里会分成在线还是离线的模式
                {
                    YoloInit(fSettings);// YOLOdetector初始化
                }
                else{

                    switch(EnDataSetNameNum)
                    {
                        // std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> Kitti_AllTrackingObjectInformation
                        case 0:// kitti tracking
                        {
                            std::string Kittiobjecttrackingfile = EstrDatasetFolder + "/ObjectTracking.txt";
                            ReadKittiObjectInfo(Kittiobjecttrackingfile);
                            break;
                        }
                        case 1:// virtual kitti
                        {
                            /// 1. 读取object离线信息
                            std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                            std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";
                            ReadVirtualKittiObjectInfo(virtualkittiobjectposefile, virtualkittibboxfile);
                            /// 2. 读取camera groundtruth
                            /// 读取结果在 vector<cv::Mat> CameraGTpose_left, CameraGTpose_right
                            std::string virtualKittiCameraGroundthPoseFile = EstrDatasetFolder + "/extrinsic.txt";
                            ReadVirtualKittiCameraGT(virtualKittiCameraGroundthPoseFile);
                            break;
                        }
                        default:
                            assert(0);
                    }

                }// TODO 直接读取离线的目标数据
            }
            else
            {
                // 目标跟踪动态slam模式,只是在线
                int tmp = fSettings["Yolo.active"];
                EbYoloActive = tmp > 0?true:false;
                if(EbYoloActive == true)
                {
                    YoloInit(fSettings);
                }
                mMultiTracker = new cv::MultiTracker();
                for(size_t i=0; i<1; i++)  // 目前设置就跟踪一个目标
                {
                    cv::Ptr<cv::Tracker> Tracker = cv::TrackerCSRT::create();
                    mvTrackers.push_back(Tracker);
                }
            }
            break;
        }

        case 2: // 目标跟踪模式, 单目标跟踪， 可以设置多个目标跟踪,
        {
            // TODO 应该也考虑kitti 或 是virtual_kitti的数据集 也有目标跟踪 模式
            EnObjectCenter = fSettings["Viewer.ObjectCenter"]; // 目标系定在哪
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"]; // 初始化目标的点数
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"]; // 是否以三角化点来初始化目标的位置
            EbSetInitPositionByPoints = temp2 > 0? 1:0;
            // 目标先验尺度，仅仅是为了画图用
            EeigUniformObjScale(0) = fSettings["Object.Width.xc"];
            EeigUniformObjScale(1) = fSettings["Object.Height.yc"];
            EeigUniformObjScale(2) = fSettings["Object.Length.zc"];
            // 目标初始的位置，相对于相机系， 其实也可以不需要目标的先验pose
            EeigInitPosition(0) = fSettings["Object.position.xc"];
            EeigInitPosition(1) = fSettings["Object.position.yc"];
            EeigInitPosition(2) = fSettings["Object.position.zc"];
            EeigInitRotation(1) = fSettings["Object.yaw.y"];
            if(EnOnlineDetectionMode)
            {
                mMultiTracker = new cv::MultiTracker();
                for(size_t i=0; i<1; i++)  // 目前设置就跟踪一个目标
                {
                    cv::Ptr<cv::Tracker> Tracker = cv::TrackerCSRT::create();
                    mvTrackers.push_back(Tracker);
                }
            }
            else
            {
                std::string objectFile = EstrDatasetFolder + "/object.txt"; // 读取该目标的离线数据
                ReadMynteyeObjectInfo(objectFile);
            }
            break;
        }

        case 3: // 自动驾驶模式
        {
            EnObjectCenter = fSettings["Viewer.ObjectCenter"]; // 目标系定在哪
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"]; // 初始化目标的点数
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"]; // 是否以三角化点来初始化目标的位置
            EbSetInitPositionByPoints = temp2 > 0? 1:0;
            switch(EnOnlineDetectionMode)
            {
                case 0: // 离线模式
                {
                    switch(EnDataSetNameNum)
                    {
                        // std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> Kitti_AllTrackingObjectInformation
                        case 0:// kitti tracking
                        {
                            std::string Kittiobjecttrackingfile = EstrDatasetFolder + "/ObjectTracking.txt";
                            ReadKittiObjectInfo(Kittiobjecttrackingfile);
                            break;
                        }
                        case 1:// virtual kitti
                        {
                            /// 1. 读取object离线信息
                            std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                            std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";
                            ReadVirtualKittiObjectInfo(virtualkittiobjectposefile, virtualkittibboxfile);
                            /// 2. 读取camera groundtruth
                            /// 读取结果在 vector<cv::Mat> CameraGTpose_left, CameraGTpose_right
                            std::string virtualKittiCameraGroundthPoseFile = EstrDatasetFolder + "/extrinsic.txt";
                            ReadVirtualKittiCameraGT(virtualKittiCameraGroundthPoseFile);
                            break;
                        }
                        default:
                            assert(0);
                    }
                    break;
                }

                case 1:
                {
                    // 因为我目前只有2D检测， 因此也需要设置统一的尺度画图用
                    EeigUniformObjScale(0) = fSettings["Object.Width.xc"];
                    EeigUniformObjScale(1) = fSettings["Object.Height.yc"];
                    EeigUniformObjScale(2) = fSettings["Object.Length.zc"];
                    //EbSetInitPositionByPoints = true;
                    // 初始化YOLO, deepSort算法
                    YoloInit(fSettings); // 1. YOLOdetector初始化
                    std::string sort_engine_path_ = fSettings["DeepSort.weightsPath"];// 2. DeepSort 算法初始化
                    mDeepSort = new DS::DeepSort(sort_engine_path_, 128, 256, 0, &mgLogger);

                    //test
                    if (EnDataSetNameNum == 0){
                        std::string Kittiobjecttrackingfile = EstrDatasetFolder + "/ObjectTracking.txt";
                        ReadKittiObjectInfo(Kittiobjecttrackingfile);
                    }

                    break; // 在线模式
                }

                default:
                    assert(0);
            }
            break;
        }

        case 4: // 终极算法测试模式，在此模式下会对选定的目标进行稳定跟踪（包括tracking和mapping），而其他目标会被认作不稳定区域而被去除
        {
            EnObjectCenter = fSettings["Viewer.ObjectCenter"]; // 目标系定在哪
            EnSelectTrackedObjId = fSettings["Object.EnSelectTrackedObjId"];
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"]; // 初始化目标的点数
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"]; // 是否以三角化点来初始化目标的位置
            EbSetInitPositionByPoints = temp2 > 0? 1:0;
            switch(EnDataSetNameNum)
            {
                case 0:// kitti tracking
                {
                    std::string Kittiobjecttrackingfile = EstrDatasetFolder + "/ObjectTracking.txt";
                    string Kitticameraposefile = EstrDatasetFolder + "/0011.txt";
                    ReadKittiPoseInfo(Kitticameraposefile);
                    ReadKittiObjectInfo(Kittiobjecttrackingfile);
                    break;
                }
                case 1:// virtual kitti
                {
                    /// 1. 读取object离线信息
                    std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                    std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";
                    ReadVirtualKittiObjectInfo(virtualkittiobjectposefile, virtualkittibboxfile);
                    /// 2. 读取camera groundtruth
                    /// 读取结果在 vector<cv::Mat> CameraGTpose_left, CameraGTpose_right
                    std::string virtualKittiCameraGroundthPoseFile = EstrDatasetFolder + "/extrinsic.txt";
                    ReadVirtualKittiCameraGT(virtualKittiCameraGroundthPoseFile);
                    break;
                }
                default:
                    assert(0);
            }

            break;
        }


        default:
            assert(0);
    }

    cout<<"offline object information and camera parameters are read."<<endl;


    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    if(ORB_SLAM2::EnDataSetNameNum == 3)
    {
        fx = EdCamProjMatrix(0, 0);
        fy = EdCamProjMatrix(1, 1);
        cx = EdCamProjMatrix(0, 2);
        cy = EdCamProjMatrix(1, 2);
    }


    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);
    K.copyTo(EK);

    //TODO 把相机参数拿出来供后面优化用
    mdCamProjMatrix.setIdentity();
    mdCamProjMatrix(0, 0) = fx;
    mdCamProjMatrix(0, 2) = cx;
    mdCamProjMatrix(1, 1) = fy;
    mdCamProjMatrix(1, 2) = cy;
    EdCamProjMatrix = mdCamProjMatrix;
    mfCamProjMatrix = mdCamProjMatrix.cast<float>();
    mdInvCamProjMatrix = mdCamProjMatrix.inverse();
    mfInvCamProjMatrix = mfCamProjMatrix.inverse();
    mpMap->mdCamProjMatrix = mdCamProjMatrix;
    mpMap->mfCamProjMatrix = mfCamProjMatrix;
    mpMap->mfInvCamProjMatrix = mfInvCamProjMatrix;

    //TODO 定义初始第一帧相对于世界坐标系, groundToInit, InitToGround
    mTc0w = cv::Mat::eye(4, 4, CV_32F);
    Eigen::Quaternionf pose_quat(EdInit_qw, EdInit_qx, EdInit_qy, EdInit_qz);
    Eigen::Matrix3f rot = pose_quat.toRotationMatrix(); // 	The quaternion is required to be normalized
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            mTc0w.at<float>(row, col) = rot(row, col);
    mTc0w.at<float>(0, 3) = EdInitX;
    mTc0w.at<float>(1, 3) = EdInitY;
    mTc0w.at<float>(2, 3) = EdInitZ;
    cv::Mat R = mTc0w.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = mTc0w.rowRange(0, 3).col(3);
    cv::Mat Rinv = R.t();
    cv::Mat Ow = -Rinv * t;
    mTwc0 = cv::Mat::eye(4, 4, CV_32F);
    Rinv.copyTo(mTwc0.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(mTwc0.rowRange(0, 3).col(3));


    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];
    Efbf = mbf;

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    mnFrameCounter = 0;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO || EbMonoUseStereoImageInitFlag==1)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD||1)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        EfStereoThDepth = mThDepth;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mvTemporalLocalWindow.reserve(EnSlidingWindowSize);


}



// YOLO init function
void Tracking::YoloInit(const cv::FileStorage &fSettings)
{
    torch::DeviceType device_type; // 处理器类型
    int isGPU = fSettings["Yolo.isGPU"];
    device_type = isGPU > 0 ? torch::kCUDA:torch::kCPU;
    EvClassNames = LoadNames("/home/zpk/SLOT/ORB_SLAM2/weights/coco.names");
    if(EvClassNames.empty())
    {
        cout<<"There is no class file of YOLO!"<<endl;
        assert(0);
    }
    EfConfThres = fSettings["Yolo.confThres"]; // 阈值
    EfIouThres = fSettings["Yolo.iouThres"];
    std::string weights = fSettings["Yolo.weightsPath"];
    //weights = "/home/liuyuzhen/SLOT/orb-slam/code_tempbyme/DetectingAndTracking/libtorch-yolov5-master-temp/weights/yolov5s.torchscript";
    mYOLODetector =  new Detector(weights, device_type);
    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    auto temp_img = cv::Mat::zeros(height, width, CV_32FC3);
    mYOLODetector->Run(temp_img, 1.0f, 1.0f); // 前两张非常慢, 可以改成线程, 和loadORB词包一起
    mYOLODetector->Run(temp_img, 1.0f, 1.0f);
}

void Tracking::ReadKittiPoseInfo(const std::string &PoseFile){
    vector<cv::Mat> Pose_temp;
    ifstream filetxt(PoseFile.c_str());
    string line;
    while(getline(filetxt,line)){
        if (!line.empty()){
            stringstream ss(line);
            cv::Mat pose = cv::Mat::eye(4,4,CV_32F);
            ss>> pose.at<float>(0,0);
            ss>> pose.at<float>(0,1);
            ss>> pose.at<float>(0,2);
            ss>> pose.at<float>(0,3);
            ss>> pose.at<float>(1,0);
            ss>> pose.at<float>(1,1);
            ss>> pose.at<float>(1,2);
            ss>> pose.at<float>(1,3);
            ss>> pose.at<float>(2,0);
            ss>> pose.at<float>(2,1);
            ss>> pose.at<float>(2,2);
            ss>> pose.at<float>(2,3);
            pose.at<float>(3,0) = 0.;
            pose.at<float>(3,1) = 0.;
            pose.at<float>(3,2) = 0.;
            pose.at<float>(3,3) = 1.;
            Pose_temp.push_back(pose.clone());
        }
        else
            assert(0);
    }
    OfflineFramePoses = Pose_temp;
}

/// kitti: 输入文件格式: frame_id object_id 类型 truncated occuluded alpha bbox dimensions location rotation_y
/// 输出格式：std::vector<std::vector<Eigen::Matrix<double, 1, 20>>> Kitti_AllTrackingObjectInformation
/// 1-20： frame_id track_id truncated occuluded alpha bbox dimensions(height, width, length) location(x,y,z) rotation score type_id is_moving x1
/// location是指在目标在相机坐标系下, type_id = 1才有效
// 需要注意occuluded 参数和virtual kitti不同
void Tracking::ReadKittiObjectInfo(const std::string &inputfile)
{
    /// 1. 初始化： 相关变量定义
    std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> all_object_temp;
    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp;

    std::string pred_frame_obj_txts = inputfile;
    /// 1.3 读取源跟踪结果文件， 转成流： filetxt
    std::ifstream filetxt(pred_frame_obj_txts.c_str());
    /// 1.4 源文件的每一行
    std::string line;
    /// 1.5 上一次的frame_id
    int frame_waibu_id = -1;
    /// 1.6 读取每一行的filetxt到line， 并进行处理
    while(getline(filetxt, line))
    {
        if(!line.empty())
        {
            /// 1.6.1 将line给stringstream: ss
            std::stringstream ss(line);
            /// 1.6.2 将ss赋给每个变量
            /// (1) 目标类型
            std::string type;
            /// (2) 帧的id， 目标的id
            int frame_id, track_id;
            /// (3) truncation的程度？？？ 是否遮挡， 目标的观测角
            double truncated, occuluded, alpha;
            /// (4) 目标的2D框
            Eigen::Vector4d bbox;
            /// (5) 目标的尺度
            Eigen::Vector3d dimensions;
            /// (6) 目标的3D位置;
            Eigen::Vector3d locations;
            /// (7) 目标的yaw角
            double rotation_y;
            /// (8) 置信度
            double score;
            /// (9) 类型：0是人，电车， dontcare等等; 1是需要跟踪的目标车辆
            double type_id;

            ss>>frame_id;
            ss>>track_id;
            ss>>type;
            if(type == "Pedestrian" || type == "Person_sitting" || type == "Cyclist" ||
            type == "Tram" || type == "Misc" || type == "DontCare")
            {
                type_id = 0;
            }
            else{
                type_id = 1;
            }


            /// 只保留'Car', 'Van', 'Truck'类型
            /// 去除'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
            //if(type == "Pedestrian" or type == "Person_sitting" or type == "Cyclist" or type == "Tram" or type == "Misc" or type == "DontCare")
            //   continue;

            // TODO  如果该帧没有检测到object该怎么办？？

            // 如果第一行不是从第0帧开始
            if(frame_id != 0 && frame_waibu_id == -1)
            {
                while(int(all_object_temp.size()) != frame_id)
                {
                    Eigen::Matrix<double, 1, 24> oneobject_oneframe;
                    oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
                    oneobject_oneframe [1] = -1;
                    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp3;
                    objects_inoneframe_temp3.push_back(oneobject_oneframe);
                    all_object_temp.push_back(objects_inoneframe_temp3);
                }
            }

            // TODO 写入操作： 写入条件， 开始读新的一张图像时，把上一帧图像的object信息写入
            // frame_waibu_id是上一行的frame_id， 二者不相同说明是这是一帧新图像(但是第一张图像也满足此条件，但第一张图像不存在上一张图像，因此跳过frame_waibu_id!=-1)
            if(frame_id != frame_waibu_id && frame_waibu_id != -1)
            {
                // 把上一次的objects_inoneframe_temp放入
                all_object_temp.push_back(objects_inoneframe_temp);
                objects_inoneframe_temp.clear();

                // 如果中间有帧没有检测到object，则填入为0的eigen
                while(int(all_object_temp.size())!=frame_id)
                {
                    Eigen::Matrix<double, 1, 24> oneobject_oneframe;
                    oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
                    oneobject_oneframe [1] = -1;
                    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp3;
                    objects_inoneframe_temp3.push_back(oneobject_oneframe);
                    all_object_temp.push_back(objects_inoneframe_temp3);
                }
            }

            ss>>truncated;
            ss>>occuluded;
            ss>>alpha;
            ss>>bbox[0];
            ss>>bbox[1];
            ss>>bbox[2];
            ss>>bbox[3];
            /// 注意： 这里重新修改bbox[2] 与 bbox[3]为width和height
            bbox[2] = bbox[2]-bbox[0];
            bbox[3] = bbox[3]-bbox[1];
            ss>>dimensions[1]; // 注意kitti是height width length, 我存的数据结构是length  height width??
            ss>>dimensions[2];
            ss>>dimensions[0];
            ss>>locations[0];
            ss>>locations[1];
            ss>>locations[2];
            ss>>rotation_y;
            score = 1;

            Eigen::Matrix<double, 1, 24> oneobject_oneframe;
            oneobject_oneframe [0] = frame_id;
            oneobject_oneframe [1] = track_id;
            oneobject_oneframe [2] = truncated;
            oneobject_oneframe [3] = occuluded; // 需要注意的是,这个含义与virtual kitti不同: integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
            oneobject_oneframe [4] = alpha;
            oneobject_oneframe [5] = bbox[0];
            oneobject_oneframe [6] = bbox[1];
            oneobject_oneframe [7] = bbox[2];
            oneobject_oneframe [8] = bbox[3];
            oneobject_oneframe [9] = dimensions[0];
            oneobject_oneframe [10] = dimensions[1];
            oneobject_oneframe [11] = dimensions[2];
            if(ORB_SLAM2::EbManualSetPointMaxDistance == false)
            {
                Eigen::Vector3f scale(dimensions[0]/2, dimensions[1]/2, dimensions[2]/2);
                ORB_SLAM2::EfInObjFramePointMaxDistance = scale.norm();
            }
            oneobject_oneframe [12] = locations[0];
            oneobject_oneframe [13] = locations[1];
            oneobject_oneframe [14] = locations[2];
            oneobject_oneframe [15] = rotation_y;
            oneobject_oneframe [16] = score;
            oneobject_oneframe [17] = type_id;

            int is_moving = 1;
            oneobject_oneframe [18] = is_moving;

            int extend = 0;
            oneobject_oneframe [19] = extend;

            // 右目相机的观测：bbox
            oneobject_oneframe [20] = 0;
            oneobject_oneframe [21] = 0;
            oneobject_oneframe [22] = 0;
            oneobject_oneframe [23] = 0;



            objects_inoneframe_temp.push_back(oneobject_oneframe);
            frame_waibu_id = frame_id;
        }
    }

    // 如果此时的object size不等于frame_waibu_id， 那说明有问题， 举例: id为12,那么目前的object从0-11都有
    if(int(all_object_temp.size())!=frame_waibu_id) // 此时的frame_waibu_id 与 frame_id相同
    {
        cout<<"Error reading offline object pose information！！！"<<endl;
        exit(0);
    }


    // 将最后一帧的object数据放入
    all_object_temp.push_back(objects_inoneframe_temp);

    // 如果最后一帧不是实际上的最后一帧, 就将中间缺失帧的object信息填为0的eigen
    if(frame_waibu_id != int(EnImgTotalNum-1))
    {
        while(all_object_temp.size() != EnImgTotalNum)
        {
            /// 为0的eigen
            Eigen::Matrix<double, 1, 24> oneobject_oneframe;
            oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
            oneobject_oneframe [1] = -1;
            std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp2;
            objects_inoneframe_temp2.push_back(oneobject_oneframe);
            all_object_temp.push_back(objects_inoneframe_temp2);
        }
    }


    EvOfflineAllObjectDetections = all_object_temp;
}

/// virtual kitti
/// 输入文件格式1pose：frame_id cameraId trackID alpha width height length
/// world_space_x world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z
/// camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
/// 输入文件格式2bbox: frame_id cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio ismoving
/// 输出文件格式(和KITTI相同)：std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> Kitti_AllTrackingObjectInformation
///  frame_id track_id truncated occuluded alpha bbox[0]-box[3] dimensions[0]-dimensions[2]
///  location[0]-location[2] rotation_y score type_id is_moving extend bbox_right[0]-bbox[3]
///// location还是指目标在相机坐标系下
/// type_id 1是车
void Tracking::ReadVirtualKittiObjectInfo(const std::string &objectposefile, const std::string &objectbboxfile)
{
    std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> all_object_temp;
    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp;

    // pose txt
    std::ifstream posetxt(objectposefile.c_str());
    // bbox txt
    std::ifstream bboxtxt(objectbboxfile.c_str());

    std::string lineforpose, lineforbbox;
    int frame_waibu_id = -1;
    int camera_right = 0;
    size_t frame_id;
    int  cameraID, trackID;
    // pose 信息
    double alpha, width, height, length, wx, wy,wz, rwy, rwx, rwz, cx, cy, cz, rcy, rcx, rcz;
    // bbox信息
    double left, right, top, bottom, number_pixels, truncation_ratio, occupancy_ratio;
    char is_moving[16];

    // 分别读取, 第一行读了不要
    getline(posetxt, lineforpose);
    getline(bboxtxt, lineforbbox);
    while(getline(posetxt, lineforpose) && getline(bboxtxt, lineforbbox))
    {
        // 分别读取一行
        if((!lineforpose.empty())&&(!lineforbbox.empty()))
        {
            std::stringstream sspose(lineforpose);
            std::stringstream ssbbox(lineforbbox);

            sspose>>frame_id;
            ssbbox>>frame_id;
            sspose>>cameraID;
            ssbbox>>cameraID;
            Eigen::Matrix<double, 1, 24> oneobject_oneframe;


            // 特殊情况1: 起始帧不是从第0帧开始
            if(frame_id != 0 && frame_waibu_id == -1)
            {
                // 填入为0的eigen
                while(all_object_temp.size() != frame_id)
                {
                    Eigen::Matrix<double, 1, 24> oneobject_oneframe;
                    oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
                    oneobject_oneframe [1] = -1;
                    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp3;
                    objects_inoneframe_temp3.push_back(oneobject_oneframe);
                    all_object_temp.push_back(objects_inoneframe_temp3);
                }
            }

            /// 写入操作：
            // 写入条件： 开始读新的一张图像时， 把上一帧图像的object信息写入
            if(int(frame_id) != frame_waibu_id && frame_waibu_id != -1)
            {
                // 把上一次的objects_inoneframe_temp放入
                all_object_temp.push_back(objects_inoneframe_temp);
                camera_right=0;
                objects_inoneframe_temp.clear();

                // 特殊情况2： 如果中间有帧没有检测到object， 则填入为0的eigen
                while(all_object_temp.size()!=frame_id)
                {
                    Eigen::Matrix<double, 1, 24> oneobject_oneframe;
                    oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
                    oneobject_oneframe [1] = -1;
                    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp3;
                    objects_inoneframe_temp3.push_back(oneobject_oneframe);
                    all_object_temp.push_back(objects_inoneframe_temp);
                }
            }

            if(cameraID==0)
            {
                /// 正常赋值操作:
                // 共用
                sspose>>trackID;
                ssbbox>>trackID;
                // pose
                sspose>>alpha;
                sspose>>width;
                sspose>>height;
                sspose>>length;
                sspose>>wx;
                sspose>>wy;
                sspose>>wz;
                sspose>>rwy;
                sspose>>rwx;
                sspose>>rwz;
                sspose>>cx;
                sspose>>cy;
                sspose>>cz;
                sspose>>rcy;
                sspose>>rcx;
                sspose>>rcz;
                // bbox
                ssbbox>>left;
                ssbbox>>right;
                ssbbox>>top;
                ssbbox>>bottom;
                ssbbox>>number_pixels;
                ssbbox>>truncation_ratio;
                ssbbox>>occupancy_ratio;
                ssbbox>>is_moving;

                oneobject_oneframe [0] = frame_id;
                oneobject_oneframe [1] = trackID;
                oneobject_oneframe [2] = truncation_ratio; // [0..1] (0: no truncation, 1: entirely truncated)
                oneobject_oneframe [3] = occupancy_ratio; // fraction of non-occuluded pixels [0..1] (0: fully occluded, 1: fully visible, independent of truncation)
                oneobject_oneframe [4] = alpha;
                oneobject_oneframe [5] = left;
                oneobject_oneframe [6] = top;
                oneobject_oneframe [7] = right-left;
                oneobject_oneframe [8] = bottom-top;
                oneobject_oneframe [9] = length; // x 是length
                oneobject_oneframe [10] = height; // y 是height
                oneobject_oneframe [11] = width; // z 是width

                if(ORB_SLAM2::EbManualSetPointMaxDistance == false)
                {
                    Eigen::Vector3f scale(length/2, height/2, width/2);
                    ORB_SLAM2::EfInObjFramePointMaxDistance = scale.norm();
                }

                oneobject_oneframe [12] = cx;
                oneobject_oneframe [13] = cy;
                oneobject_oneframe [14] = cz;
                oneobject_oneframe [15] = rcy; // rcx, rcz为什么没有
                oneobject_oneframe [16] = rcx; // 1 score
                oneobject_oneframe [17] = 1; //  1 type_id， 直接就是车

                //if(is_moving=="True")
                if(strcmp(is_moving,"True") == 0)     //相等则返回0
                    oneobject_oneframe [18] = 1;
                else
                    oneobject_oneframe [18] = 0;

                //int extend = 0;
                oneobject_oneframe [19] = rcz;
                objects_inoneframe_temp.push_back(oneobject_oneframe);

            }
            else{
                // bbox
                ssbbox>>trackID;
                ssbbox>>left;
                ssbbox>>right;
                ssbbox>>top;
                ssbbox>>bottom;
                ssbbox>>number_pixels;
                ssbbox>>truncation_ratio;
                ssbbox>>occupancy_ratio;
                ssbbox>>is_moving;
                if(objects_inoneframe_temp.size()==0)
                {
                    cout<<"reading error"<<endl;
                    exit(0);
                }
                else{

                    objects_inoneframe_temp[camera_right][20] = left;
                    objects_inoneframe_temp[camera_right][21] = top;
                    objects_inoneframe_temp[camera_right][22] = right - left;
                    objects_inoneframe_temp[camera_right][23] = bottom - top;

                }

                camera_right++;
            }
            frame_waibu_id = frame_id;
        }
    }


    if(int(all_object_temp.size())!=frame_waibu_id)
    {
        cout<<"Error reading offline object pose information！！！"<<endl;
        exit(0);
    }

    // 将最后一帧的object数据放入
    all_object_temp.push_back(objects_inoneframe_temp);

    // 如果最后一帧不是实际上的最后一帧, 就将中间缺失帧的object信息填为0的eigen
    if(frame_waibu_id != int(EnImgTotalNum-1))
    {
        while(all_object_temp.size() != EnImgTotalNum)
        {
            /// 为0的eigen
            Eigen::Matrix<double, 1, 24> oneobject_oneframe;
            oneobject_oneframe = Eigen::Matrix<double, 1, 24>::Zero();
            oneobject_oneframe [1] = -1;
            std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp2;
            objects_inoneframe_temp2.push_back(oneobject_oneframe);
            all_object_temp.push_back(objects_inoneframe_temp2);
        }
    }


    EvOfflineAllObjectDetections = all_object_temp;

}



/// 读取真实的camera pose
/// 输入文件格式为： frame_id cameraID r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3 0 0 0 1
/// 输出文件格式为： vector<cv::Mat> CameraGTpose, Mat的格式和上面一样
void Tracking::ReadVirtualKittiCameraGT(const std::string &cameraposefile)
{
    std::ifstream cameraposetxt(cameraposefile.c_str());
    std::string linecamerapose;
    // 第一行不要
    getline(cameraposetxt, linecamerapose);
    int frame_id;
    int camera_ID;

    while(getline(cameraposetxt, linecamerapose))
    {
        if(!linecamerapose.empty())
        {
            std::stringstream sscamerapose(linecamerapose);
            sscamerapose>>frame_id;
            sscamerapose>>camera_ID;
            cv::Mat Pose_tmp = cv::Mat::eye(4, 4, CV_32F);// 32位浮点数
            // 左目
            if(camera_ID==0)
            {
                sscamerapose >> Pose_tmp.at<float>(0,0) >> Pose_tmp.at<float>(0,1) >>Pose_tmp.at<float>(0,2) >>Pose_tmp.at<float>(0,3)
                             >> Pose_tmp.at<float>(1,0) >> Pose_tmp.at<float>(1,1) >> Pose_tmp.at<float>(1, 2) >> Pose_tmp.at<float>(1, 3)
                             >> Pose_tmp.at<float>(2,0) >> Pose_tmp.at<float>(2,1) >> Pose_tmp.at<float>(2,2) >> Pose_tmp.at<float>(2,3)
                             >> Pose_tmp.at<float>(3,0) >> Pose_tmp.at<float>(3,1) >> Pose_tmp.at<float>(3,2) >> Pose_tmp.at<float>(3,3);

                EvLeftCamGTPose.push_back(Pose_tmp);

            }
            // 右目
            else{

                sscamerapose >> Pose_tmp.at<float>(0,0) >> Pose_tmp.at<float>(0,1) >>Pose_tmp.at<float>(0,2) >>Pose_tmp.at<float>(0,3)
                             >> Pose_tmp.at<float>(1,0) >> Pose_tmp.at<float>(1,1) >> Pose_tmp.at<float>(1, 2) >> Pose_tmp.at<float>(1, 3)
                             >> Pose_tmp.at<float>(2,0) >> Pose_tmp.at<float>(2,1) >> Pose_tmp.at<float>(2,2) >> Pose_tmp.at<float>(2,3)
                             >> Pose_tmp.at<float>(3,0) >> Pose_tmp.at<float>(3,1) >> Pose_tmp.at<float>(3,2) >> Pose_tmp.at<float>(3,3);

                EvRightCamGTPose.push_back(Pose_tmp);

            }

        }
    }


}


/// Mynteye 读取object info: 要求只有一个目标, 数据总量等于目标的个数
void Tracking::ReadMynteyeObjectInfo(const std::string &objInfo)
{
    EvOfflineAllObjectDetections.reserve(EnImgTotalNum);
    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp;

    std::string pred_frame_obj_txts = objInfo;
    std::ifstream filetxt(pred_frame_obj_txts.c_str());
    std::string line;
    //int count = 0;
    while(getline(filetxt, line))
    {
        objects_inoneframe_temp.clear();
        if(line.empty())
            assert(0);

        Eigen::Matrix<double, 1, 24> oneobject_oneframe;
        std::stringstream ss(line);

        int frame_id;
        int track_id = 1;// (2) 帧的id， 目标的id
        double truncated = 0, occuluded = 1, alpha = 0;// (3) truncation的程度？？？ 是否遮挡， 目标的观测角
        Eigen::Vector4d bbox; // (4) 目标的2D框
        //Eigen::Vector3d dimensions(0.18, 0.53, 0.27);// (5) 目标的尺度xyz对应着目标坐标系的xyz, 而目标坐标系是与相机坐标系一致的, 即y是朝下
        //Eigen::Vector3d locations(0, 0.83, 1.6);// (6) 目标的相机系3D位置;
        //double rotation_y = 0;// (7) 目标的yaw角
        Eigen::Vector3d dimensions = EeigUniformObjScale;
        Eigen::Vector3d locations = EeigInitPosition;
        double rotation_y = EeigInitRotation(1);

        double score = 1;// (8) 置信度
        int type_id = 1;// (9) 类型：0是人，电车， dontcare等等; 1是需要跟踪的目标车辆
        int is_moving = 1;
        int extend = 0;

        ss>>frame_id;
        ss>>bbox[0];
        ss>>bbox[1];
        ss>>bbox[2]; // yolo 检测的输出的就是宽和高
        ss>>bbox[3];

        oneobject_oneframe [0] = frame_id;
        oneobject_oneframe [1] = track_id;
        oneobject_oneframe [2] = truncated;
        oneobject_oneframe [3] = occuluded; // 需要注意的是,这个含义与virtual kitti不同: integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
        oneobject_oneframe [4] = alpha;
        oneobject_oneframe [5] = bbox[0];
        oneobject_oneframe [6] = bbox[1];
        oneobject_oneframe [7] = bbox[2];
        oneobject_oneframe [8] = bbox[3];
        oneobject_oneframe [9] = dimensions[0];
        oneobject_oneframe [10] = dimensions[1];
        oneobject_oneframe [11] = dimensions[2];
        if(ORB_SLAM2::EbManualSetPointMaxDistance == false)
        {
            Eigen::Vector3f scale(dimensions[0]/2, dimensions[1]/2, dimensions[2]/2);
            ORB_SLAM2::EfInObjFramePointMaxDistance = scale.norm();
        }
        oneobject_oneframe [12] = locations[0];
        oneobject_oneframe [13] = locations[1];
        oneobject_oneframe [14] = locations[2];
        oneobject_oneframe [15] = rotation_y;
        oneobject_oneframe [16] = score;
        oneobject_oneframe [17] = type_id;
        oneobject_oneframe [18] = is_moving;
        oneobject_oneframe [19] = extend;
        oneobject_oneframe [20] = 0;// 右目相机的观测：bbox
        oneobject_oneframe [21] = 0;
        oneobject_oneframe [22] = 0;
        oneobject_oneframe [23] = 0;

        objects_inoneframe_temp.push_back(oneobject_oneframe);
        EvOfflineAllObjectDetections.push_back(objects_inoneframe_temp);
    }

//    if(EvOfflineAllObjectDetections.size() != EnImgTotalNum)
//    {
//        cout<<"目标信息个数与图像数量不相等!!"<<endl;
//        //assert(0);
//    }
}

std::vector<std::string> Tracking::LoadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open())
    {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Tracking::SetObjectLocalMapper(ObjectLocalMapping *pObjectLocalMapper)
{
    mpObjectLocalMapper=pObjectLocalMapper;
}


void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imColor = imRectLeft.clone();

    cv::Mat imGrayRight = imRectRight;
    mImGrayRight = imGrayRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    switch (EnSLOTMode)
    {
        case 0: // SLAM mode
        {
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
            break;
        }

        case 1: // Dynamic SLAM mode
        {
            if(EnDynaSLAMMode == 0)
            {
                mCurrentFrame = Frame(mImGray,imGrayRight, imColor,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,
                                      mYOLODetector);
            }
            else
            {
                mCurrentFrame = Frame(mImGray,imGrayRight, imColor,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,
                                      mYOLODetector, mMultiTracker, mvTrackers);
            }
            break;
        }

        case 2: // Object tracking mode
        {
            mCurrentFrame = Frame(mImGray,imGrayRight, imColor,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,
                                  mMultiTracker, mvTrackers);
            break;
        }

        case 3: // Autonomous Driving mode
        {
            mCurrentFrame = Frame(mImGray,imGrayRight, imColor,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,
                                  mYOLODetector, mDeepSort);
            break;
        }

        case 4: // Test mode
        {
            mCurrentFrame = Frame(mImGray,imGrayRight, imColor,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,
                    EnSLOTMode);
            break;
        }

        default:
            assert(0);
    }

    auto t2 = std::chrono::steady_clock::now();
    Track();
    auto t3 = std::chrono::steady_clock::now();
    //cout<<"Time comparision-----Frame establish: "<<std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count()<<", Track(): "<<std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count()<<endl;
    return mCurrentFrame.mTcw.clone();
}



void Tracking::Track()
{
    // mState为tracking的状态机
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }


    mLastProcessedState=mState;


    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);


    if(mState==NOT_INITIALIZED)
    {
        StereoInitialization();

        mpFrameDrawer->Update(this);
        mpMapDrawer->UpdateCurrentMap(this);


        if(mState!=OK)
            return;
    }
    else
    {

        bool bOK;


        auto t1 = std::chrono::steady_clock::now();
        if(mState==OK)
        {

            InheritObjFromLastFrame();

            CheckReplacedInLastFrame();


            if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            {

                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                bOK = TrackWithMotionModel();

                if(!bOK)
                    bOK = TrackReferenceKeyFrame();
            }
        }
        else
        {

            bOK = Relocalization();
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if(bOK)

            bOK = TrackLocalMap();
        auto t2 = std::chrono::steady_clock::now();

        switch(EnSLOTMode)
        {
            case 0: // SLAM
            {
                break;
            }

            case 1: // Semantic Dynamic SLAM
            {
                break;
            }

            case 2: // Object Tracking
            {

                if (mCurrentFrame.mnDetObj != 0)
                {
                    /// 9.2 Construct 3D objects (MapObject) from 2D instance detection
                    TrackMapObject();
                    if (mCurrentFrame.mvInLastFrameTrackedObjOrders.size() != 0)
                    {
                        CheckReplacedMapObjectPointsInLastFrame();
                        TrackLastFrameObjectPoint(false);
                    }

                    if (mCurrentFrame.mvTotalTrackedObjOrders.size() != 0)
                    {
                        TrackObjectLocalMap();
                    }
                }
                break;
            }

            case 3:
            {

                if (mCurrentFrame.mnDetObj != 0)
                {
                    TrackMapObject();
                    if(mCurrentFrame.mvInLastFrameTrackedObjOrders.size()!=0)
                    {
                        CheckReplacedMapObjectPointsInLastFrame();
                        TrackLastFrameObjectPoint(false);
                    }

                    if(mCurrentFrame.mvTotalTrackedObjOrders.size()!=0)
                    {
                        TrackObjectLocalMap();
                    }
                }

                break;
            }

            case 4: // 终极测试模式
            {

                TrackMapObject();
                if (mCurrentFrame.mvInLastFrameTrackedObjOrders.size() != 0)
                {
                    CheckReplacedMapObjectPointsInLastFrame();
                    TrackLastFrameObjectPoint(false);
                }

                if (mCurrentFrame.mvTotalTrackedObjOrders.size() != 0)
                {
                    TrackObjectLocalMap();
                }
                break;
            }

            default:
                assert(0);
        }
        auto t3 = std::chrono::steady_clock::now();

        // Moving Objects Recognition
        DynamicStaticDiscrimination();
        auto t4 = std::chrono::steady_clock::now();



        if(bOK)
            mState = OK;
        else
        {
            cout<<RED<<"Tracking is Lost!";
            mState = LOST;
        }

        //cout<<"Current mvDetectionObjects: "<<mCurrentFrame.mvDetectionObjects.size()<<", Current mvObjKeys: "<<mCurrentFrame.mvObjKeys.size()<<", mnDetObj: "<<mCurrentFrame.mnDetObj<<endl;
        mpFrameDrawer->Update(this);
        mpMapDrawer->UpdateCurrentMap(this);


        if(bOK)
        {
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPoseAndId(mCurrentFrame.mTcw, mCurrentFrame.mnId);

            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                }
            }
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
                pMP = NULL;
            }
            mlpTemporalPoints.clear();

            if(NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }

            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }

        }
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;


        // Post-Processing
        switch(EnSLOTMode)
        {
            case 0: // SLAM
            {
                break;
            }

            case 1: // Semantic Dynamic SLAM
            {
                break;
            }

            case 2: // Object Tracking
            {
                for(size_t i=0; i<mCurrentFrame.mnDetObj; i++)
                {
                    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[i];
                    MapObject* pMO = mCurrentFrame.mvMapObjects[i];
                    if(pDet==NULL||pMO==NULL)
                        assert(0);
                    if(pMO->mnFirstObservationFrameId == mCurrentFrame.mnId)
                        continue;

                    for(size_t j=0; j<mCurrentFrame.mvpMapObjectPoints[i].size(); j++)
                    {
                        MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[i][j];
                        if(pMP==NULL)
                            continue;
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvpMapObjectPoints[i][j] = static_cast<MapObjectPoint*>(NULL);
                            mCurrentFrame.mvbObjKeysOutlier[i][j] = false;
                        }
                    }
                    for(list<MapObjectPoint*>::iterator lit = pMO->mlpTemporalPoints.begin(), lend =  pMO->mlpTemporalPoints.end(); lit!=lend; lit++)
                    {
                        MapObjectPoint* pMP = *lit;
                        delete pMP;
                        pMP = NULL;
                    }
                    pMO->mlpTemporalPoints.clear();

                    if(pDet->mbTrackOK == true)
                    {
                        pMO->UpdateVelocity(mCurrentFrame.mnId);
                        if(NeedNewObjectKeyFrame(i))
                        {
                            CreateNewObjectKeyFrame(i);
                        }
                    }
                    else{
                        cout<<RED<<"Object "<<pDet->mnObjectID<<" tracking fails!"<<WHITE<<endl;
                        MapObjectReInit(i);
                    }
                }
                for(size_t n=0; n<mCurrentFrame.mnDetObj; n++)
                {
                    MapObject* pMO = mCurrentFrame.mvMapObjects[n];
                    if(pMO==NULL || pMO->mpReferenceObjKF == NULL)
                        continue;
                    g2o::SE3Quat x = pMO->GetCFInFrameObjState(mCurrentFrame.mnId).pose * pMO->GetCFObjectKeyFrameObjState(pMO->mpReferenceObjKF).pose.inverse();
                    pMO->mlRelativeFramePoses[mCurrentFrame.mnId] = make_pair(pMO->mpReferenceObjKF, x);
                }
                break;
            }

            case 3: // 自动驾驶模式
            {
                for(size_t i=0; i<mCurrentFrame.mnDetObj; i++)
                {
                    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[i];
                    MapObject* pMO = mCurrentFrame.mvMapObjects[i];
                    if(pDet==NULL||pMO==NULL)
                        assert(0);
                    if(pMO->mnFirstObservationFrameId == mCurrentFrame.mnId)
                        continue;

                    for(size_t j=0; j<mCurrentFrame.mvpMapObjectPoints[i].size(); j++)
                    {
                        MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[i][j];
                        if(pMP==NULL)
                            continue;
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvpMapObjectPoints[i][j] = static_cast<MapObjectPoint*>(NULL);
                            mCurrentFrame.mvbObjKeysOutlier[i][j] = false;
                        }
                    }
                    for(list<MapObjectPoint*>::iterator lit = pMO->mlpTemporalPoints.begin(), lend =  pMO->mlpTemporalPoints.end(); lit!=lend; lit++)
                    {
                        MapObjectPoint* pMP = *lit;
                        delete pMP;
                        pMP = NULL;
                    }
                    pMO->mlpTemporalPoints.clear();

                    if(pDet->mbTrackOK == true)
                    {
                        pMO->UpdateVelocity(mCurrentFrame.mnId);
                        if(NeedNewObjectKeyFrame(i))
                        {
                            CreateNewObjectKeyFrame(i);
                        }
                    }
                    else{
                        cout<<RED<<"Object "<<pDet->mnObjectID<<" tracking fails!"<<WHITE<<endl;
                        MapObjectReInit(i);
                    }
                }
                for(size_t n=0; n<mCurrentFrame.mnDetObj; n++)
                {
                    MapObject* pMO = mCurrentFrame.mvMapObjects[n];
                    if(pMO==NULL || pMO->mpReferenceObjKF == NULL)
                        continue;
                    g2o::SE3Quat x = pMO->GetCFInFrameObjState(mCurrentFrame.mnId).pose * pMO->GetCFObjectKeyFrameObjState(pMO->mpReferenceObjKF).pose.inverse();
                    pMO->mlRelativeFramePoses[mCurrentFrame.mnId] = make_pair(pMO->mpReferenceObjKF, x);
                }
                break;
            }

            case 4:
            {
                for(size_t i=0; i<mCurrentFrame.mnDetObj; i++)
                {
                    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[i];
                    MapObject* pMO = mCurrentFrame.mvMapObjects[i];
                    if(pDet==NULL||pMO==NULL)
                        continue;
                    if(pMO->mnFirstObservationFrameId == mCurrentFrame.mnId)
                        continue;

                    for(size_t j=0; j<mCurrentFrame.mvpMapObjectPoints[i].size(); j++)
                    {
                        MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[i][j];
                        if(pMP==NULL)
                            continue;
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvpMapObjectPoints[i][j] = static_cast<MapObjectPoint*>(NULL);
                            mCurrentFrame.mvbObjKeysOutlier[i][j] = false;
                        }
                    }
                    for(list<MapObjectPoint*>::iterator lit = pMO->mlpTemporalPoints.begin(), lend =  pMO->mlpTemporalPoints.end(); lit!=lend; lit++)
                    {
                        MapObjectPoint* pMP = *lit;
                        delete pMP;
                        pMP = NULL;
                    }
                    pMO->mlpTemporalPoints.clear();

                    if(pDet->mbTrackOK == true)
                    {
                        pMO->UpdateVelocity(mCurrentFrame.mnId);
                        if(NeedNewObjectKeyFrame(i))
                        {
                            CreateNewObjectKeyFrame(i);
                        }
                    }
                    else{
                        //cout<<RED<<"Object "<<pDet->mnObjectID<<" tracking fails!"<<WHITE<<endl;
                        MapObjectReInit(i);
                    }
                }
                for(size_t n=0; n<mCurrentFrame.mnDetObj; n++)
                {
                    MapObject* pMO = mCurrentFrame.mvMapObjects[n];
                    if(pMO==NULL || pMO->mpReferenceObjKF == NULL)
                        continue;
                    g2o::SE3Quat x = pMO->GetCFInFrameObjState(mCurrentFrame.mnId).pose * pMO->GetCFObjectKeyFrameObjState(pMO->mpReferenceObjKF).pose.inverse();
                    pMO->mlRelativeFramePoses[mCurrentFrame.mnId] = make_pair(pMO->mpReferenceObjKF, x);
                }
                break;
            }

            default:
                assert(0);
        }
        auto t5 = std::chrono::steady_clock::now();
//        cout<<"Time comparision-----camera tracking: "<<std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count()<<", Object tracking: "<<std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count()
//        <<"， DynamicfDetection: "<<std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count()
//        <<", postprocessing:"<<std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count()<<endl;
    }

    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

    if(mState!=NOT_INITIALIZED)
    {
        mLastFrame = Frame(mCurrentFrame);
    }

}


/// 跟踪动态目标, 由2D objects建立3D objects
/// 3D objects属性: MapObject 最关键的两个属性是allFrameCuboidPoses(普通帧pose) 与 allDynamicPoses (关键帧pose)
/// 策略： 若出现过(则利用速度模型)则更新3D object的pose(也可选择直接由2D离线结果给定): 若在时间阈值内，则利用速度模型;
/// 若不在，则直接利用2D离线结果(即3D目标检测);
/// 若没有出现过(在AllObjects里没有)，说明该object第一次观测到，则建立3D object(其初始位姿由2D离线结果给定)并加入AllObjects
bool Tracking::TrackMapObject()
{
    if(mCurrentFrame.mvDetectionObjects.size()==0)
        return false;

    //cout<<endl<<"3D目标跟踪开始： "<<endl;

    /// 1. 遍历所有当前帧检测的2D objects：cuboids_on_frame.
    // 判断之前是否出现过(在3D objects(AllObjects)是否有相同的):
    // 若出现过(则利用速度模型)则更新3D object的pose: 更新方法: 若在时间阈值内，则利用速度模型; 若不在，则直接利用3D目标检测结果;
    for(size_t i=0; i<mCurrentFrame.mnDetObj;i++)
    {
        DetectionObject *candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];

        if(candidate_cuboid==NULL)
            continue;

        // 标志位： 这次检测的2D object是否出现过
        bool object_exists = false;

        // 遍历AllObjects： 目前出现过的所有3D objects.
        // 判断方法是判断2D bouding box与3D bounding box的truth id是否相同
        for(size_t j=0; j<AllObjects.size();j++)
        {
            MapObject *object_temp = AllObjects[j];
            // 判定2D object的id与3D object的id是否相同
            if(object_temp->mnTruthID == candidate_cuboid->mnObjectID)
            {
                g2o::ObjectState cInFrameLatestObjState;
                // 更新上一次的目标状态
                int nLatestFrameId = object_temp->GetCFLatestFrameObjState(cInFrameLatestObjState);// Tco
                if(nLatestFrameId == -1)
                    assert(0);
                if(!object_temp->mlRelativeFramePoses.count(nLatestFrameId)) // 如果最近帧没有参考关键帧肯定有问题
                    assert(0);
                if(object_temp->mpReferenceObjKF !=  object_temp->mlRelativeFramePoses[nLatestFrameId].first) // 参考帧不一致
                    assert(0);
                cv::Mat camera_Tcl = cv::Mat::eye(4, 4, CV_32F);
                if (!mCurrentFrame.mTcw.empty() && !mLastFrame.mTwc.empty()){
                    camera_Tcl = mCurrentFrame.mTcw.clone() * mLastFrame.mTwc.clone();
                }
                cInFrameLatestObjState.pose = Converter::toSE3Quat(camera_Tcl) * cInFrameLatestObjState.pose; // 更新

                if(cInFrameLatestObjState.scale[0]<0.05)
                    assert(0);

                g2o::ObjectState cuboid_current_temp;// 更新object在当前帧的pose： cuboid_current_temp
                double delta_t = (mCurrentFrame.mnId - nLatestFrameId) * EdT;
                if(delta_t < EdMaxObjMissingDt && EbUseOfflineAllObjectDetectionPosesFlag == false)// 判断当前帧离上一次观测到object是否没过去很久 (做为是否可以使用速度模型的条件)
                {
                    /// 匀速模型,预测object的下一帧pose, 并且更新目标速度
                    //object_temp->mVirtualVelocity = Vector6d::Zero();
                    cuboid_current_temp = cInFrameLatestObjState;
                    bool currentpose = InitializeCurrentObjPose(i, cuboid_current_temp);
                    if (!currentpose)
                    // 被遮挡情况下，利用速度预测
                    cuboid_current_temp.UsingVelocitySetPredictPos(object_temp->mVirtualVelocity, delta_t);
                }
                else{
                    /// (2) 如果不在阈值内, 直接用当前帧的检测结果：作为该object的最新pose
                    // 1) 从检测结果(candidate_cuboid)中读出读出当前帧cuboid的pose: cuboid_current_temp, 这是在camera系
//                    Vector9d cube_pose;
//                    cube_pose << candidate_cuboid->mPos[0], candidate_cuboid->mPos[1], candidate_cuboid->mPos[2], 0, candidate_cuboid->mdRotY, 0, candidate_cuboid->mScale[0],
//                            candidate_cuboid->mScale[1], candidate_cuboid->mScale[2];
//                    cuboid_current_temp.fromMinimalVector(cube_pose);
                    //g2o::SE3Quat frame_pose_to_init = Converter::toSE3Quat(mCurrentFrame.mTwc);
                    //cuboid_current_temp = cuboid_current_temp.transform_from(frame_pose_to_init);
                    cuboid_current_temp = cInFrameLatestObjState;
                    InitializeCurrentObjPose(i, cuboid_current_temp);
                }
                // Fine Tuning
                FineTuningUsing2dBox(i,cuboid_current_temp);

                g2o::ObjectState Swo(mCurrentFrame.mSETcw.inverse() * cuboid_current_temp.pose, cuboid_current_temp.scale);
                object_temp->SetInFrameObjState(Swo, mCurrentFrame.mnId);// 设置该object的当前帧观测
                object_temp->SetCFInFrameObjState(cuboid_current_temp, mCurrentFrame.mnId);


                candidate_cuboid->SetMapObject(object_temp);// cuboid 所属object给它绑定， 但是object的观测没有加因为用不到
                object_temp->AddFrameObservation(mCurrentFrame.mnId, i);// 该object被观测次数+1
                mCurrentFrame.mvMapObjects[i] = object_temp;// object在帧中与它的观测一一对应
                // 存入该object的当前帧观测
                object_temp->mmmDetections[mCurrentFrame.mnId] = candidate_cuboid;

                //cv::waitKey(0);

                object_exists = true;
                if(nLatestFrameId == int(mLastFrame.mnId))
                {
                    //cout<<"目标: "<<candidate_cuboid->mnObjectID<<" 成功跟踪上一帧"<<", 目标速度: "<<object_temp->mVirtualVelocity.transpose()<<" dt: "<<delta_t<<endl;;
                    int nInLastFrameDetObjOrder = mLastFrame.FindDetectionObject(candidate_cuboid);
                    if(nInLastFrameDetObjOrder == -1)
                        assert(0);
                    mCurrentFrame.mvInLastFrameTrackedObjOrders.push_back(make_pair(size_t(nInLastFrameDetObjOrder), i));
                }
                else{
                    candidate_cuboid->SetDynamicFlag(object_temp->GetDynamicFlag()) ;
                    //cout<<"Object: "<<candidate_cuboid->mnObjectID<<"  successfully tracked, but not in the last frame, dynamics： " <<candidate_cuboid->GetDynamicFlag()<<endl;
                }
                mCurrentFrame.mvTotalTrackedObjOrders.push_back(i);
                break;
            }

        }
        // 若没有出现过(在AllObjects里没有)，说明该object第一次观测到，则建立3D object并加入AllObjects
        if(object_exists==false)// 该目标第一次出现
        {
            MapObjectInit(i);
        }
    }

    bool bFlag = mCurrentFrame.mvTotalTrackedObjOrders.size() ==0 && mCurrentFrame.mvnNewConstructedObjOrders.size() == 0;
    if(bFlag)
    {
        cout<<"This frame does not track or construct any object."<<endl;
    }
    return !bFlag;
}

bool Tracking::InitializeCurrentObjPose(const int &i, g2o::ObjectState &Pose) {
    //每帧initialize对准2D框，创建关键帧时考虑fit 3D框
    //另外 没点的物体tracking状态没有保持，修改一下(还是加上了Reinit)
    DetectionObject* candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];
    int nNum = 0; // 三角化目标点
    vector<pair<Eigen::Vector3d, int>> Pcjs;
    Pcjs.reserve(mCurrentFrame.mvObjKeysUn[i].size());
    for(size_t n=0; n<mCurrentFrame.mvObjKeysUn[i].size();n++)
    {
        float z = mCurrentFrame.mvObjPointDepth[i][n];

        if(z>0)
        {
            cv::Mat x3Dc = mCurrentFrame.UnprojectStereodynamic(i, n, false);
            Eigen::Vector3d Pcj = Converter::toVector3d(x3Dc);
            Pcjs.push_back(make_pair(Pcj, n));
            nNum++;

        }
    }

    // RANSAC滤出一些离谱点
    std::vector<int> best_inlier_set;
    float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: candidate_cuboid->mTruthPosInCameraFrame.scale.norm();
    unsigned int l = Pcjs.size();
    cv::RNG rng;
    double bestScore = -1.;
    int iterations = 0.8 * Pcjs.size();
    for(int k=0; k<iterations; k++) // 迭代次数
    {
        std::vector<int> inlier_set;
        int i1 = rng(l);
        const Eigen::Vector3d& p1 = Pcjs[i1].first;
        double score = 0;
        for(int u=0; u<l; u++)
        {
            Eigen::Vector3d v = Pcjs[u].first-p1;
            if( v.norm() < fMaxDis ) // 到直线的距离
            {
                inlier_set.push_back(u);// 将点添加到集合中
                score++;
            }
        }
        if(score > bestScore)
        {
            bestScore = score;
            best_inlier_set = inlier_set;
        }
    }
    if (best_inlier_set.size()<3)
        return false;
    Eigen::Vector3d eigObjPosition(0, 0, 0);
    for(auto &q: best_inlier_set)
    {
        eigObjPosition+=Pcjs[q].first;
    }
    eigObjPosition /= best_inlier_set.size();
    if (eigObjPosition[2]>8)
    eigObjPosition[2] += 0.2*candidate_cuboid->mScale[0];
    eigObjPosition[1] = 0 + candidate_cuboid->mScale[1]/2;
    Pose.setTranslation(eigObjPosition);
    //cout<<"Object "<<i<<", initialize Pose is True!"<<endl;

    return true;
}
void Tracking::FineTuningUsing2dBox(const int &i, g2o::ObjectState &Pose)
{
    DetectionObject * cuboid = mCurrentFrame.mvDetectionObjects[i];
    // Fine Tuning
    // Align 2D box
    cv::Rect rectbox = cuboid->mrectBBox;
    auto Tcw = mCurrentFrame.mSETcw;
    auto K = mdCamProjMatrix;
    auto proj = Pose.projectOntoImageRectFromCamera(K);
    cv::Rect projbox =cv::Rect(proj[0],proj[1],proj[2]-proj[0],proj[3]-proj[1]);
//    cv::Mat showimg;
//    mImGray.copyTo(showimg);
//    cv::rectangle(showimg, projbox,cv::Scalar(0,0,255), 2);
//    cv::rectangle(showimg, rectbox,cv::Scalar(255,0,0), 2);
//    cv::imshow("Projection 2d box comparison", showimg);
//    cv::waitKey(0);
    // Rectbox和projbox的对准策略：
    // 解耦三个变量对准
    // 方框中心作差，移动3D框中心
    // 方框高度利用深度微调。
    // 方框宽度利用3D框yaw角来微调。

    // 1.方框中心y坐标调整
    double y1 = Pose.translation()[1];
    double y = Pose.translation()[1];
    cv::Point_<double> delta_center = (projbox.tl()+projbox.br()) / 2 - (rectbox.tl()+rectbox.br()) / 2;
    double step = 0.01;//m
    for(int i=0;i<400;i++){
        //cout<<"delta center: "<<delta_center.x<<" "<<delta_center.y<<endl;
        y = Pose.translation()[1];
        int direction = 1;
        if (delta_center.y<0) direction = -1;

        y = y - direction * step;
        Pose.setTranslation(Eigen::Vector3d(Pose.translation()[0],y,Pose.translation()[2]));
        proj = Pose.projectOntoImageRectFromCamera(K);
        projbox =cv::Rect(proj[0],proj[1],proj[2]-proj[0],proj[3]-proj[1]);
        delta_center = (projbox.tl()+projbox.br()) / 2 - (rectbox.tl()+rectbox.br()) / 2;
        if (abs(delta_center.y)<1) { break; }
    }
    //cout << "前后对比情况" << y1 << " " << Pose.translation()[1] << ", gt" << cuboid->mTruthPosInCameraFrame.translation()[1] << endl;

    // 2.根据投影方框高度到2D检测方框中心对比来调整深度
    double z = Pose.translation()[2];
    if (z>8) {
        double z1 = Pose.translation()[2];
        double delta_h = projbox.height - rectbox.height;
        step = 0.05;//m
        for (int i = 0; i < 400; i++) {
            //cout << delta_h << " ";
            z = Pose.translation()[2];
            int direction = 1;
            if (delta_h < 0) direction = -1;

            z = z + direction * step;
            Pose.setTranslation(Eigen::Vector3d(Pose.translation()[0], Pose.translation()[1], z));
            proj = Pose.projectOntoImageRectFromCamera(K);
            projbox = cv::Rect(proj[0], proj[1], proj[2] - proj[0], proj[3] - proj[1]);
            delta_h = projbox.height - rectbox.height;
            if (abs(delta_h) < 1) {
                //cout << endl << "迭代次数" << i << "  " << delta_h << endl;
                break;
            }
        }
        //cout << "前后对比情况" << z1 << " " << Pose.translation()[2] << ", gt"<< cuboid->mTruthPosInCameraFrame.translation()[2] << endl;
    }

    // 3.根据投影方框中心到2D检测方框中心对比，调整x坐标
    double x = Pose.translation()[0];
    double x1 = Pose.translation()[0];
    delta_center = (projbox.tl()+projbox.br()) / 2 - (rectbox.tl()+rectbox.br()) / 2;
    step = 0.01;//m
    for (int i = 0; i < 400; i++) {
        //cout<<delta_center.x<<" ";
        x = Pose.translation()[0];
        int direction = 1;
        if (delta_center.x<0) direction = -1;

        x = x - direction * step;
        Pose.setTranslation(Eigen::Vector3d(x,Pose.translation()[1],Pose.translation()[2]));
        proj = Pose.projectOntoImageRectFromCamera(K);
        projbox =cv::Rect(proj[0],proj[1],proj[2]-proj[0],proj[3]-proj[1]);
        delta_center = (projbox.tl()+projbox.br()) / 2 - (rectbox.tl()+rectbox.br()) / 2;
        if (abs(delta_center.x) < 1) {
            //cout << endl << "迭代次数" << i << "  " << delta_center.x << endl;
            break;
        }
    }
    //cout << "前后对比情况" << x1 << " " << Pose.translation()[0] << ", gt"<< cuboid->mTruthPosInCameraFrame.translation()[0] << endl;

//    // 4.根据投影方框的宽度到2D检测框宽度，调整yaw角
//    double yaw = Pose.pose.toXYZPRYVector()[4];
//    double yaw1 = Pose.pose.toXYZPRYVector()[4];
//    double delta_w = projbox.width - rectbox.width;
//    step = 0.0001;//m
//    for (int i = 0; i < 400; i++) {
//        cout << delta_w << " ";
//        yaw = Pose.pose.toXYZPRYVector()[4];
//        int direction = 1;
//        if (delta_w < 0) direction = -1;
//
//        yaw = yaw - direction * step;
//        cout<<yaw<<endl;
//        Pose.setRotation(Eigen::Vector3d(0,yaw,0));
//        proj = Pose.projectOntoImageRectFromCamera(K);
//        projbox = cv::Rect(proj[0], proj[1], proj[2] - proj[0], proj[3] - proj[1]);
//        delta_w = projbox.width - rectbox.width;
//        if (abs(delta_w) <= 1) {
//            cout << endl << "迭代次数" << i << "  " << delta_w << endl;
//            break;
//        }
//    }
//    cout << "前后对比情况" << yaw1 << " " << yaw << ", gt"<< cuboid->mTruthPosInCameraFrame.pose.toXYZPRYVector()[4] << endl;
//
//    cv::Mat showimg2;
//    mImGray.copyTo(showimg2);
//    cv::rectangle(showimg2, projbox,cv::Scalar(0,0,255), 2);
//    cv::rectangle(showimg2, rectbox,cv::Scalar(255,0,0), 2);
//    cv::imshow("Projection 2d box comparison", showimg2);

    // 3D框对齐点云
    // 首先利用初始yaw角获得o系和o系下的点云
    // 然后fit一个3D框，计算该3D框的objectstate，替换掉原来的Objctstate。
}
void Tracking::MapObjectInit(const int &i)
{
    DetectionObject* candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];
    g2o::ObjectState gt = candidate_cuboid->mTruthPosInCameraFrame;
    g2o::ObjectState InitPose = candidate_cuboid->mTruthPosInCameraFrame;

    // initialize the object pose
    //InitPose.setRotation(Eigen::Vector3d(0,0,0));
    //InitPose.setScale(ORB_SLAM2::EeigUniformObjScale);
    int nNum = 0; // 三角化目标点
    vector<pair<Eigen::Vector3d, int>> Pcjs;
    Pcjs.reserve(mCurrentFrame.mvObjKeysUn[i].size());
    for(size_t n=0; n<mCurrentFrame.mvObjKeysUn[i].size();n++)
    {
        float z = mCurrentFrame.mvObjPointDepth[i][n];

        if(z>0)
        {
            cv::Mat x3Dc = mCurrentFrame.UnprojectStereodynamic(i, n, false);
            Eigen::Vector3d Pcj = Converter::toVector3d(x3Dc);
            Pcjs.push_back(make_pair(Pcj, n));
            nNum++;

        }
    }

    // RANSAC滤出一些离谱点
    std::vector<int> best_inlier_set;
    float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: InitPose.scale.norm();
    unsigned int l = Pcjs.size();
    cv::RNG rng;
    double bestScore = -1.;
    int iterations = 0.8 * Pcjs.size();
    for(int k=0; k<iterations; k++) // 迭代次数
    {
        std::vector<int> inlier_set;
        int i1 = rng(l);
        const Eigen::Vector3d& p1 = Pcjs[i1].first;
        double score = 0;
        for(int u=0; u<l; u++)
        {
            Eigen::Vector3d v = Pcjs[u].first-p1;
            if( v.norm() < fMaxDis ) // 到直线的距离
            {
                inlier_set.push_back(u);// 将点添加到集合中
                score++;
            }
        }
        if(score > bestScore)
        {
            bestScore = score;
            best_inlier_set = inlier_set;
        }
    }
    if (best_inlier_set.size()<3)
        return;
    Eigen::Vector3d eigObjPosition(0, 0, 0);
    for(auto &q: best_inlier_set)
    {
        eigObjPosition+=Pcjs[q].first;
    }
    eigObjPosition /= best_inlier_set.size();

    // Soft prior for car
    if (eigObjPosition[2]<8)
        return;

    eigObjPosition[2] += 0.2*candidate_cuboid->mScale[0];
    eigObjPosition[1] = 0 + candidate_cuboid->mScale[1]/2;

    cout<<"Object: "<<candidate_cuboid->mnObjectID<<" initposition: "<<eigObjPosition.transpose()<<endl;

    //if(EbSetInitPositionByPoints && best_inlier_set.size() > 0)
    // Fine Tuning
    InitPose.setTranslation(eigObjPosition);
    FineTuningUsing2dBox(i,InitPose);

    /// 3.1.2 读出当前帧cuboid的pose:   Two = Twc * Tco
    MapObject *new_object = new MapObject(candidate_cuboid->mnObjectID, candidate_cuboid->GetDynamicFlag(),mCurrentFrame.mnId, true, Vector6d::Zero(), InitPose, true);// 建立object对象，设置其当前帧测量
    new_object->mbPoseInit = candidate_cuboid->mInitflag;
    new_object->SetInFrameObjState(g2o::ObjectState(mCurrentFrame.mSETcw.inverse() * InitPose.pose, InitPose.scale), mCurrentFrame.mnId);
    AllObjects.push_back(new_object);
    candidate_cuboid->SetMapObject(new_object);
    mpMap->AddMapObject(new_object);
    mCurrentFrame.mvMapObjects[i] = new_object;
    new_object->AddFrameObservation(mCurrentFrame.mnId, i);
    mCurrentFrame.mvnNewConstructedObjOrders.push_back(i);
    ObjectKeyFrame* pKFini = new ObjectKeyFrame(mCurrentFrame, i, true);// 建立目标关键帧
    new_object->mnLastKeyFrameId = mCurrentFrame.mnId; // 该目标的上一次关键帧就是这帧
    new_object->SetCFObjectKeyFrameObjState(pKFini, InitPose);

    // 存入该object的当前帧观测
    new_object->mmmDetections[mCurrentFrame.mnId] = candidate_cuboid;

    int nReal = 0;
    for(auto &j: best_inlier_set)
    {
        Eigen::Vector3d Pcj = Pcjs[j].first;
        int n = Pcjs[j].second;
        Eigen::Vector3d x3Do = InitPose.pose.inverse().map(Pcj);//Toc * Pcj
        if(x3Do.norm() > fMaxDis)
            continue;
        MapObjectPoint* pNewMP = new MapObjectPoint(new_object, Converter::toCvMat(x3Do), Converter::toCvMat(Pcj), pKFini);
        if (!pNewMP->mpRefObjKF) assert(0);
        pNewMP->AddObservation(pKFini,n);
        pKFini->AddMapObjectPoint(pNewMP,n);
        pNewMP->ComputeDistinctiveDescriptors();
        new_object->AddMapObjectPoint(pNewMP);

        pNewMP->UpdateNormalAndDepth();
        mCurrentFrame.mvpMapObjectPoints[i][n]=pNewMP;
        nReal++;
    }

    // 需不需要根据这个点数多少说明目标不行
    cout<<"Object: "<<candidate_cuboid->mnObjectID<<"  constructed successfully, "<<" features number: "<<nReal<<endl;
    cout<<endl;
    if(EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4)
        mpObjectLocalMapper->InsertOneObjKeyFrame(pKFini);
    new_object->mpReferenceObjKF = pKFini; // 更新目标的参考关键帧
}

void Tracking::MapObjectReInit(const int &order)
{
    MapObject* pMO = mCurrentFrame.mvMapObjects[order];
    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[order];
    g2o::ObjectState InitPose = pDet->mTruthPosInCameraFrame;

    // initialize the object pose
//    InitPose.setRotation(Eigen::Vector3d(0,0,0));
//    InitPose.setScale(ORB_SLAM2::EeigUniformObjScale);
    pMO->ClearMapObjectPoint();

    {
        // 1. 三角化目标点， 求平均值

        vector<pair<Eigen::Vector3d, int>> Pcjs;
        Pcjs.reserve(mCurrentFrame.mvObjKeysUn[order].size());


        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.mvObjKeysUn[order].size());
        for(size_t n=0; n<mCurrentFrame.mvObjKeysUn[order].size();n++)
        {
            float z = mCurrentFrame.mvObjPointDepth[order][n];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z, n));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());
            int nNum = 0; // 三角化目标点
            for(size_t j=0; j<vDepthIdx.size(); j++)
            {
                cv::Mat x3Dc = mCurrentFrame.UnprojectStereodynamic(order, vDepthIdx[j].second, false);
                Eigen::Vector3d Pcj = Converter::toVector3d(x3Dc);
                Pcjs.push_back(make_pair(Pcj, vDepthIdx[j].second));
                nNum++;
                if(vDepthIdx[j].first>2*mThDepth && nNum>100) // 意思是，超过有100个点，并且距离已经不准确了
                    break;
            }

        }



        /*
        for(size_t n=0; n<mCurrentFrame.mvObjKeysUn[order].size();n++)
        {
            float z = mCurrentFrame.mvObjPointDepth[order][n];
            //if(z > mThDepth)
            //    continue;
            if(z>0)
            {
                cv::Mat x3Dc = mCurrentFrame.UnprojectStereodynamic(order, n, false);
                Eigen::Vector3d Pcj = Converter::toVector3d(x3Dc);
                Pcjs.push_back(make_pair(Pcj, n));
                nNum++;
            }
            if(z > mThDepth && nNum > 50)
                break;
        }*/

        // RANSAC滤出一些离谱点
        std::vector<int> best_inlier_set;
        float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: InitPose.scale.norm();
        unsigned int l = Pcjs.size();
        cv::RNG rng;
        double bestScore = -1.;
        int iterations = Pcjs.size();
        for(int k=0; k<iterations; k++) // 迭代次数
        {
            std::vector<int> inlier_set;
            int i1 = rng(l);
            const Eigen::Vector3d& p1 = Pcjs[i1].first;
            double score = 0;
            for(int u=0; u<l; u++)
            {
                Eigen::Vector3d v = Pcjs[u].first-p1;
                if( v.norm() < fMaxDis ) // 到直线的距离
                {
                    inlier_set.push_back(u);// 将点添加到集合中
                    score++;
                }
            }
            if(score > bestScore)
            {
                bestScore = score;
                best_inlier_set = inlier_set;
            }
        }
        if (best_inlier_set.size()<=3)
            return;
        Eigen::Vector3d eigObjPosition(0, 0, 0);
        for(auto &q: best_inlier_set)
        {

            eigObjPosition+=Pcjs[q].first;
        }
        eigObjPosition /= best_inlier_set.size();
        if (eigObjPosition[2]<8)
            return;
        // 2. 是否用目标点平均位置修正目标pose
        if (eigObjPosition[2]>8)
        eigObjPosition[2] += 0.2*pDet->mScale[0];
        eigObjPosition[1] = 0 + pDet->mScale[1]/2;
        //if(EbSetInitPositionByPoints && best_inlier_set.size() > 0)

        InitPose.setTranslation(eigObjPosition);
        FineTuningUsing2dBox(order,InitPose);
        // 直接修正，目标的pose
        pMO->SetCFInFrameObjState(InitPose, mCurrentFrame.mnId);

        ObjectKeyFrame* pOKF = new ObjectKeyFrame(mCurrentFrame, order, true);
        pMO->mpReferenceObjKF = pOKF;
        pMO->SetCFObjectKeyFrameObjState(pOKF,pDet->mTruthPosInCameraFrame);

        // 新建立的目标点
        int num1 = 0;
        for(auto &j: best_inlier_set)
        {
            Eigen::Vector3d Pcj = Pcjs[j].first;
            int n = Pcjs[j].second;
            Eigen::Vector3d x3Do = InitPose.pose.inverse().map(Pcj);//Toc * Pcj
            if(x3Do.norm() > fMaxDis)
                continue;

            MapObjectPoint* pNewMP = new MapObjectPoint(pMO, Converter::toCvMat(x3Do), Converter::toCvMat(Pcj), pOKF);
            if (!pNewMP->mpRefObjKF) assert(0);
            pNewMP->AddObservation(pOKF,n);
            pOKF->AddMapObjectPoint(pNewMP, n);
            pMO->AddMapObjectPoint(pNewMP);

            pNewMP->ComputeDistinctiveDescriptors();
            pNewMP->UpdateNormalAndDepth();

            mCurrentFrame.mvpMapObjectPoints[order][n]=pNewMP;
            num1++;
        }
        //cout<<"目标: "<<pDet->mnObjectID<<"跟踪失败后，重新建立"<<"新三角化目标点数: "<<num1<<endl;
        if(EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4)
            mpObjectLocalMapper->InsertOneObjKeyFrame(pOKF);
    }
}

void Tracking::InheritObjFromLastFrame(){

    if(mCurrentFrame.mvDetectionObjects.size()==0)
        return;
    for(size_t i=0; i<mCurrentFrame.mnDetObj;i++) {
        DetectionObject *candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];

        if (candidate_cuboid == NULL)
            continue;

        // 遍历AllObjects： 目前出现过的所有3D objects.
        // 判断方法是判断2D bouding box与3D bounding box的truth id是否相同
        for (size_t j = 0; j < AllObjects.size(); j++) {
            MapObject *object_temp = AllObjects[j];
            // 判定2D object的id与3D object的id是否相同
            if (object_temp->mnTruthID == candidate_cuboid->mnObjectID) {
                candidate_cuboid->SetMapObject(object_temp);// cuboid 所属object给它绑定
            }
            //mapobject上一帧匹配到观测，但该帧没有匹配到，则需要进一步判断是否需要预测框
            // 判断上一帧是否在图像边界
            // 预测框
            // 判断该框是否在图像边界
            // 构造一下 DetectionObject
        }
    }
    StaticPointRecoveryFromObj();
}


void Tracking::DynamicStaticDiscrimination()
{
    // 把这些点当成是静态点计算重投影误差
    //cout<<endl<<YELLOW<<"动静态属性判断: "<<WHITE<<endl;
    vector<pair<size_t, size_t>> vnpTrackedObjOrders = mCurrentFrame.mvInLastFrameTrackedObjOrders;
    vector<DetectionObject*> vDetObj = mCurrentFrame.mvDetectionObjects;
    size_t nInCurrentFrameOrder;
    for(size_t l=0; l<vnpTrackedObjOrders.size(); l++)
    {
        nInCurrentFrameOrder = vnpTrackedObjOrders[l].second;
        DetectionObject* cDO = vDetObj[nInCurrentFrameOrder];
        if(cDO == NULL || mLastFrame.FindDetectionObject(cDO) == -1)  // 当前帧或上一帧没有观测 说明有问题
            assert(0);
        MapObject* pMO = cDO->GetMapObject();
        g2o::ObjectState curent_ObjSta = pMO->GetCFInFrameObjState(mCurrentFrame.mnId);

//        if (pMO->mbDynamicChanged){
//            cDO->SetDynamicFlag(pMO->GetDynamicFlag());
//            continue;
//        }
        double depth = curent_ObjSta.translation()[2];
        //cout<<"目标 "<<cDO->mnObjectID<<" 距离 "<< depth;
        //if (pMO->mVirtualVelocity.norm()!=0) cout<<RED<<", 速度 "<<pMO->mVirtualVelocity.norm()<<WHITE;
        cout<<endl;
        if (depth<7 || depth >mThDepth){
            cDO->SetDynamicFlag(pMO->GetDynamicFlag());
            continue;
        }
        //引入图像先验判断该object是否为正前方的移动车辆
        double middle_x = mImGray.size[1]/2;
        double current_px = cDO->mrectBBox.x+cDO->mrectBBox.width/2;
        double a = abs(current_px-middle_x);

        if (a<(cDO->mrectBBox.width/2+60)){
            cDO->SetDynamicFlag(true);
            cDO->GetMapObject()->SetDynamicFlag(true);
            //cout<<RED<<"目标"<<cDO->mnObjectID<<", 先验: "<<a<<" "<<aa<<endl<<WHITE;
            continue;
        }

        // 计算动态平均值
        double monoDynaValAvg = 0, stereoDynaValAvg = 0;
        vector<double> monoDynaVal, stereoDynaVal;
        int monoPointNum = 0, stereoPointNum = 0;

        vector<MapObjectPoint*> vMOPs = mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder];
        g2o::ObjectState last_objSta = pMO->GetCFInFrameObjState(mLastFrame.mnId); // Tco
        g2o::SE3Quat last_pose = mLastFrame.mSETcw; // Tcw
        g2o::SE3Quat current_pose = mCurrentFrame.mSETcw; // Tcw
        for(size_t j=0; j<vMOPs.size(); j++)
        {
            if(vMOPs[j] == NULL)
                continue;
            // 如果目标特征点太少怎么
            // 计算track上的特征点个数
            MapObjectPoint* pMOP = vMOPs[j];
            Eigen::Vector3d Plc = last_objSta.pose * pMOP->GetInObjFrameEigenPosition(); // pcj = Tco * Poj
            Eigen::Vector3d Pc = current_pose * last_pose.inverse() * Plc; //transformation from last frame to current frame
            // 假设它是静态,在当前帧的投影位置
            //Eigen::Vector3d Pc = Tcw * wP;
            const double invz = 1.0 / Pc[2];
            if(mCurrentFrame.mvuObjKeysRight[nInCurrentFrameOrder][j] < 0)
            {

                Eigen::Vector2d Zp;
                Zp[0] = mdCamProjMatrix(0,2) + Pc[0] * invz * mdCamProjMatrix(0, 0);
                Zp[1] = mdCamProjMatrix(1,2) + Pc[1] * invz * mdCamProjMatrix(1, 1);
                // 计算与真实观测之间的误差
                const cv::KeyPoint &kpUn = mCurrentFrame.mvObjKeysUn[nInCurrentFrameOrder][j];
                Eigen::Matrix<double, 2, 1> obs;
                obs<<kpUn.pt.x, kpUn.pt.y;
                const float invSigma2 = mCurrentFrame.mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix2d Info = Eigen::Matrix2d::Identity()*invSigma2;
                Eigen::Vector2d error;
                error = obs - Zp;
                //cout<<"单目: "<<error.dot(Info * error);
                monoDynaVal.push_back(error.dot(Info * error));
                monoPointNum++;

            }
            else{
                Eigen::Vector3d Zp;
                Zp[0] = mdCamProjMatrix(0,2) + Pc[0] * invz * mdCamProjMatrix(0, 0);
                Zp[1] = mdCamProjMatrix(1,2) + Pc[1] * invz * mdCamProjMatrix(1, 1);
                Zp[2] = Zp[0] - mbf * invz;
                // 计算与真实观测之间的误差
                const cv::KeyPoint &kpUn = mCurrentFrame.mvObjKeysUn[nInCurrentFrameOrder][j];
                const float &kp_ur = mCurrentFrame.mvuObjKeysRight[nInCurrentFrameOrder][j];
                Eigen::Matrix<double, 3, 1> obs;
                obs<<kpUn.pt.x, kpUn.pt.y, kp_ur;
                const float   invSigma2 = mCurrentFrame.mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                Eigen::Vector3d error;
                error = obs - Zp;
                //cout<<"双目: "<<error.dot(Info * error);
                stereoDynaVal.push_back(error.dot(Info * error));
                stereoPointNum++;
            }
        }
        if (monoPointNum>=5)
        {   // outlier removal
            sort(monoDynaVal.begin(), monoDynaVal.end());

            double median_monoDynaVal = monoDynaVal[int(monoDynaVal.size() / 2 + 0.5)];
            for (auto it = monoDynaVal.begin(); it != monoDynaVal.end();) {
                if ((*it) > 5 * median_monoDynaVal) {
                    //cout << "  " << (*it) << "/" << median_monoDynaVal;
                    monoDynaVal.erase(it);
                    monoPointNum--;
                } else it++;
            }
            //cout << endl;
            monoDynaValAvg = accumulate(monoDynaVal.begin(), monoDynaVal.end(), 0.0) / monoPointNum;
        }
        if (stereoPointNum>=5)
        {   // outlier removal
            sort(stereoDynaVal.begin(), stereoDynaVal.end());
            double median_stereoDynaVal = stereoDynaVal[int(stereoDynaVal.size() / 2 + 0.5)];

            for (auto it = stereoDynaVal.begin(); it != stereoDynaVal.end();) {
                if ((*it) > 5 * median_stereoDynaVal) {
                    //cout << "  " << (*it) << "/" << median_stereoDynaVal;
                    stereoDynaVal.erase(it);
                    stereoPointNum--;
                } else it++;
            }
            //cout << endl;
            stereoDynaValAvg = accumulate(stereoDynaVal.begin(), stereoDynaVal.end(), 0.0) / stereoPointNum;
        }
        //auto bbox = cDO->mrectBBox.width * cDO->mrectBBox.height;
        cDO->SetDynamicFlag(monoDynaValAvg, stereoDynaValAvg);
        if (monoDynaValAvg >0 || stereoDynaValAvg >0) {
            auto df = cDO->GetDynamicFlag();
            pMO->DynamicDetection(df);
            pMO->SetDynamicFlag(df);
        } else
        {
            cDO->SetDynamicFlag(pMO->GetDynamicFlag());
        }
        //cout<<RED<<"目标 "<<cDO->mnObjectID<<", 动态性 "<<cDO->GetDynamicFlag()<<" 是否Recover特征点 "<<cDO->IsRecovered<<" 单目点数 "<< monoPointNum <<"  单目平均动态 "<<monoDynaValAvg<<" 双目点数 "<< stereoPointNum <<" 双目平均动态 "<<stereoDynaValAvg<<endl<<endl;
        //cout<<WHITE;
    }
    //检查一下该帧是否有新目标被检测为静态
    StaticPointRecoveryFromObj();
}

void Tracking::StaticPointRecoveryFromObj(){
    // Static Points
    auto &vKeys = mCurrentFrame.mvKeys;
    auto &vKeysUn = mCurrentFrame.mvKeysUn;
    auto &vuRight = mCurrentFrame.mvuRight;
    auto &vDepth = mCurrentFrame.mvDepth;
    auto &Descriptors = mCurrentFrame.mDescriptors;
    auto &N = mCurrentFrame.N;
    auto &vpMapPoints = mCurrentFrame.mvpMapPoints;
    auto &vbOutlier = mCurrentFrame.mvbOutlier;

    // Object Points
    auto &vDetectionObjects = mCurrentFrame.mvDetectionObjects;
    auto &vObjKeys = mCurrentFrame.mvOriKeys;
    auto &vObjKeysUn = mCurrentFrame.mvOriKeysUn;
    auto &vuObjKeysRight = mCurrentFrame.mvuOriRight;
    auto &vObjPointDepth = mCurrentFrame.mvOriDepth;
    auto &vObjPointsDescriptors = mCurrentFrame.mOriDescriptors;
    for (int i = 0; i < vDetectionObjects.size(); ++i) {
        DetectionObject* cDO = vDetectionObjects[i];
        if (!cDO)
            continue;
        MapObject* pMO = cDO->GetMapObject();
        if (pMO){
            //bool DFixed = pMO->mbDynamicChanged;
            // 如果该目标动静态属性已固定且为静态
            if (pMO->GetDynamicFlag()==false){
                if (!cDO->IsRecovered){
                    for (int j = 0; j < vObjKeys[i].size(); ++j) {
                        vKeys.push_back(vObjKeys[i][j]);
                        vKeysUn.push_back(vObjKeysUn[i][j]);
                        vuRight.push_back(vuObjKeysRight[i][j]);
                        vDepth.push_back(vObjPointDepth[i][j]);
                        Descriptors.push_back(vObjPointsDescriptors[i].row(j));

                        vpMapPoints.push_back(static_cast<MapPoint*>(NULL));
                        vbOutlier.push_back(false);
                    }
                    // 防止重复Recover
                    cDO->IsRecovered = vObjKeys[i].size();
                }

            }
        }

    }
    //cout << "Before Recovery: " << mCurrentFrame.N ;
    N = vKeys.size();
    //cout << ", After Recovery: " << mCurrentFrame.N << endl;
    // FIXME 如果不修改，初始化mGrid，两次执行后匹配的特征点特别少，还没想通是为啥。
    //  只在本函数中执行不在Frame中执行的话，初始化的时候分配到Grid的特征点应该就没法用了。
    mCurrentFrame.AssignFeaturesToGrid();
}

void Tracking::CheckReplacedMapObjectPointsInLastFrame()
{
    unordered_set<int> hashTmp;
    for(size_t n=0; n<mLastFrame.mvMapObjects.size(); n++)
    {
        for(size_t i =0; i<mLastFrame.mvpMapObjectPoints[n].size(); i++)
        {
            MapObjectPoint* pMP = mLastFrame.mvpMapObjectPoints[n][i];
            if (pMP)
            {
                MapObjectPoint *pRep = pMP->GetReplaced();
                if(pRep)
                {
                    if (hashTmp.count(pRep->mnId))
                        mLastFrame.mvpMapObjectPoints[n][i] = static_cast<MapObjectPoint *>(NULL); // static_cast 强制类型转换
                    else {
                        mLastFrame.mvpMapObjectPoints[n][i] = pRep;
                        hashTmp.insert(pRep->mnId);
                    }
                }
                else if (hashTmp.count(pMP->mnId)) {
                    mLastFrame.mvpMapObjectPoints[n][i] = static_cast<MapObjectPoint *>(NULL);
                }
                else{
                    hashTmp.insert(pMP->mnId);
                }
            }
        }
    }
}

/// 将上一帧的3D 动态mappoint通过光流跟踪传递到当前帧的mvpMapPointsdynamic
void Tracking::TrackLastFrameObjectPoint(const bool &bUseTruthObjPoseFlag)
{
    cout<<endl;
    //cout<<YELLOW<<"目标点跟踪上一帧开始:"<<endl;
    cout<<WHITE;

    // 1. 如果是双目或者RGBD， 更新上一帧的动态landmarks
    vector<pair<size_t, size_t>> vInLastFrameTrackedObjOrders = mCurrentFrame.mvInLastFrameTrackedObjOrders;
    //cout<<"临时三角化上一帧动态点: "<<endl;
    for(size_t n=0; n<vInLastFrameTrackedObjOrders.size(); n++)
    {
        size_t nLastOrder = vInLastFrameTrackedObjOrders[n].first;
        MapObject* pMO = mLastFrame.mvMapObjects[nLastOrder];
        DetectionObject* cCuboidTemp = mLastFrame.mvDetectionObjects[nLastOrder];
        if(pMO->mnLastKeyFrameId == mLastFrame.mnId)// 如果上一帧对于该目标恰好是关键帧则返回
            continue;
        if(pMO->mnFirstObservationFrameId == mLastFrame.mnId) // 该目标是上
        {
            if(pMO->mVirtualVelocity != Vector6d::Zero())
                assert(0);
            continue;
        }

        int nPoints = 0;// 三角化
        int nTotalPoints = 0;
        vector<pair<float, int>> vDepthIdx;
        for(size_t i=0; i<mLastFrame.mvObjPointDepth[nLastOrder].size();i++)
        {
            float z = mLastFrame.mvObjPointDepth[nLastOrder][i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }
        if(vDepthIdx.empty())
            continue;
        sort(vDepthIdx.begin(),vDepthIdx.end());
        for(size_t j=0; j<vDepthIdx.size();j++)
        {
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;
            MapObjectPoint* pMP = mLastFrame.mvpMapObjectPoints[nLastOrder][i];
            if(!pMP)
                bCreateNew = true;
            else if((EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4) && pMP->Observations()<1)
            {
                bCreateNew = true;// 目标点的关键帧观测小于 1
            }
            if(bCreateNew)
            {
                g2o::ObjectState gcCuboidTmp;
                cv::Mat map_to_obj, x3Dc;
                x3Dc = mLastFrame.UnprojectStereodynamic(nLastOrder, i, false);// 得到在相机系位置
                if(bUseTruthObjPoseFlag)
                {
                    gcCuboidTmp = cCuboidTemp->mTruthPosInCameraFrame;
                    map_to_obj = Converter::toCvMat(gcCuboidTmp.pose.inverse().map(Converter::toVector3d(x3Dc)));
                }
                else{
                    gcCuboidTmp = pMO->GetCFInFrameObjState(mLastFrame.mnId); // Tco
                    map_to_obj = Converter::toCvMat(gcCuboidTmp.pose.inverse().map(Converter::toVector3d(x3Dc))); // Poj = Tco.inverse() * Pcj
                }
                Eigen::Vector3f feature_in_object = Converter::toVector3f(map_to_obj);// 转成向量好判断
                float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: gcCuboidTmp.scale.norm();
                if (feature_in_object.norm() > fMaxDis)
                {
                    continue;
                }
                MapObjectPoint *pNewMP = new MapObjectPoint(pMO, map_to_obj, x3Dc, &mLastFrame, nLastOrder, i);// 建立动态特征点（要不要与object相互绑定）
                mLastFrame.mvpMapObjectPoints[nLastOrder][i] = pNewMP;// 加入上一帧
                pMO->mlpTemporalPoints.push_back(pNewMP); // 放到临时容器中, 不需要放入cuboidtemp中吗
                cCuboidTemp->AddMapObjectPoint(i, pNewMP); // 应该是不需要添加观测什么的
                nPoints++;
                nTotalPoints++;
            }
            else
            {
                nTotalPoints++;
            }
            if(vDepthIdx[j].first>2*mThDepth)
                break;
        }
        //cout<<"目标: "<< pMO->mnTruthID<<" 三角化目标点数： "<<nTotalPoints<<endl;
    }

    // 2. 跟踪上一帧的目标点
    vector<size_t> vnNeedToBeOptimized;
    vnNeedToBeOptimized.reserve(vInLastFrameTrackedObjOrders.size());
    for(size_t n=0; n<vInLastFrameTrackedObjOrders.size(); n++)
    {
        size_t nLastOrder = vInLastFrameTrackedObjOrders[n].first;
        size_t nCurrentOrder = vInLastFrameTrackedObjOrders[n].second;
        DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[n];
        ORBmatcher matcher(0.9,true);

        int temp = 0;
        switch(temp)
        {
            case 0: // 直接暴力匹配
            {
                vector<MapObjectPoint* >vpMapObjectPointMatches;
                int nmatches = matcher.SearchByBruceMatching(mLastFrame, mCurrentFrame, nLastOrder, nCurrentOrder, vpMapObjectPointMatches);
                mCurrentFrame.mvpMapObjectPoints[nCurrentOrder] = vpMapObjectPointMatches;
                //int nmatches = matcher.SearchByBruceMatchingWithGMS(mLastFrame, mCurrentFrame, nLastOrder, nCurrentOrder);
                if(nmatches < 10)
                    pDet->mbTrackOK = false;
                else
                {
                    vnNeedToBeOptimized.push_back(nCurrentOrder); // 没被优化的目标肯定是跟踪状态是false
                }
                break;
            }

            case 1: // 离线光流跟踪
            {
                int nmatches = matcher.SearchByOfflineOpticalFlowTracking(mLastFrame, mCurrentFrame, nLastOrder, nCurrentOrder);
                if(nmatches < 10)
                    pDet->mbTrackOK = false;
                else
                {
                    pDet->mbTrackOK = true;
                    vnNeedToBeOptimized.push_back(nCurrentOrder);
                }

                break;
            }

            default:
                assert(0);
        }
    }

    if(vnNeedToBeOptimized.size() == 0)
        return;

    if(1)
    {
        Optimizer::CFSE3ObjStateOptimization(&mCurrentFrame, vnNeedToBeOptimized, false);// BA, 只优化有足够特征点的
        for(size_t n=0; n<vnNeedToBeOptimized.size(); n++) // 遍历被优化后的目标
        {
            int t1=0, t2 =0, t3=0, t4 =0;
            size_t i = vnNeedToBeOptimized[n];
            DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[i];
            int nmatchesMap = 0;
            for(std::size_t m=0; m< mCurrentFrame.mvpMapObjectPoints[i].size(); m++)
            {
                if(mCurrentFrame.mvpMapObjectPoints[i][m]==NULL)
                    continue;
                t1++;
                if(mCurrentFrame.mvbObjKeysOutlier[i][m]) // 如果该点是外点
                {
                    MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[i][m];
                    mCurrentFrame.mvpMapObjectPoints[i][m]=static_cast<MapObjectPoint*>(NULL);
                    mCurrentFrame.mvbObjKeysOutlier[i][m]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    t2++;
                }
                else if(mCurrentFrame.mvpMapObjectPoints[i][m]->Observations()>0)
                {
                    t3++;
                    nmatchesMap++;
                }
                else{
                    t4++;
                }
            }


//            if(EnSLOTMode == 3) // 自动驾驶模式是没有目标关键帧的
//            {
//                if(t3+t4<10) //  优化完成, 需要进一步看看这些目标的跟踪状态是否跟踪成功
//                    pDet->mbTrackOK = false;
//                else
//                    pDet->mbTrackOK = true;
//                cout<<"目标"<<pDet->mnObjectID<<" 跟踪上一帧内点数量:"<<t3+t4<<" tracking状态: "<<pDet->mbTrackOK<<endl;
//            }
//            else
            {
                if(nmatchesMap<10) //  优化完成, 需要进一步看看这些目标的跟踪状态是否跟踪成功
                    pDet->mbTrackOK = false;
                else
                    pDet->mbTrackOK = true;
                //cout<<"目标"<<pDet->mnObjectID<<" 跟踪上一帧内点数量:"<<nmatchesMap<<" tracking状态: "<<pDet->mbTrackOK<<endl;
            }
        }
    }

}

void Tracking::TrackObjectLocalMap()
{
    cout<<endl;
    //cout<<YELLOW<<"目标点跟踪局部地图开始:"<<WHITE<<endl;

    vector<size_t> vnNeedToBeOptimized;
    vnNeedToBeOptimized.reserve(mCurrentFrame.mvTotalTrackedObjOrders.size());
    for(size_t n=0; n<mCurrentFrame.mvTotalTrackedObjOrders.size(); n++)
    {
        size_t nOrder = mCurrentFrame.mvTotalTrackedObjOrders[n];
        DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[nOrder];
//        if(pDet->mbTrackOK == false) // 只有前面跟踪上一帧或跟踪参考帧成功了的目标(此时目标点数目不会少)才会跟踪局部地图
//            continue;
        UpdateObjectLocalKeyFrames(nOrder); // 1. 更新局部关键帧
        UpdateObjectLocalPoints(nOrder); // 2. 更新局部地图点
        int nTracks = SearchObjectLocalPoints(nOrder);
        //cout<<"跟踪局部地图3D点数: "<<nTracks<<endl;

        vnNeedToBeOptimized.push_back(nOrder);
    }
    if(vnNeedToBeOptimized.size() == 0)
        return;

    if(1)
    {
        Optimizer::CFSE3ObjStateOptimization(&mCurrentFrame, vnNeedToBeOptimized, false);
        for(size_t n=0; n<vnNeedToBeOptimized.size(); n++)// 优化完剔除外点
        {
            size_t nOrder = vnNeedToBeOptimized[n];
            DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[nOrder];
            int t1=0, t2=0, t3=0, t4=0;
            for(size_t m=0; m<mCurrentFrame.mvpMapObjectPoints[nOrder].size(); m++)
            {
                MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[nOrder][m];
                if(pMP==NULL)
                    continue;
                t1++;
                if(mCurrentFrame.mvbObjKeysOutlier[nOrder][m] == false)// 如果不是outlier
                {
                    t2++;
                    pMP->IncreaseFound();
                    if(pMP->Observations()>0) // 需要是关键帧里的点
                    {
                        t3++;
                        pDet->mnMatchesInliers++;
                    }

                }
                else{
                    t4++;
                    mCurrentFrame.mvpMapObjectPoints[nOrder][m] = static_cast<MapObjectPoint*>(NULL); // 注意只是从当前帧删除, 并未将其outlier标志位置为false
                }
            }

            if(pDet->mnMatchesInliers>10) // 优化完成需要进一步判断目标的跟踪状态
                pDet->mbTrackOK = true;
            else
                pDet->mbTrackOK = false;
            //cout<<"目标"<<pDet->mnObjectID<<" 跟踪局部地图内点数量:"<<pDet->mnMatchesInliers<<" tracking状态: "<<pDet->mbTrackOK<<" t1:"<<t1<<" t2:"<<t2<<" t3:"<<t3<<" t4: "<<t4<<endl;
        }
    }
}

int Tracking::SearchObjectLocalPoints(const size_t &n)
{

    for(vector<MapObjectPoint*>::iterator vit=mCurrentFrame.mvpMapObjectPoints[n].begin(), vend=mCurrentFrame.mvpMapObjectPoints[n].end(); vit!=vend; vit++)
    {
        MapObjectPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapObjectPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;
    int nRealMatch=0;
    MapObject* pMO = mCurrentFrame.mvMapObjects[n];
    for(vector<MapObjectPoint*>::iterator vit=pMO->mvpLocalMapObjectPoints.begin(), vend=pMO->mvpLocalMapObjectPoints.end(); vit!=vend; vit++)
    {
        MapObjectPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) // 跳过标记点
            continue;
        if(pMP->isBad())
            continue;
        if(mCurrentFrame.isInFrustum(pMP, n, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }
    //cout<<" 符合投影要求3D点数: "<<nToMatch<<" ";

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        nRealMatch = matcher.SearchByProjection(mCurrentFrame, n, pMO->mvpLocalMapObjectPoints,th); // 投影的方式来找点
    }

    return nRealMatch;
}

void Tracking::UpdateObjectLocalKeyFrames(const size_t &nInCurrentFrameOrder) // 更新局部关键帧
{
    MapObject* pMO = mCurrentFrame.mvMapObjects[nInCurrentFrameOrder];
    if(pMO==NULL)
        return;
    map<ObjectKeyFrame*,int> keyframeCounter;
    for(size_t i=0; i<mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder].size(); i++) // 得到当前帧的共视关键帧, 并统计权重
    {
        if(mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder][i])
        {
            MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder][i];
            if(!pMP->isBad())
            {
                const map<ObjectKeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<ObjectKeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder][i]=NULL;
            }
        }
    }
    if(keyframeCounter.empty())
        return;
    int max=0;
    ObjectKeyFrame* pKFmax= static_cast<ObjectKeyFrame*>(NULL);
    pMO->mvLocalObjectKeyFrames.clear();
    pMO->mvLocalObjectKeyFrames.reserve(3*keyframeCounter.size());
    // 共视关键帧加入局部地图
    for(map<ObjectKeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        ObjectKeyFrame* pKF = it->first;
        if(pKF->isBad())
            continue;
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }
        pMO->mvLocalObjectKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }
    // 共视关键帧的共视关键帧, 子关键帧, 父关键帧, 也加入局部地图
    for(vector<ObjectKeyFrame*>::const_iterator itKF=pMO->mvLocalObjectKeyFrames.begin(), itEndKF=pMO->mvLocalObjectKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        if(pMO->mvLocalObjectKeyFrames.size()>80)// Limit the number of keyframes
            break;
        ObjectKeyFrame* pKF = *itKF;
        const vector<ObjectKeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<ObjectKeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            ObjectKeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    pMO->mvLocalObjectKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        const set<ObjectKeyFrame*> spChilds = pKF->GetChilds();
        for(set<ObjectKeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            ObjectKeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    pMO->mvLocalObjectKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        ObjectKeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                pMO->mvLocalObjectKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }
    if(pKFmax)
    {
        // 直接对目标设置其参考关键帧
        MapObject* pMO= mCurrentFrame.mvMapObjects[nInCurrentFrameOrder];
        if(pMO==NULL)
            return;
        pMO->mpReferenceObjKF = pKFmax;
    }
}

void Tracking::UpdateObjectLocalPoints(const size_t &nOrder)
{

    MapObject* pMO = mCurrentFrame.mvMapObjects[nOrder];
    if(pMO==NULL)
        return;

    pMO->mvpLocalMapObjectPoints.clear();

    std::unordered_set<int> hash;
    for(vector<ObjectKeyFrame*>::const_iterator itKF=pMO->mvLocalObjectKeyFrames.begin(), itEndKF=pMO->mvLocalObjectKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        ObjectKeyFrame* pKF = *itKF;
        const vector<MapObjectPoint*> vpMPs = pKF->GetMapObjectPointMatches();
        for(vector<MapObjectPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapObjectPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId) // 防止重复添加
                continue;

            if(!pMP->isBad())
            {
                pMO->mvpLocalMapObjectPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;

                if(!hash.count(pMP->mnId)) // 这个地方是为什么？
                    hash.insert(pMP->mnId);
                else
                    assert(0);
            }
        }
    }
   //cout<<"目标"<<pMO->mnTruthID<<"局部关键帧数:"<<pMO->mvLocalObjectKeyFrames.size()<<" 局部3D点数: "<<pMO->mvpLocalMapObjectPoints.size()<<" ";

   return;
}

bool Tracking::NeedNewObjectKeyFrame(const size_t &nOrder)
{
    bool bNeedInsert = false;
    DetectionObject* pDetObj = mCurrentFrame.mvDetectionObjects[nOrder];
    MapObject* pMO = mCurrentFrame.mvMapObjects[nOrder];
    if(pMO == NULL || pDetObj==NULL)
        assert(0);
    //const bool bFlag1 = mCurrentFrame.mnId>=(pMO->mnLastKeyFrameId+mMaxFrames);// 条件1: 很长时间没有为这个object加入关键帧
    const bool bFlag1 = mCurrentFrame.mnId>=(pMO->mnLastKeyFrameId+6);// 条件1: 很长时间没有为这个object加入关键帧
    int nNonTrackedClose = 0;// 统计该目标当前帧已有的地图点数量
    int nTrackedClose= 0;// 与 还可以生成的地图点数量
    for(size_t i =0; i<mCurrentFrame.mvObjPointDepth[nOrder].size(); i++)
    {
        if(mCurrentFrame.mvObjPointDepth[nOrder][i]>0 && mCurrentFrame.mvObjPointDepth[nOrder][i]<mThDepth)
        {
            if(mCurrentFrame.mvpMapObjectPoints[nOrder][i] && !mCurrentFrame.mvbObjKeysOutlier[nOrder][i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }
    bool bNeedToInsertFlag =(nTrackedClose<20)&&(nNonTrackedClose>20);
    int nRefMatches = mpReferenceKF->TrackedMapPoints(2); //得到参考关键帧中该目标的观测次数超过2的目标点数
    const bool bFlag3 = (pDetObj->mnMatchesInliers < nRefMatches*0.1 || bNeedToInsertFlag);// 条件3: 该object跟踪效果不佳, 所有的inliers
    const bool bFlag4 = pDetObj->mnMatchesInliers>20;// 条件4: 该目标在该帧的inliers数量要超过阈值
    if((bFlag1||bFlag3) && bFlag4)
    {
        pDetObj->mbNeedCreateNewOKFFlag = true;
        if(bNeedInsert == false)
            bNeedInsert = true;
    }
    return bNeedInsert;
}


void Tracking::CreateNewObjectKeyFrame(const size_t &nOrder)
{
    // 直接在这里新建关键帧
    MapObject* moObjectTmp = mCurrentFrame.mvMapObjects[nOrder];
    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[nOrder];

    if(moObjectTmp==NULL || pDet==NULL)
        assert(0);
    if(pDet->mbNeedCreateNewOKFFlag==false)
        assert(0);

    ObjectKeyFrame* pOKF = new ObjectKeyFrame(mCurrentFrame, nOrder, false); // 设置参考关键帧为当前关键帧, 当前帧的参考关键帧为当前关键帧
    moObjectTmp->mpReferenceObjKF = pOKF; // 设置目标的参考关键帧
    moObjectTmp->SetCFObjectKeyFrameObjState(pOKF, moObjectTmp->GetCFInFrameObjState(mCurrentFrame.mnId));// 设置目标关键帧状态
    //cout<<"OKF Frame ID: "<<pOKF->mnFrameId<<", KF ID: "<<pOKF->mnId<<", Object ID"<<moObjectTmp->mnTruthID<<endl;
    vector<pair<float,int> > vDepthIdx;// 双目三角化出来一些新的地图点
    size_t nObjKeysNum = mCurrentFrame.mvObjKeys[nOrder].size();
    vDepthIdx.reserve(nObjKeysNum);
    for(size_t i=0; i<nObjKeysNum; i++)
    {
        float z = mCurrentFrame.mvObjPointDepth[nOrder][i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }
    if(!vDepthIdx.empty())
    {
        sort(vDepthIdx.begin(),vDepthIdx.end());
        int nPoints = 0;
        for(size_t j=0; j<vDepthIdx.size();j++)
        {
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;
            MapObjectPoint* pMP = mCurrentFrame.mvpMapObjectPoints[nOrder][i];
            if(!pMP)
            {
                bCreateNew = true;
            }
            else if(pMP->Observations()<1)   // 该点被关键帧的观测次数少于1
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapObjectPoints[nOrder][i] = static_cast<MapObjectPoint*>(NULL);
            }

            if(bCreateNew)
            {

                cv::Mat Pcj = mCurrentFrame.UnprojectStereodynamic(nOrder, i, false);
                g2o::ObjectState ObjState = moObjectTmp->GetCFInFrameObjState(mCurrentFrame.mnId);
                g2o::SE3Quat Tco = ObjState.pose;
                Eigen::Vector3d Poj = Tco.inverse().map(Converter::toVector3d(Pcj));

                // 目标点位置超出了目标尺度范围
                float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: ObjState.scale.norm();
                if(Poj.norm() > fMaxDis)
                    continue;

                MapObjectPoint* pNewMP = new MapObjectPoint(moObjectTmp, Converter::toCvMat(Poj), Pcj, pOKF);// 关键帧建立目标地图点
                if (!pNewMP->mpRefObjKF) assert(0);
                pNewMP->AddObservation(pOKF,i);// 该点增加关键帧观测
                pOKF->AddMapObjectPoint(pNewMP, i);// 该关键帧增加该地图点
                pNewMP->ComputeDistinctiveDescriptors();// 该点计算描述子
                pNewMP->UpdateNormalAndDepth();// 该点更新深度
                moObjectTmp->AddMapObjectPoint(pNewMP);

                if(mCurrentFrame.mvbObjKeysOutlier[nOrder][i] == false)// 如果该点对应的是outlier, 则不把该点放入当前帧
                {
                    mCurrentFrame.mvpMapObjectPoints[nOrder][i] = pNewMP;// 该点加入当前帧
                }
                // 该点加入地图
                nPoints++;
            }
            else
            {
                nPoints++;
            }
            if(vDepthIdx[j].first>2*mThDepth && nPoints>100)
                break;
        }
    }
    //创建关键帧时，新三角化的点比较准确，这时候有点漂移的框就拟合不上
    if(EnSLOTMode == 2 || EnSLOTMode ==3 ||EnSLOTMode == 4) // 只有目标跟踪模式才插到局部建图线程去
    {
        mpObjectLocalMapper->InsertOneObjKeyFrame(pOKF);
    }

    moObjectTmp->mnLastKeyFrameId = mCurrentFrame.mnId;

}

/// 双目初始化函数： 得到初始3D 静态 mappoints， 初始3D objects， 初始3D 动态 mappoints
void Tracking::StereoInitialization()
{
    /// 双目初始化的条件是检测到的特征点数量超过500
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        /// 1. 设定初始位姿
        if(EbSetWorldFrameOnGroundFlag)
        {
            /// 1.1 可以自己设定世界系为GroundToInit
            mCurrentFrame.SetPose(mTwc0);
        } else{
            /// 1.2 世界系与首帧camera系重合
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        }

        /// 2.  2D objects建立3D objects
        if(mCurrentFrame.mnDetObj!=0 && 0) // 第一帧没有track
        {
            TrackMapObject();
        }


        /// 3. 将当前关键帧构建为初始关键帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        cout<<"!!!Stereo initialization: create the first keyframe: "<<pKFini->mnId<<"          total_id:    "<<mCurrentFrame.mnId<<endl;

        /// 5. 在局部地图中添加该初始关键帧
        mpMap->AddKeyFrame(pKFini);


        /// 6. 构造3D 静态 mappoints
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                /// 6.1 通过反投影得到该特征点的3D坐标
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                //cout<<"双目初始化动态点： "<<x3D<<endl;

                /// 6.2 将3D点构造为mappoint
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                /// 6.3 为该mappoint添加属性:
                /// 6.3.1 观测到该mappoint的关键帧
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                /// 6.3.2 计算描述子
                pNewMP->ComputeDistinctiveDescriptors();
                /// 6.3.3 计算平均观测方向和深度范围
                pNewMP->UpdateNormalAndDepth();
                /// 6.3.4 在地图中添加该mappoint
                mpMap->AddMapPoint(pNewMP);
                /// 6.3.5 为当前帧的添加该mappoint
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " static points" << "   and   "<<mpMap->MapObjectPointsInMap()<<"   dynamic points"<<endl;


        /// 8. 在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        /// 9. 更新mLastFrame
        mLastFrame = Frame(mCurrentFrame);

        /// 10. 更新mnLastKeyFrameId, mpLastKeyFrame
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        /// 11. 将当前关键帧加入局部关键帧集mvpLocalKeyFrames
        mvpLocalKeyFrames.push_back(pKFini);
        /// 12. 将地图中的静态点, 和动态点分别加入局部静态点集, 局部动态点集
        mvpLocalMapPoints=mpMap->GetAllMapPoints();

        /// 13. 设置mpmpReferenceKF为当前关键帧
        mpReferenceKF = pKFini;
        /// 14. 设置当前帧的参考关键帧为当前关键帧
        mCurrentFrame.mpReferenceKF = pKFini;

        /// 15. 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        /// 16. ??
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        /// 17. For mpMapDrawer
        mpMapDrawer->SetCurrentCameraPoseAndId(mCurrentFrame.mTcw, mCurrentFrame.mnId);

        mState=OK;
    }

}

/// 检查上一帧中的MapPoints是否被替换
/// Local Mapping线程可能会将关键帧中某些MapPoints进行替换，
/// 由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            /// 2. 得到该landmark的替换点, 然后将替换点 替换掉上一帧中的landmark
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/// 对参考关键帧的MapPoints进行跟踪
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    /// 1. 将当前帧的描述子转化为BoW向量
    cout<<"TrackReferenceKeyFrame"<<endl;
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    /// 2. 建立ORBmatcher类对象
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    /// 3. 静态点: 当前帧与参考关键帧静态特征点匹配, 匹配点存在vpMapPointMatches
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    //cout<<"静态跟踪参考帧3D点数: "<<nmatches<<endl;

    /// 5. 如果静态点匹配数量小于15,则返回
    if(nmatches<15)
        return false;

    /// 6. 将vpMapPointMatches存入当前帧
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    /// 7. 设置当前帧的位姿优化初值(上一帧的位姿)
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    //cout<<mCurrentFrame.mTcw<<endl;
    /// 8. 优化当前帧的位姿, 利用3D-2D的重投影约束
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    /// 9. 剔除优化后的outlier匹配点（MapPoints）
    int nmatchesMap = 0;
    /// 9.1 遍历该帧的所有静态特征点
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            /// 9.2 如果它的标志位为1, 则进行剔除
            if(mCurrentFrame.mvbOutlier[i])
            {
                /// 9.2.1 取出该静态landmark
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                /// 9.2.2 剔除
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                /// 9.2.3 标志位置为false
                mCurrentFrame.mvbOutlier[i]=false;
                /// 9.2.4 ???TODO
                pMP->mbTrackInView = false;
                /// 9.2.5 ???TODO
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }/// 9.3 如果它的标志位为0, 且被观测次数大于0, 则nmatchesMap++TODO??
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    /// 10. 如果nmatchesMap>=10, 返回TRUE
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking) // 上一帧是关键帧就不做,因为关键帧本来会生成地图点
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }
    if(vDepthIdx.empty())
        return;
    sort(vDepthIdx.begin(),vDepthIdx.end());
    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;
        bool bCreateNew = false;
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }
        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }
        if(vDepthIdx[j].first>2*mThDepth && nPoints>100) // 这个条件的含义是 插入所有满足距离条件的点, 可以大于100. 如果满足距离条件的点不足100, 那就插入100个最近的点. 如果所有的点数都不足100, 则全部插入
            break;
    }
}

/// 根据匀速度模型对上一帧的MapPoints进行跟踪
bool Tracking::TrackWithMotionModel()
{
    /// 1. 初始化ORBmatcher类对象
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    /// 2. 对于双目或rgbd摄像头，根据深度值为上一帧生成新的MapPoints
    UpdateLastFrame();

    /// 3. 设置当前帧位姿的优化初值(匀速模型)
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    /// 4. 初始化当前帧的静态地图点集: mvpMapPoints, 为NULL
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    //th 阈值
    /// 5. 当前帧与上一帧的地图点进行匹配, 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    /// 6. 如果跟踪的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    //cout<<"静态跟踪上一帧3D点数: "<<nmatches<<endl;

    /// 9. 如果静态点跟踪数量很少, 则返回false
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    /// 10. 优化相机位姿(利用静态点)
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    /// 11. 剔除outlier的mvpMapPoints: 仅仅对静态点
    int nmatchesMap = 0;
    /// 11.1 遍历当前帧的landmark
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            /// 11.2 判断其outlier标志位是否为true
            if(mCurrentFrame.mvbOutlier[i])
            {
                /// 11.3 取出该mappoint
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                /// 11.4 从当前帧中删除
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                /// 11.5 设置该mappoint的属性???TODO
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            /// 11.2 (标志位)如果为false, 则判断其观测次数是否大于0
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }


    // 静态点的重投影误差
    if(0)
    {
        for(size_t i=0; i<mCurrentFrame.mvpMapPoints.size(); i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP == NULL) // 只有可能是跟踪上一帧来的
                continue;
            cv::Mat wP = pMP->GetWorldPos();
            Eigen::Vector3d eigwP, eigcP;
            eigwP[0] = wP.at<float>(0);
            eigwP[1] = wP.at<float>(1);
            eigwP[2] = wP.at<float>(2);
            eigcP = mCurrentFrame.mSETcw * eigwP;
            const double invz = 1.0/eigcP[2];

            // 这个点是否有右目匹配点
            if(mCurrentFrame.mvuRight[i] < 0) // 说明是单目
            {
                Eigen::Vector2d Zp;
                Zp[0] = mdCamProjMatrix(0,2) + eigcP[0] * invz * mdCamProjMatrix(0,0);
                Zp[1] = mdCamProjMatrix(1,2) + eigcP[1] * invz * mdCamProjMatrix(1,1);
                // 计算与真实观测之间的误差
                const cv::KeyPoint &kpUn = mCurrentFrame.mvKeys[i];
                Eigen::Matrix<double, 2, 1> obs;
                obs<<kpUn.pt.x, kpUn.pt.y;
                const float invSigma2 = mCurrentFrame.mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix2d Info = Eigen::Matrix2d::Identity()*invSigma2;
                Eigen::Vector2d error;
                error = obs - Zp;
                cout<<"静态单目点重投影误差:"<<error.dot(Info*error)<<endl; // 肯定小于5.991, 因为当作outlier去除了
            }
            else{
                Eigen::Vector3d Zp;
                Zp[0] = mdCamProjMatrix(0,2) + eigcP[0] * invz * mdCamProjMatrix(0,0);
                Zp[1] = mdCamProjMatrix(1,2) + eigcP[1] * invz * mdCamProjMatrix(1,1);
                Zp[2] = Zp[0] - mbf * invz;
                // 计算与真实观测之间的误差
                const cv::KeyPoint &kpUn = mCurrentFrame.mvKeys[i];
                const float &kp_ur = mCurrentFrame.mvuRight[i];
                Eigen::Matrix<double, 3, 1> obs;
                obs<<kpUn.pt.x, kpUn.pt.y, kp_ur;
                const float invSigma2 = mCurrentFrame.mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                Eigen::Vector3d error;
                error = obs - Zp;
                cout<<"静态双目点重投影误差:"<<error.dot(Info*error)<<endl; // 肯定是小于7.815, 因为当作outlier去除了
            }
        }
    }



    /// 12. 如果设置成onlyTracking模式 (要求的nmatches更多),??
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }
    return nmatchesMap>=10;
}

/// 对Local Map的MapPoints进行跟踪
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    /// 1. 更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
    UpdateLocalMap();
    /// 2. 在局部地图中查找与当前帧匹配的MapPoints
    SearchLocalPoints();

    // Optimize Pose
    /// 4. 更新局部所有MapPoints后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    /// 5. 更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            /// 5.1 遍历所有的静态landmark, 判断其outlier标志位
            if(!mCurrentFrame.mvbOutlier[i])
            {
                /// 5.1.1 若不是outlier, 则增加一次观测次数
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                /// 5.1.2 判断目前模式
                if(!mbOnlyTracking)
                {
                    // 该MapPoint被其它关键帧观测到过
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    // 记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO) /// 5.1.1 若是outlier, 判断是否是双目, 是双目则直接从当前帧删除
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    /// 6. 决定是否跟踪成功, 才重定位不久同时inlier比较少,则false, 或则inlier太少返回false
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/// 判断是否需要插入关键帧: 判断当前普通帧是否为关键帧
bool Tracking::NeedNewKeyFrame()
{
    /// 1. 如果用户在界面上选择重定位，那么将不插入关键帧
    /// 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    /// 2. 如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) // 分别返回mbStopped 和 mbStopRequested信号
        return false;

    /// 3. ?????TODO
    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    /// 4. 判断是否距离上一次插入关键帧的时间太短
    /// mCurrentFrame.mnId是当前帧的ID, mnLastRelocFrameId是最近一次重定位帧的ID
    /// mMaxFrames等于图像输入的频率,如果关键帧比较少，则考虑插入关键帧
    /// 或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames) // 若距离上一次重定位不超过1s同时地图里有一定量的关键帧则跳过
        return false;

    // Tracked MapPoints in the reference keyframe
    /// 5. 得到参考关键帧跟踪到的MapPoints数量
    /// 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); // 参考关键帧中的(被关键帧观测次数不少于nMinObs)地图点数量

    // Local Mapping accept keyframes?
    /// 6. 查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames(); // 返回mbAcceptKeyFrames

    // Check how many "close" points are being tracked and how many could be potentially created.
    /// 7. 对于双目或RGBD摄像头, 当前帧已有的地图点数量nTrackedClose, 和可以产生但还没有跟上的地图点数nNonTrackedClose
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);// 已跟踪上的数量比较上, 还可以产生的比较多

    // Thresholds
    /// 8. 决策是否需要插入关键帧
    /// 8.1 设定inlier阈值, 和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;///关键帧只有一帧，那么插入关键帧的阈值设置很低

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    /// 超过1s没有插入关键帧
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    /// localMapper处于空闲状态
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    /// 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ; // inliers比较少, 或者可以进一步生成比较多地图点
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    /// 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);// c2?

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle) // 如果mbAcceptKeyFrames为true, 则这里为true
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA(); // 若mbAcceptKeyFrames为false, 则设置mbAbortBA为true, TODO 为什么不加锁?
            if(mSensor!=System::MONOCULAR)
            {
                /// 队列里不能阻塞太多关键帧
                /// tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                /// 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3) // 返回mlNewKeyFrames.size(), 双目的话, 队列里少于3个关键帧可以插入
                    return true;
                else
                    return false;
            }
            else // 如果是单目, mbAcceptKeyFrames为false, 直接不准插入
                return false;
        }
    }
    else
        return false;
}

/// 创建新的关键帧, 对于非单目的情况，同时创建新的MapPoints
void Tracking::CreateNewKeyFrame()
{
    // TODO
    if(!mpLocalMapper->SetNotStop(true)) // 如果mbStopped为真, 则这里直接返回false; 若mbStopped为假, 则mbNotStop=flag, 并且返回true
        return;

    // 将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    //cout<<endl<<BLUE<<"TODO:    created new keyframe!    "<<pKF->mnId<<"           total ID          "<<pKF->mnFrameId<<endl;
    cout<<WHITE<<endl;
    mpReferenceKF = pKF;// 设置tracking中的参考关键帧为当前关键帧；当前普通帧的参考关键帧为当前关键帧
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)// 对于双目或rgbd摄像头, 为当前帧生成新的mappoints；与UpdateLastFrame中的那一部分代码功能相同
    {

        mCurrentFrame.UpdatePoseMatrices();// 因为前面优化只是更新了Tcw, 这里根据Tcw再更新mRcw, mtcw和mRwc、mOw
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }
        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());// 对vDepthIdx进行排序: 从小到大按深度
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;// 遍历vDepthIdx中特征点id
                bool bCreateNew = false;// 初始化标志位: bCreateNew
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                {
                    bCreateNew = true;
                }
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }
                auto kp = mCurrentFrame.mvKeysUn[i];

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->mnDynamicFlag=true;
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);
                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }
                if(vDepthIdx[j].first>2*mThDepth && nPoints>100)
                    break;
            }
        }
        cout<<endl;
    }
    mpLocalMapper->InsertKeyFrame(pKF);// 将当前关键帧插入mpLocalMapper类的mlNewKeyFrames队列
    mpLocalMapper->SetNotStop(false); // 设置mbNotStop为false, 返回true TODO 有什么用???
    mnLastKeyFrameId = mCurrentFrame.mnId;// 更新mnLastKeyFrameId, mpLastKeyFrame
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL); // 这个是什么意思
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;
    int nmatch =0;
    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        nmatch++;
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        int nStaticTracks = matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        //cout<<"静态跟踪局部地图3D点数: "<<nStaticTracks<<endl;
    }
}


void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);//局部地图的mappoint, 画图时画成不同颜色

    // Update
    UpdateLocalKeyFrames();//更新局部关键帧, 这里不改
    UpdateLocalPoints();//更新局部地图点, 如果有动态点也更新局部动态地图点.
}

void Tracking::UpdateLocalPoints()//更新局部地图点, 如果有动态点也更新局部动态地图点.
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
    //cout<<"静态局部3D点数量: "<<mvpLocalMapPoints.size()<<endl;
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;
        KeyFrame* pKF = *itKF;
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
    //cout<<"静态局部关键帧数量: "<<mvpLocalKeyFrames.size()<<endl;
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
