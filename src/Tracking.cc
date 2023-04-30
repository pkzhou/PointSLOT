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

    EnSLOTMode = fSettings["SLOT.MODE"];
    if(EnSLOTMode == 1)
        EnDynaSLAMMode = fSettings["DynaSLAM.MODE"];

    EnOnlineDetectionMode = fSettings["Object.EnOnlineDetectionMode"];

    int tmp1 = fSettings["Object.EbManualSetPointMaxDistance"];
    EbManualSetPointMaxDistance = tmp1 > 0 ? 1:0;
    if(EbManualSetPointMaxDistance)
        EfInObjFramePointMaxDistance = fSettings["Object.EfInObjFramePointMaxDistance"];



    switch(EnSLOTMode)
    {
        case 0:
        {
            break;
        }
        case 1:
        {
            if(EnDynaSLAMMode == 0)
            {

                if(EnOnlineDetectionMode)
                {
                    YoloInit(fSettings);// YOLOdetector initialization
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
                            std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                            std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";

                            std::string virtualKittiCameraGroundthPoseFile = EstrDatasetFolder + "/extrinsic.txt";
                            ReadVirtualKittiCameraGT(virtualKittiCameraGroundthPoseFile);
                            break;
                        }
                        default:
                            assert(0);
                    }

                }
            }
            else
            {
                int tmp = fSettings["Yolo.active"];
                EbYoloActive = tmp > 0?true:false;
                if(EbYoloActive == true)
                {
                    YoloInit(fSettings);
                }
                mMultiTracker = new cv::MultiTracker();
                for(size_t i=0; i<1; i++)
                {
                    cv::Ptr<cv::Tracker> Tracker = cv::TrackerCSRT::create();
                    mvTrackers.push_back(Tracker);
                }
            }
            break;
        }

        case 2:
        {
            EnObjectCenter = fSettings["Viewer.ObjectCenter"];
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"];
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"];
            EbSetInitPositionByPoints = temp2 > 0? 1:0;

            EeigUniformObjScale(0) = fSettings["Object.Width.xc"];
            EeigUniformObjScale(1) = fSettings["Object.Height.yc"];
            EeigUniformObjScale(2) = fSettings["Object.Length.zc"];

            EeigInitPosition(0) = fSettings["Object.position.xc"];
            EeigInitPosition(1) = fSettings["Object.position.yc"];
            EeigInitPosition(2) = fSettings["Object.position.zc"];
            EeigInitRotation(1) = fSettings["Object.yaw.y"];
            if(EnOnlineDetectionMode)
            {
                mMultiTracker = new cv::MultiTracker();
                for(size_t i=0; i<1; i++)
                {
                    cv::Ptr<cv::Tracker> Tracker = cv::TrackerCSRT::create();
                    mvTrackers.push_back(Tracker);
                }
            }
            else
            {
                std::string objectFile = EstrDatasetFolder + "/object.txt";
                ReadMynteyeObjectInfo(objectFile);
            }
            break;
        }

        case 3:
        {
            EnObjectCenter = fSettings["Viewer.ObjectCenter"];
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"];
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"];
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
                            std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                            std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";
                            ReadVirtualKittiObjectInfo(virtualkittiobjectposefile, virtualkittibboxfile);
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

                    EeigUniformObjScale(0) = fSettings["Object.Width.xc"];
                    EeigUniformObjScale(1) = fSettings["Object.Height.yc"];
                    EeigUniformObjScale(2) = fSettings["Object.Length.zc"];
                    //EbSetInitPositionByPoints = true;
                    YoloInit(fSettings); // 1. YOLOdetector
                    std::string sort_engine_path_ = fSettings["DeepSort.weightsPath"];// 2. DeepSort
                    mDeepSort = new DS::DeepSort(sort_engine_path_, 128, 256, 0, &mgLogger);

                    //test
                    if (EnDataSetNameNum == 0){
                        std::string Kittiobjecttrackingfile = EstrDatasetFolder + "/ObjectTracking.txt";
                        ReadKittiObjectInfo(Kittiobjecttrackingfile);
                    }

                    break;
                }

                default:
                    assert(0);
            }
            break;
        }

        case 4:
        {
            EnObjectCenter = fSettings["Viewer.ObjectCenter"];
            EnSelectTrackedObjId = fSettings["Object.EnSelectTrackedObjId"];
            EnInitDetObjORBFeaturesNum = fSettings["Object.EnInitDetObjORBFeaturesNum"];
            int temp2 = fSettings["Object.EbSetInitPositionByPoints"];
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

                    std::string virtualkittiobjectposefile = EstrDatasetFolder + "/pose.txt";
                    std::string virtualkittibboxfile = EstrDatasetFolder + "/bbox.txt";

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
    torch::DeviceType device_type;
    int isGPU = fSettings["Yolo.isGPU"];
    device_type = isGPU > 0 ? torch::kCUDA:torch::kCPU;
    EvClassNames = LoadNames("/home/zpk/SLOT/ORB_SLAM2/weights/coco.names");
    if(EvClassNames.empty())
    {
        cout<<"There is no class file of YOLO!"<<endl;
        assert(0);
    }
    EfConfThres = fSettings["Yolo.confThres"];
    EfIouThres = fSettings["Yolo.iouThres"];
    std::string weights = fSettings["Yolo.weightsPath"];
    //weights = "/home/liuyuzhen/SLOT/orb-slam/code_tempbyme/DetectingAndTracking/libtorch-yolov5-master-temp/weights/yolov5s.torchscript";
    mYOLODetector =  new Detector(weights, device_type);
    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    auto temp_img = cv::Mat::zeros(height, width, CV_32FC3);
    mYOLODetector->Run(temp_img, 1.0f, 1.0f);
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

/// kitti: input: frame_id object_id class truncated occuluded alpha bbox dimensions location rotation_y
/// output: std::vector<std::vector<Eigen::Matrix<double, 1, 20>>> Kitti_AllTrackingObjectInformation
/// 1-20： frame_id track_id truncated occuluded alpha bbox dimensions(height, width, length) location(x,y,z) rotation score type_id is_moving x1
/// location: object position in camera frame
void Tracking::ReadKittiObjectInfo(const std::string &inputfile)
{

    std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> all_object_temp;
    std::vector<Eigen::Matrix<double, 1, 24>> objects_inoneframe_temp;

    std::string pred_frame_obj_txts = inputfile;
    std::ifstream filetxt(pred_frame_obj_txts.c_str());
    std::string line;
    int frame_waibu_id = -1;
    while(getline(filetxt, line))
    {
        if(!line.empty())
        {
            std::stringstream ss(line);
            std::string type;
            int frame_id, track_id;
            double truncated, occuluded, alpha;
            Eigen::Vector4d bbox;
            Eigen::Vector3d dimensions;
            Eigen::Vector3d locations;
            double rotation_y;
            double score;
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

            //if(type == "Pedestrian" or type == "Person_sitting" or type == "Cyclist" or type == "Tram" or type == "Misc" or type == "DontCare")
            //   continue;

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

            if(frame_id != frame_waibu_id && frame_waibu_id != -1)
            {
                all_object_temp.push_back(objects_inoneframe_temp);
                objects_inoneframe_temp.clear();

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

            bbox[2] = bbox[2]-bbox[0];
            bbox[3] = bbox[3]-bbox[1];
            ss>>dimensions[1];
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
            oneobject_oneframe [3] = occuluded;
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

            oneobject_oneframe [20] = 0;
            oneobject_oneframe [21] = 0;
            oneobject_oneframe [22] = 0;
            oneobject_oneframe [23] = 0;



            objects_inoneframe_temp.push_back(oneobject_oneframe);
            frame_waibu_id = frame_id;
        }
    }

    if(int(all_object_temp.size())!=frame_waibu_id)
    {
        cout<<"Error reading offline object pose information！！！"<<endl;
        exit(0);
    }

    all_object_temp.push_back(objects_inoneframe_temp);

    if(frame_waibu_id != int(EnImgTotalNum-1))
    {
        while(all_object_temp.size() != EnImgTotalNum)
        {

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
/// pose：frame_id cameraId trackID alpha width height length
/// world_space_x world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z
/// camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
/// bbox: frame_id cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio ismoving
/// output：std::vector<std::vector<Eigen::Matrix<double, 1, 24>>> Kitti_AllTrackingObjectInformation
///  frame_id track_id truncated occuluded alpha bbox[0]-box[3] dimensions[0]-dimensions[2]
///  location[0]-location[2] rotation_y score type_id is_moving extend bbox_right[0]-bbox[3]
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
    // pose
    double alpha, width, height, length, wx, wy,wz, rwy, rwx, rwz, cx, cy, cz, rcy, rcx, rcz;
    // bbox
    double left, right, top, bottom, number_pixels, truncation_ratio, occupancy_ratio;
    char is_moving[16];

    getline(posetxt, lineforpose);
    getline(bboxtxt, lineforbbox);
    while(getline(posetxt, lineforpose) && getline(bboxtxt, lineforbbox))
    {
        if((!lineforpose.empty())&&(!lineforbbox.empty()))
        {
            std::stringstream sspose(lineforpose);
            std::stringstream ssbbox(lineforbbox);

            sspose>>frame_id;
            ssbbox>>frame_id;
            sspose>>cameraID;
            ssbbox>>cameraID;
            Eigen::Matrix<double, 1, 24> oneobject_oneframe;


            if(frame_id != 0 && frame_waibu_id == -1)
            {
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

            if(int(frame_id) != frame_waibu_id && frame_waibu_id != -1)
            {
                all_object_temp.push_back(objects_inoneframe_temp);
                camera_right=0;
                objects_inoneframe_temp.clear();

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
                oneobject_oneframe [9] = length; // x length
                oneobject_oneframe [10] = height; // y height
                oneobject_oneframe [11] = width; // z width

                if(ORB_SLAM2::EbManualSetPointMaxDistance == false)
                {
                    Eigen::Vector3f scale(length/2, height/2, width/2);
                    ORB_SLAM2::EfInObjFramePointMaxDistance = scale.norm();
                }

                oneobject_oneframe [12] = cx;
                oneobject_oneframe [13] = cy;
                oneobject_oneframe [14] = cz;
                oneobject_oneframe [15] = rcy; // rcx, rcz
                oneobject_oneframe [16] = rcx; // 1 score
                oneobject_oneframe [17] = 1; //  1 type_id

                //if(is_moving=="True")
                if(strcmp(is_moving,"True") == 0)
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

    all_object_temp.push_back(objects_inoneframe_temp);

    if(frame_waibu_id != int(EnImgTotalNum-1))
    {
        while(all_object_temp.size() != EnImgTotalNum)
        {
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


void Tracking::ReadVirtualKittiCameraGT(const std::string &cameraposefile)
{
    std::ifstream cameraposetxt(cameraposefile.c_str());
    std::string linecamerapose;

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
            cv::Mat Pose_tmp = cv::Mat::eye(4, 4, CV_32F);
            if(camera_ID==0)
            {
                sscamerapose >> Pose_tmp.at<float>(0,0) >> Pose_tmp.at<float>(0,1) >>Pose_tmp.at<float>(0,2) >>Pose_tmp.at<float>(0,3)
                             >> Pose_tmp.at<float>(1,0) >> Pose_tmp.at<float>(1,1) >> Pose_tmp.at<float>(1, 2) >> Pose_tmp.at<float>(1, 3)
                             >> Pose_tmp.at<float>(2,0) >> Pose_tmp.at<float>(2,1) >> Pose_tmp.at<float>(2,2) >> Pose_tmp.at<float>(2,3)
                             >> Pose_tmp.at<float>(3,0) >> Pose_tmp.at<float>(3,1) >> Pose_tmp.at<float>(3,2) >> Pose_tmp.at<float>(3,3);

                EvLeftCamGTPose.push_back(Pose_tmp);

            }
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
        int track_id = 1;
        double truncated = 0, occuluded = 1, alpha = 0;
        Eigen::Vector4d bbox;
        Eigen::Vector3d dimensions = EeigUniformObjScale;
        Eigen::Vector3d locations = EeigInitPosition;
        double rotation_y = EeigInitRotation(1);

        double score = 1;
        int type_id = 1;
        int is_moving = 1;
        int extend = 0;

        ss>>frame_id;
        ss>>bbox[0];
        ss>>bbox[1];
        ss>>bbox[2];
        ss>>bbox[3];

        oneobject_oneframe [0] = frame_id;
        oneobject_oneframe [1] = track_id;
        oneobject_oneframe [2] = truncated;
        oneobject_oneframe [3] = occuluded;

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

            case 4:
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

            case 3: // Autonomous Driving Mode
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


/// Track, Construct 3D objects from 2D instances
///3D objects attribute: The two most critical attributes of MapObject are allFrameCuboidPoses (regular frame pose) and allDynamicPoses (keyframe pose)
///Strategy: If it has occurred (then use the velocity model), update the pose of the 3D object (or choose to be directly given by the 2D offline results): If it is within the time threshold, use the velocity model;

bool Tracking::TrackMapObject()
{
    if(mCurrentFrame.mvDetectionObjects.size()==0)
        return false;

    /// 1. 遍历所有当前帧检测的2D objects：cuboids_on_frame.

    for(size_t i=0; i<mCurrentFrame.mnDetObj;i++)
    {
        DetectionObject *candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];

        if(candidate_cuboid==NULL)
            continue;

        bool object_exists = false;

        for(size_t j=0; j<AllObjects.size();j++)
        {
            MapObject *object_temp = AllObjects[j];
            if(object_temp->mnTruthID == candidate_cuboid->mnObjectID)
            {
                g2o::ObjectState cInFrameLatestObjState;
                int nLatestFrameId = object_temp->GetCFLatestFrameObjState(cInFrameLatestObjState);// Tco
                if(nLatestFrameId == -1)
                    assert(0);
                if(!object_temp->mlRelativeFramePoses.count(nLatestFrameId))
                    assert(0);
                if(object_temp->mpReferenceObjKF !=  object_temp->mlRelativeFramePoses[nLatestFrameId].first)
                    assert(0);
                cv::Mat camera_Tcl = cv::Mat::eye(4, 4, CV_32F);
                if (!mCurrentFrame.mTcw.empty() && !mLastFrame.mTwc.empty()){
                    camera_Tcl = mCurrentFrame.mTcw.clone() * mLastFrame.mTwc.clone();
                }
                cInFrameLatestObjState.pose = Converter::toSE3Quat(camera_Tcl) * cInFrameLatestObjState.pose;

                if(cInFrameLatestObjState.scale[0]<0.05)
                    assert(0);

                g2o::ObjectState cuboid_current_temp;
                double delta_t = (mCurrentFrame.mnId - nLatestFrameId) * EdT;
                if(delta_t < EdMaxObjMissingDt && EbUseOfflineAllObjectDetectionPosesFlag == false)
                {
                    //Uniform velocity model, predicts the next frame pose of the object, and updates the target speed
                    //object_temp->mVirtualVelocity = Vector6d::Zero();
                    cuboid_current_temp = cInFrameLatestObjState;
                    bool currentpose = InitializeCurrentObjPose(i, cuboid_current_temp);
                    if (!currentpose)
                        //Using speed prediction in case of occlusion
                        cuboid_current_temp.UsingVelocitySetPredictPos(object_temp->mVirtualVelocity, delta_t);
                }
                else{
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
                object_temp->SetInFrameObjState(Swo, mCurrentFrame.mnId);
                object_temp->SetCFInFrameObjState(cuboid_current_temp, mCurrentFrame.mnId);

                candidate_cuboid->SetMapObject(object_temp);
                object_temp->AddFrameObservation(mCurrentFrame.mnId, i);
                mCurrentFrame.mvMapObjects[i] = object_temp;
                object_temp->mmmDetections[mCurrentFrame.mnId] = candidate_cuboid;

                //cv::waitKey(0);

                object_exists = true;
                if(nLatestFrameId == int(mLastFrame.mnId))
                {
                    //cout<<"Object: "<<candidate_cuboid->mnObjectID<<" success"<<", speed: "<<object_temp->mVirtualVelocity.transpose()<<" dt: "<<delta_t<<endl;;
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
        // If the object is observed for the first time, establish a 3D object and add All Objects
        if(object_exists==false)
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
    //align the pose with the 2D bounding box
    DetectionObject* candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];
    int nNum = 0;
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

    // RANSAC
    std::vector<int> best_inlier_set;
    float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: candidate_cuboid->mTruthPosInCameraFrame.scale.norm();
    unsigned int l = Pcjs.size();
    cv::RNG rng;
    double bestScore = -1.;
    int iterations = 0.8 * Pcjs.size();
    for(int k=0; k<iterations; k++)
    {
        std::vector<int> inlier_set;
        int i1 = rng(l);
        const Eigen::Vector3d& p1 = Pcjs[i1].first;
        double score = 0;
        for(int u=0; u<l; u++)
        {
            Eigen::Vector3d v = Pcjs[u].first-p1;
            if( v.norm() < fMaxDis )
            {
                inlier_set.push_back(u);
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

    // 1. bounding box center y alignment
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


    // 2.bounding box height
    // depth alignment
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
                break;
            }
        }

    }

    // 3. center x alignment
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
            break;
        }
    }
}
void Tracking::MapObjectInit(const int &i)
{
    DetectionObject* candidate_cuboid = mCurrentFrame.mvDetectionObjects[i];
    g2o::ObjectState gt = candidate_cuboid->mTruthPosInCameraFrame;
    g2o::ObjectState InitPose = candidate_cuboid->mTruthPosInCameraFrame;

    // initialize the object pose
    //InitPose.setRotation(Eigen::Vector3d(0,0,0));
    //InitPose.setScale(ORB_SLAM2::EeigUniformObjScale);
    int nNum = 0;
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

    // RANSAC
    std::vector<int> best_inlier_set;
    float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: InitPose.scale.norm();
    unsigned int l = Pcjs.size();
    cv::RNG rng;
    double bestScore = -1.;
    int iterations = 0.8 * Pcjs.size();
    for(int k=0; k<iterations; k++)
    {
        std::vector<int> inlier_set;
        int i1 = rng(l);
        const Eigen::Vector3d& p1 = Pcjs[i1].first;
        double score = 0;
        for(int u=0; u<l; u++)
        {
            Eigen::Vector3d v = Pcjs[u].first-p1;
            if( v.norm() < fMaxDis )
            {
                inlier_set.push_back(u);
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

    /// 3.1.2 Read the current pose:   Two = Twc * Tco
    MapObject *new_object = new MapObject(candidate_cuboid->mnObjectID, candidate_cuboid->GetDynamicFlag(),mCurrentFrame.mnId, true, Vector6d::Zero(), InitPose, true);// Construct the 3d object
    new_object->mbPoseInit = candidate_cuboid->mInitflag;
    new_object->SetInFrameObjState(g2o::ObjectState(mCurrentFrame.mSETcw.inverse() * InitPose.pose, InitPose.scale), mCurrentFrame.mnId);
    AllObjects.push_back(new_object);
    candidate_cuboid->SetMapObject(new_object);
    mpMap->AddMapObject(new_object);
    mCurrentFrame.mvMapObjects[i] = new_object;
    new_object->AddFrameObservation(mCurrentFrame.mnId, i);
    mCurrentFrame.mvnNewConstructedObjOrders.push_back(i);
    ObjectKeyFrame* pKFini = new ObjectKeyFrame(mCurrentFrame, i, true);// Construct the object keyframe
    new_object->mnLastKeyFrameId = mCurrentFrame.mnId;
    new_object->SetCFObjectKeyFrameObjState(pKFini, InitPose);


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

    cout<<"Object: "<<candidate_cuboid->mnObjectID<<"  constructed successfully, "<<" features number: "<<nReal<<endl;
    cout<<endl;
    if(EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4)
        mpObjectLocalMapper->InsertOneObjKeyFrame(pKFini);
    new_object->mpReferenceObjKF = pKFini;
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
            int nNum = 0;
            for(size_t j=0; j<vDepthIdx.size(); j++)
            {
                cv::Mat x3Dc = mCurrentFrame.UnprojectStereodynamic(order, vDepthIdx[j].second, false);
                Eigen::Vector3d Pcj = Converter::toVector3d(x3Dc);
                Pcjs.push_back(make_pair(Pcj, vDepthIdx[j].second));
                nNum++;
                if(vDepthIdx[j].first>2*mThDepth && nNum>100)
                    break;
            }

        }


        // RANSAC滤出一些离谱点
        std::vector<int> best_inlier_set;
        float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: InitPose.scale.norm();
        unsigned int l = Pcjs.size();
        cv::RNG rng;
        double bestScore = -1.;
        int iterations = Pcjs.size();
        for(int k=0; k<iterations; k++)
        {
            std::vector<int> inlier_set;
            int i1 = rng(l);
            const Eigen::Vector3d& p1 = Pcjs[i1].first;
            double score = 0;
            for(int u=0; u<l; u++)
            {
                Eigen::Vector3d v = Pcjs[u].first-p1;
                if( v.norm() < fMaxDis )
                {
                    inlier_set.push_back(u);
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
        if (eigObjPosition[2]>8)
        eigObjPosition[2] += 0.2*pDet->mScale[0];
        eigObjPosition[1] = 0 + pDet->mScale[1]/2;
        //if(EbSetInitPositionByPoints && best_inlier_set.size() > 0)

        InitPose.setTranslation(eigObjPosition);
        FineTuningUsing2dBox(order,InitPose);

        pMO->SetCFInFrameObjState(InitPose, mCurrentFrame.mnId);

        ObjectKeyFrame* pOKF = new ObjectKeyFrame(mCurrentFrame, order, true);
        pMO->mpReferenceObjKF = pOKF;
        pMO->SetCFObjectKeyFrameObjState(pOKF,pDet->mTruthPosInCameraFrame);


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

        for (size_t j = 0; j < AllObjects.size(); j++) {
            MapObject *object_temp = AllObjects[j];

            if (object_temp->mnTruthID == candidate_cuboid->mnObjectID) {
                candidate_cuboid->SetMapObject(object_temp);
            }
            //If the last frame of the mapobject matches the observation, but the frame does not match,
            // it is necessary to further determine whether a prediction frame is needed
            // TODO
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
            // If the target dynamic and static properties are fixed and static
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
                    // Prevent duplicate Recover
                    cDO->IsRecovered = vObjKeys[i].size();
                }

            }
        }

    }
    //cout << "Before Recovery: " << mCurrentFrame.N ;
    N = vKeys.size();
    //cout << ", After Recovery: " << mCurrentFrame.N << endl;
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
                        mLastFrame.mvpMapObjectPoints[n][i] = static_cast<MapObjectPoint *>(NULL);
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


void Tracking::TrackLastFrameObjectPoint(const bool &bUseTruthObjPoseFlag)
{
    cout<<endl;
    //cout<<YELLOW<<"Object point tracking last frame start:"<<endl;
    cout<<WHITE;


    vector<pair<size_t, size_t>> vInLastFrameTrackedObjOrders = mCurrentFrame.mvInLastFrameTrackedObjOrders;

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

        int nPoints = 0;
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
                bCreateNew = true;
            }
            if(bCreateNew)
            {
                g2o::ObjectState gcCuboidTmp;
                cv::Mat map_to_obj, x3Dc;
                x3Dc = mLastFrame.UnprojectStereodynamic(nLastOrder, i, false);
                if(bUseTruthObjPoseFlag)
                {
                    gcCuboidTmp = cCuboidTemp->mTruthPosInCameraFrame;
                    map_to_obj = Converter::toCvMat(gcCuboidTmp.pose.inverse().map(Converter::toVector3d(x3Dc)));
                }
                else{
                    gcCuboidTmp = pMO->GetCFInFrameObjState(mLastFrame.mnId); // Tco
                    map_to_obj = Converter::toCvMat(gcCuboidTmp.pose.inverse().map(Converter::toVector3d(x3Dc))); // Poj = Tco.inverse() * Pcj
                }
                Eigen::Vector3f feature_in_object = Converter::toVector3f(map_to_obj);
                float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: gcCuboidTmp.scale.norm();
                if (feature_in_object.norm() > fMaxDis)
                {
                    continue;
                }
                MapObjectPoint *pNewMP = new MapObjectPoint(pMO, map_to_obj, x3Dc, &mLastFrame, nLastOrder, i);
                mLastFrame.mvpMapObjectPoints[nLastOrder][i] = pNewMP;
                pMO->mlpTemporalPoints.push_back(pNewMP);
                cCuboidTemp->AddMapObjectPoint(i, pNewMP);
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

    }


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
            case 0: // Brute Matching
            {
                vector<MapObjectPoint* >vpMapObjectPointMatches;
                int nmatches = matcher.SearchByBruceMatching(mLastFrame, mCurrentFrame, nLastOrder, nCurrentOrder, vpMapObjectPointMatches);
                mCurrentFrame.mvpMapObjectPoints[nCurrentOrder] = vpMapObjectPointMatches;
                //int nmatches = matcher.SearchByBruceMatchingWithGMS(mLastFrame, mCurrentFrame, nLastOrder, nCurrentOrder);
                if(nmatches < 10)
                    pDet->mbTrackOK = false;
                else
                {
                    vnNeedToBeOptimized.push_back(nCurrentOrder);
                }
                break;
            }

            case 1: // Offline optical flow tracking
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
        Optimizer::CFSE3ObjStateOptimization(&mCurrentFrame, vnNeedToBeOptimized, false);// BA
        for(size_t n=0; n<vnNeedToBeOptimized.size(); n++)
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
                if(mCurrentFrame.mvbObjKeysOutlier[i][m])
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

            {
                if(nmatchesMap<10)
                    pDet->mbTrackOK = false;
                else
                    pDet->mbTrackOK = true;
                //cout<<"Object"<<pDet->mnObjectID<<" Track the number of points in the previous frame: "<<nmatchesMap<<" tracking status: "<<pDet->mbTrackOK<<endl;
            }
        }
    }

}

void Tracking::TrackObjectLocalMap()
{
    cout<<endl;
    //cout<<YELLOW<<"The Object point tracking local map starts:"<<WHITE<<endl;

    vector<size_t> vnNeedToBeOptimized;
    vnNeedToBeOptimized.reserve(mCurrentFrame.mvTotalTrackedObjOrders.size());
    for(size_t n=0; n<mCurrentFrame.mvTotalTrackedObjOrders.size(); n++)
    {
        size_t nOrder = mCurrentFrame.mvTotalTrackedObjOrders[n];
        DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[nOrder];

        UpdateObjectLocalKeyFrames(nOrder);
        UpdateObjectLocalPoints(nOrder);
        int nTracks = SearchObjectLocalPoints(nOrder);

        vnNeedToBeOptimized.push_back(nOrder);
    }
    if(vnNeedToBeOptimized.size() == 0)
        return;

    if(1)
    {
        Optimizer::CFSE3ObjStateOptimization(&mCurrentFrame, vnNeedToBeOptimized, false);
        for(size_t n=0; n<vnNeedToBeOptimized.size(); n++)
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
                if(mCurrentFrame.mvbObjKeysOutlier[nOrder][m] == false)
                {
                    t2++;
                    pMP->IncreaseFound();
                    if(pMP->Observations()>0)
                    {
                        t3++;
                        pDet->mnMatchesInliers++;
                    }

                }
                else{
                    t4++;
                    mCurrentFrame.mvpMapObjectPoints[nOrder][m] = static_cast<MapObjectPoint*>(NULL);
                }
            }

            if(pDet->mnMatchesInliers>10)
                pDet->mbTrackOK = true;
            else
                pDet->mbTrackOK = false;
            //cout<<"Object"<<pDet->mnObjectID<<" Track the number of points in the local map: "<<pDet->mnMatchesInliers<<" tracking status: "<<pDet->mbTrackOK<<" t1:" <<t1<<" t2:"<<t2<<" t3:"<<t3<<" t4: "<<t4<<endl;
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
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        if(mCurrentFrame.isInFrustum(pMP, n, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        nRealMatch = matcher.SearchByProjection(mCurrentFrame, n, pMO->mvpLocalMapObjectPoints,th);
    }

    return nRealMatch;
}

void Tracking::UpdateObjectLocalKeyFrames(const size_t &nInCurrentFrameOrder)
{
    MapObject* pMO = mCurrentFrame.mvMapObjects[nInCurrentFrameOrder];
    if(pMO==NULL)
        return;
    map<ObjectKeyFrame*,int> keyframeCounter;
    for(size_t i=0; i<mCurrentFrame.mvpMapObjectPoints[nInCurrentFrameOrder].size(); i++)
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
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;

            if(!pMP->isBad())
            {
                pMO->mvpLocalMapObjectPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;

                if(!hash.count(pMP->mnId))
                    hash.insert(pMP->mnId);
                else
                    assert(0);
            }
        }
    }
    //cout<<"Object"<<pMO->mnTruthID<<"Number of local key frames:"<<pMO->mvLocalObjectKeyFrames.size()<<"Number of local 3D points: "<<pMO->mvpLocalMapObjectPoints.size( )<<" ";

   return;
}

bool Tracking::NeedNewObjectKeyFrame(const size_t &nOrder)
{
    bool bNeedInsert = false;
    DetectionObject* pDetObj = mCurrentFrame.mvDetectionObjects[nOrder];
    MapObject* pMO = mCurrentFrame.mvMapObjects[nOrder];
    if(pMO == NULL || pDetObj==NULL)
        assert(0);
    //const bool bFlag1 = mCurrentFrame.mnId>=(pMO->mnLastKeyFrameId+mMaxFrames);
    const bool bFlag1 = mCurrentFrame.mnId>=(pMO->mnLastKeyFrameId+6);// Condition 1: No keyframe has been added to this object for a long time
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
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
    int nRefMatches = mpReferenceKF->TrackedMapPoints(2);
    const bool bFlag3 = (pDetObj->mnMatchesInliers < nRefMatches*0.1 || bNeedToInsertFlag);// Condition 3: The object tracking effect is not good, all inliers
    const bool bFlag4 = pDetObj->mnMatchesInliers>20;// Condition 4: The number of inliers of the target in this frame must exceed the threshold
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

    MapObject* moObjectTmp = mCurrentFrame.mvMapObjects[nOrder];
    DetectionObject* pDet = mCurrentFrame.mvDetectionObjects[nOrder];

    if(moObjectTmp==NULL || pDet==NULL)
        assert(0);
    if(pDet->mbNeedCreateNewOKFFlag==false)
        assert(0);

    ObjectKeyFrame* pOKF = new ObjectKeyFrame(mCurrentFrame, nOrder, false);
    moObjectTmp->mpReferenceObjKF = pOKF;
    moObjectTmp->SetCFObjectKeyFrameObjState(pOKF, moObjectTmp->GetCFInFrameObjState(mCurrentFrame.mnId));
    //cout<<"OKF Frame ID: "<<pOKF->mnFrameId<<", KF ID: "<<pOKF->mnId<<", Object ID"<<moObjectTmp->mnTruthID<<endl;
    vector<pair<float,int> > vDepthIdx;// stereo triangulation
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
            else if(pMP->Observations()<1)
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


                float fMaxDis = EbManualSetPointMaxDistance ? EfInObjFramePointMaxDistance: ObjState.scale.norm();
                if(Poj.norm() > fMaxDis)
                    continue;

                MapObjectPoint* pNewMP = new MapObjectPoint(moObjectTmp, Converter::toCvMat(Poj), Pcj, pOKF);
                if (!pNewMP->mpRefObjKF) assert(0);
                pNewMP->AddObservation(pOKF,i);
                pOKF->AddMapObjectPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                moObjectTmp->AddMapObjectPoint(pNewMP);

                if(mCurrentFrame.mvbObjKeysOutlier[nOrder][i] == false)
                {
                    mCurrentFrame.mvpMapObjectPoints[nOrder][i] = pNewMP;
                }
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

    if(EnSLOTMode == 2 || EnSLOTMode ==3 ||EnSLOTMode == 4)
    {
        mpObjectLocalMapper->InsertOneObjKeyFrame(pOKF);
    }

    moObjectTmp->mnLastKeyFrameId = mCurrentFrame.mnId;

}


void Tracking::StereoInitialization()
{

    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin

        if(EbSetWorldFrameOnGroundFlag)
        {
            mCurrentFrame.SetPose(mTwc0);
        } else{
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        }

        if(mCurrentFrame.mnDetObj!=0 && 0) // 第一帧没有track
        {
            TrackMapObject();
        }

        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        cout<<"!!!Stereo initialization: create the first keyframe: "<<pKFini->mnId<<"          total_id:    "<<mCurrentFrame.mnId<<endl;

        mpMap->AddKeyFrame(pKFini);

        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);

                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " static points" << "   and   "<<mpMap->MapObjectPointsInMap()<<"   dynamic points"<<endl;


        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);

        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPoseAndId(mCurrentFrame.mTcw, mCurrentFrame.mnId);

        mState=OK;
    }

}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector

    cout<<"TrackReferenceKeyFrame"<<endl;
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver

    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    mCurrentFrame.SetPose(mLastFrame.mTcw);
    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
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
        if(vDepthIdx[j].first>2*mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame

    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }
    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();
    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

                if(!mbOnlyTracking)
                {

                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;


    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames) // 若距离上一次重定位不超过1s同时地图里有一定量的关键帧则跳过
        return false;

    // Tracked MapPoints in the reference keyframe

    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames(); // 返回mbAcceptKeyFrames

    // Check how many "close" points are being tracked and how many could be potentially created.

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

    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ; // inliers比较少, 或者可以进一步生成比较多地图点
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);// c2?

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    // TODO
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    //cout<<endl<<BLUE<<"TODO:    created new keyframe!    "<<pKF->mnId<<"           total ID          "<<pKF->mnFrameId<<endl;
    cout<<WHITE<<endl;
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {

        mCurrentFrame.UpdatePoseMatrices();
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
            sort(vDepthIdx.begin(),vDepthIdx.end());
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;
                bool bCreateNew = false;
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
    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);
    mnLastKeyFrameId = mCurrentFrame.mnId;
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
                *vit = static_cast<MapPoint*>(NULL);
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
    }
}


void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
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
