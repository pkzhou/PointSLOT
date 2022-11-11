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
#include "Parameters.h"
#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "DetectionObject.h"
#include "MapObject.h"
#include "MapPoint.h"
#include "MapObjectPoint.h"
#include "YOLOdetector.h"
#include <thread>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    if(EbSLOTFlag)
    {
        mvObjKeys = frame.mvObjKeys;
        mvObjKeysUn = frame.mvObjKeys;
        mvpMapObjectPoints = frame.mvpMapObjectPoints;
        mvObjPointsDescriptors = frame.mvObjPointsDescriptors;
        mvuObjKeysRight = frame.mvuObjKeysRight;
        mvObjPointDepth = frame.mvObjPointDepth;
        mvbObjKeysMatchedFlag = frame.mvbObjKeysMatchedFlag;

        mnDetObj = frame.mnDetObj;

        mvMapObjects = frame.mvMapObjects;
        mvDetectionObjects = frame.mvDetectionObjects;
        mvbObjKeysOutlier = frame.mvbObjKeysOutlier;

        mObjMask = frame.mObjMask.clone();
        mRawImg = frame.mRawImg.clone();
        mForwardOpticalImg = frame.mForwardOpticalImg.clone();


        mvObjKeysGrid.resize(mnDetObj);
        for(size_t k=0; k<mnDetObj; k++)
        {
            mvObjKeysGrid[k].resize(FRAME_GRID_COLS);
            for (int i = 0; i < FRAME_GRID_COLS; i++)
            {
                mvObjKeysGrid[k][i].resize(FRAME_GRID_ROWS);
                for (int j = 0; j < FRAME_GRID_ROWS; j++)
                {
                    mvObjKeysGrid[k][i][j] = frame.mvObjKeysGrid[k][i][j];
                }
            }
        }
    }
}

cv::Mat Demo(cv::Mat& img,
             const std::vector<std::vector<Detection>>& detections,
             const std::vector<std::string>& class_names,
             bool label = true)
{

    if (!detections.empty())
    {
        for (const auto& detection : detections[0])
        {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label)
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                              cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                              cv::Point(box.tl().x + s_size.width, box.tl().y),
                              cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }

    return img;
}
// Stereo SLAM
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
        :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Semantic Dynamic SLAM
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
             const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
             Detector* YOLODetector)
        :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mpReferenceKF(static_cast<KeyFrame*>(NULL))
{

    // Frame ID
    mnId=nNextId++;
    mRawImg = imLeft.clone();
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mMaskImgRight = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mnDetObj = 0;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);


    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;


    UndistortKeyPoints();

    ComputeStereoMatches();

    vector<DetectionObject*> vDetectionObjects;
    switch(EnOnlineDetectionMode)
    {
        case 0:
        {
            OfflineDetectObject(vDetectionObjects);// Read offline data
            int temp = 1;
            if(temp)
            {
                switch(EnDataSetNameNum)
                {
                    case 0:{
                        // Kitti_tracking
                        std::string MOTS_PNG_folder = EstrDatasetFolder + "/Segmentation/";
                        mMaskImg = ReadKittiSegmentationImage(MOTS_PNG_folder, ORB_SLAM2::EnStartFrameId);
                        break;
                    }
                    case 1: {
                        // Virtual_kitti
                        std::string MOTS_forvirtualKitti = EstrDatasetFolder + "/Segpgm/";
                        mMaskImg = ReadVirtualKittiSegmentationImage(MOTS_forvirtualKitti, ORB_SLAM2::EnStartFrameId);
                        break;
                    }
                    default:
                        assert(0);
                }
            }
            else
            {

            }
            break;
        }

        case 1: // online
        {
            std::vector<std::vector<Detection>> result = YOLODetector->Run(imColor, EfConfThres, EfIouThres);
            if(result.size() != 0)
            {
                mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
                for(auto &m: result[0])
                {
                    if(m.class_idx == 2 || m.class_idx == 7 ) // car/truck as dynamic object
                    {
                        cv::Rect tmp = m.bbox;
                        cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
                        m0.setTo(255);
                    }
                }
            }
            break;
        }
        default:
            assert(0);
    }


    AssignFeatures(vDetectionObjects);

    N = mvKeys.size();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Object Tracking + Dynamic SLAM
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
             const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
             Detector* YOLODetector, cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers)
        :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    mnId=nNextId++;
    mRawImg = imLeft.clone();
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mMaskImgRight = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mnDetObj = 0;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);


    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;


    UndistortKeyPoints();

    ComputeStereoMatches();

    // tracker
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    vector<DetectionObject*> vDetectionObjects;
    Online2DObjectTracking(multiTracker, vTrackers, vDetectionObjects);
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    float th1 = 5, th2 = 5, th3 = 5, th4 = 5;
    for(size_t i=0; i < vDetectionObjects.size(); i++)
    {
        cv::Rect tmp = vDetectionObjects[i]->mrectBBox;
        int id = vDetectionObjects[i]->mnObjectID;
        cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
        m0.setTo(255);
    }

    if(EbYoloActive)
    {
        std::vector<std::vector<Detection>> result = YOLODetector->Run(imColor, EfConfThres, EfIouThres);
        if(result.size() != 0)
        {
            for(auto &m: result[0])
            {
                if(m.class_idx == 2 || m.class_idx == 7 )
                {
                    cv::Rect tmp = m.bbox;
                    cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
                    m0.setTo(255);
                }
            }
        }
    }

    vDetectionObjects.clear();
    AssignFeatures(vDetectionObjects);

    N = mvKeys.size();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        mbInitialComputations=false;
    }


    mb = mbf/fx;


    AssignFeaturesToGrid();
}


// Object Tracking Mode
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
                 const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
                 cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers)
            :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
             mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    mnId=nNextId++;
    mRawImg = imLeft.clone();
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mMaskImgRight = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mnDetObj = 0;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);

    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;


    UndistortKeyPoints();
    ComputeStereoMatches();


    vector<DetectionObject*> vDetectionObjects;
    if(ORB_SLAM2::EnOnlineDetectionMode)
    {
        Online2DObjectTracking(multiTracker, vTrackers, vDetectionObjects);
    }
    else
    {
        OfflineDetectObject(vDetectionObjects);
    }

    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    float th1 = 5, th2 = 5, th3 = 5, th4 = 5;
    for(size_t i=0; i < vDetectionObjects.size(); i++)
    {
        cv::Rect tmp = vDetectionObjects[i]->mrectBBox;
        int id = vDetectionObjects[i]->mnObjectID;
        cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
        m0.setTo(255);
        cv::Mat m1 = mMaskImg(cv::Rect(tmp.x + th1, tmp.y + th2, tmp.width-th3-th1, tmp.height - th2 - th4));
        m1.setTo(id+1);
    }
    cout<<endl<<" Frame "<< mnId<<"  The number of detected 2D objects: "<<vDetectionObjects.size()<<endl;


    AssignFeatures(vDetectionObjects);

    N = mvKeys.size();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Autonomous Driving Mode
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
             const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
             Detector* YOLODetector, DS::DeepSort* deepSort)
        :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    mnId=nNextId++;
    mRawImg = imLeft.clone();
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mMaskImgRight = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mnDetObj = 0;
    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    vector<DetectionObject*> vDetectionObjects;

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    // YOLO detection
    thread threadBBox(&Frame::DetectYOLO,this,std::ref(imColor),std::ref(vDetectionObjects),YOLODetector,deepSort,imLeft,imRight);
    threadLeft.join();
    threadRight.join();
    threadBBox.join();

    N = mvKeys.size();
    if(mvKeys.empty())
        return;


    UndistortObjKeyPoints();
    ComputeObjStereoMatches();

    cout<<"The number of object features: "<<mvTempObjKeys.size();

    UndistortKeyPoints();
    ComputeStereoMatches();


    switch(EnOnlineDetectionMode)
    {
        case 0:
        {
            OfflineDetectObject(vDetectionObjects);// Read offline data
            int temp = 1;
            if(temp)
            {
                switch(EnDataSetNameNum)
                {
                    case 0:{
                        // Kitti_tracking
                        std::string MOTS_PNG_folder = EstrDatasetFolder + "/Segmentation/";
                        mMaskImg = ReadKittiSegmentationImage(MOTS_PNG_folder, ORB_SLAM2::EnStartFrameId);

                        if(EbUseSegementation == false)
                        {
                            mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
                            double th1 = 10, th2 = 10, th3 = 10, th4 = 10;
                            for(size_t i=0; i < vDetectionObjects.size(); i++)
                            {
                                cv::Rect tmp = vDetectionObjects[i]->mrectBBox;
                                int id = vDetectionObjects[i]->mnObjectID;
                                cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
                                m0.setTo(255);
                                cv::Mat m1 = mMaskImg(cv::Rect(tmp.x + th1, tmp.y + th2, tmp.width-th3-th1, tmp.height - th2 - th4));
                                m1.setTo(id+1);
                            }
                        }

                        break;
                    }
                    case 1: {
                        // Virtual_kitti
                        // The pixel value is trackID+1， 0 denotes the background
                        std::string MOTS_forvirtualKitti = EstrDatasetFolder + "/Segpgm/";
                        mMaskImg = ReadVirtualKittiSegmentationImage(MOTS_forvirtualKitti, ORB_SLAM2::EnStartFrameId);
                        // Read the optical flow images
                        std::string ForwardOpticalFlowFolder = EstrDatasetFolder + "/forwardFlow/";
                        mForwardOpticalImg = ReadVirtualKittiForwardOpticalFlow(ForwardOpticalFlowFolder, ORB_SLAM2::EnStartFrameId);
                        break;
                    }
                    default:
                        assert(0);
                }
            }
            else
            {


            }
            break;
        }

        case 1: // online
        {
            // This process has been performed in parallel with ORB extraction.
            //DetectYOLO(imColor,vDetectionObjects,YOLODetector,deepSort);
            break;
        }
        default:
            assert(0);
    }

    cout<<endl<<YELLOW<<"Frame: "<<mnId<<" The number of 2D detected objects:"<<vDetectionObjects.size()<<WHITE<<endl;

    // Input a mask.
    // The area whose mask is 0 is background, 255 is sensitive, and 1-249 is the object area (object ID+1)
    AssignFeatures(vDetectionObjects);


    N = mvKeys.size();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        mbInitialComputations=false;
    }

    mb = mbf/fx;


    AssignFeaturesToGrid();

}


// Test Mode
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat & imColor,
             const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
             const int& SLOTMode)
        :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    if(SLOTMode != 4)
        assert(0);
    mnId=nNextId++;
    mRawImg = imLeft.clone();
    mMaskImg = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mMaskImgRight = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    mnDetObj = 0;
    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();


    vector<DetectionObject*> vDetectionObjects;
    OfflineDetectObject(vDetectionObjects);
    switch(EnDataSetNameNum)
    {
        case 0:{
            // Kitti_tracking
            std::string MOTS_PNG_folder = EstrDatasetFolder + "/Segmentation/";
            mMaskImg = ReadKittiSegmentationImage(MOTS_PNG_folder, ORB_SLAM2::EnStartFrameId);
            mMaskImgRight = ReadKittiSegmentationImage(MOTS_PNG_folder, ORB_SLAM2::EnStartFrameId, true);
            break;
        }
        case 1: {
            // Virtual_kitti
            // The pixel value is trackID+1， 0 denotes the background
            std::string MOTS_forvirtualKitti = EstrDatasetFolder + "/Segpgm/";
            mMaskImg = ReadVirtualKittiSegmentationImage(MOTS_forvirtualKitti, ORB_SLAM2::EnStartFrameId);
            // Read the optical flow images
            std::string ForwardOpticalFlowFolder = EstrDatasetFolder + "/forwardFlow/";
            mForwardOpticalImg = ReadVirtualKittiForwardOpticalFlow(ForwardOpticalFlowFolder, ORB_SLAM2::EnStartFrameId);
            break;
        }
        default:
            assert(0);
    }

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    thread threadObjORB(&Frame::ExtractObjORB,this,imLeft,imRight,ref(vDetectionObjects));
    threadLeft.join();
    threadRight.join();
    threadObjORB.join();

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    UndistortObjKeyPoints();
    ComputeObjStereoMatches();



    cout<<endl<<YELLOW<<"Frame: "<<mnId<<" The number of 2D detected objects:"<<vDetectionObjects.size()<<WHITE<<endl;

    // Input a mask.
    // The area whose mask is 0 is background, 255 is sensitive, and 1-249 is the object area (object ID+1)
    AssignFeatures(vDetectionObjects);


    N = mvKeys.size();
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        mbInitialComputations=false;
    }

    mb = mbf/fx;


    AssignFeaturesToGrid();

}

void Frame::AssignFeatures(vector<DetectionObject*> &vDetectionObjects)
{
    std::vector<cv::KeyPoint> mvKeys_cp;
    cv::Mat mDescriptors_cp;
    std::vector<float> mvuRight_cp;
    std::vector<float> mvDepth_cp;
    std::vector<cv::KeyPoint> mvKeysUn_cp;

    size_t nObjNum = vDetectionObjects.size();
    std::vector<std::vector<cv::KeyPoint>> mvObjKeys_cp;
    vector<cv::Mat> mvObjPointsDescriptors_cp;
    std::vector<std::vector<cv::KeyPoint>> mvObjKeysUn_cp;
    std::vector<std::vector<float>> mvObjPointDepth_cp;
    std::vector<size_t> vnEffectiveObjPointDepthNums(nObjNum, 0);
    std::vector<std::vector<float>> mvuObjKeysRight_cp;

    std::vector<std::vector<cv::KeyPoint>> mvOriKeys_cp;
    std::vector<std::vector<cv::KeyPoint>> mvOriKeysUn_cp;
    vector<cv::Mat> mOriDescriptors_cp;
    vector<std::vector<float>> mvuOriRight_cp;
    vector<std::vector<float>> mvOriDepth_cp;

    mvObjKeys_cp.resize(nObjNum);
    mvObjPointsDescriptors_cp.resize(nObjNum);
    mvObjKeysUn_cp.resize(nObjNum);
    mvObjPointDepth_cp.resize(nObjNum);
    mvuObjKeysRight_cp.resize(nObjNum);

    mvOriKeys_cp.resize(nObjNum);
    mOriDescriptors_cp.resize(nObjNum);
    mvOriKeysUn_cp.resize(nObjNum);
    mvOriDepth_cp.resize(nObjNum);
    mvuOriRight_cp.resize(nObjNum);

    mvDetectionObjects.reserve(nObjNum);
    mvObjKeys.reserve(nObjNum);
    mvObjPointsDescriptors.reserve(nObjNum);
    mvObjKeysUn.reserve(nObjNum);
    mvObjPointDepth.reserve(nObjNum);
    mvuObjKeysRight.reserve(nObjNum);

    mvOriKeys.reserve(nObjNum);
    mOriDescriptors.reserve(nObjNum);
    mvOriKeysUn.reserve(nObjNum);
    mvOriDepth.reserve(nObjNum);
    mvuOriRight.reserve(nObjNum);

    // Features assignment
    // The feature point belongs to background in the area with a pixel value of 0;
    // The feature point belongs to objects in the area where the pixel value is not 0 (except 255, 255 is the ignored area)
    int staticnumber = 0;
    for (size_t i = 0; i < mvKeys.size(); i++)
    {
        int x = mvKeys[i].pt.x;
        int y = mvKeys[i].pt.y;


        if (int(mMaskImg.at<u_int8_t>(y, x)) == 0)
        {

            mvKeys_cp.push_back(mvKeys[i]);
            mDescriptors_cp.push_back(mDescriptors.row(i));
            mvKeysUn_cp.push_back(mvKeysUn[i]);
            mvuRight_cp.push_back(mvuRight[i]);
            mvDepth_cp.push_back(mvDepth[i]);
            staticnumber++;
        }

        else{
            // Save the sparse feature points of the object area
            // Object area: the pixel value is not 0 and not 255
            int object_id_temp = int(mMaskImg.at<u_int8_t>(y, x));
            if (object_id_temp != 0 && object_id_temp != 255)
            {
                for(size_t j=0; j<nObjNum;j++)
                {
                    if (!vDetectionObjects[j])
                        continue;
                    int id;
                    if (vDetectionObjects[j]->mnObjectID>255)
                        id = vDetectionObjects[j]->mnObjectID - 255;
                    else id = vDetectionObjects[j]->mnObjectID;
                    if (id == object_id_temp - 1) // 2D 检测的object id = mask object id -1
                    {
                        mvOriKeys_cp[j].push_back(mvKeys[i]);
                        mOriDescriptors_cp[j].push_back(mDescriptors.row(i));
                        mvOriKeysUn_cp[j].push_back(mvKeysUn[i]);
                        mvuOriRight_cp[j].push_back(mvuRight[i]);
                        mvOriDepth_cp[j].push_back(mvDepth[i]);
                        // use original ORB features for object tracking
//                        mvObjKeys_cp[j].push_back(mvKeys[i]);
//                        mvObjPointsDescriptors_cp[j].push_back(mDescriptors.row(i));
//                        mvObjPointDepth_cp[j].push_back(mvDepth[i]);
//                        mvObjKeysUn_cp[j].push_back(mvKeysUn[i]);
//                        mvuObjKeysRight_cp[j].push_back(mvuRight[i]);
//                        if(mvDepth[i] > 0)
//                        {
//                            vnEffectiveObjPointDepthNums[j]++;
//                        }
                        break;
                    }
                }
            }

        }

    }
    // More Dense features for Object area
    for (size_t i = 0; i < mvTempObjKeys.size(); i++)
    {
        int x = mvTempObjKeys[i].pt.x;
        int y = mvTempObjKeys[i].pt.y;

        //Object area: the pixel value is not 0 and not 255
        int object_id_temp = int(mMaskImg.at<u_int8_t>(y, x));
        if (object_id_temp != 0 && object_id_temp != 255)
        {
            for(size_t j=0; j<nObjNum;j++)
            {
                if (!vDetectionObjects[j])
                    continue;
                int id;
                if (vDetectionObjects[j]->mnObjectID>255)
                    id = vDetectionObjects[j]->mnObjectID - 255;
                else id = vDetectionObjects[j]->mnObjectID;
                if (id == object_id_temp - 1) //2D detected object id=mask object id - 1
                {
                    mvObjKeys_cp[j].push_back(mvTempObjKeys[i]);
                    mvObjPointsDescriptors_cp[j].push_back(mTempObjPointsDescriptors.row(i));
                    mvObjPointDepth_cp[j].push_back(mvTempObjDepth[i]);
                    mvObjKeysUn_cp[j].push_back(mvTempObjKeysUn[i]);
                    mvuObjKeysRight_cp[j].push_back(mvuTempObjKeysRight[i]);
                    //if(mvDepth[i] > 0 && mvDepth[i] <mThDepth)
                    if(mvTempObjDepth[i] > 0 && mvTempObjDepth[i]< 2*mThDepth)
                    {
                        vnEffectiveObjPointDepthNums[j]++;
                    }
                    break;
                }
            }
        }

    }

    //cout<<"Features assignment starts： ";
   // cout<<"Frame： "<<mnId<<"   static features number： "<<mvKeys.size();

    if(EnSLOTMode == 1)
        return;

    for(size_t i=0; i<nObjNum;i++)
    {
        if (!vDetectionObjects[i])
            continue;
        size_t nFeaNum = mvObjKeys_cp[i].size();
        size_t nEffectiveDepthFeaNum = vnEffectiveObjPointDepthNums[i];
        DetectionObject* cCuboidTmp = vDetectionObjects[i];
        //cout<<"  Object "<< cCuboidTmp->mnObjectID<<" 2D features number： "<<nFeaNum;
//        if(nFeaNum < EnInitDetObjORBFeaturesNum || nEffectiveDepthFeaNum < 0.8*EnInitDetObjORBFeaturesNum || EobjTrackEndHash.count(cCuboidTmp->mnObjectID)) // 如果点太少 或者 合适点太少 或者 该目标已经跟踪结束
//        {
//            vDetectionObjects[i] = static_cast<DetectionObject*>(NULL);
//            for (int j = 0; j < mvOriKeys_cp[i].size(); ++j) {
//                mvKeys_cp.push_back(mvOriKeys_cp[i][j]);
//                mDescriptors_cp.push_back(mOriDescriptors_cp[i].row(j));
//                mvDepth_cp.push_back(mvOriDepth_cp[i][j]);
//                mvKeysUn_cp.push_back(mvOriKeysUn_cp[i][j]);
//                mvuRight_cp.push_back(mvuOriRight_cp[i][j]);
//            }
//            cout<<RED<<"  Deleted  "<<WHITE;
//            delete cCuboidTmp;
//            continue;
//        }
        mvDetectionObjects.push_back(cCuboidTmp);
        mvObjKeys.push_back(mvObjKeys_cp[i]);
        mvObjPointsDescriptors.push_back(mvObjPointsDescriptors_cp[i]);
        mvObjKeysUn.push_back(mvObjKeysUn_cp[i]);
        mvObjPointDepth.push_back(mvObjPointDepth_cp[i]);
        mvuObjKeysRight.push_back(mvuObjKeysRight_cp[i]);

        mvOriKeys.push_back(mvOriKeys_cp[i]);
        mOriDescriptors.push_back(mOriDescriptors_cp[i]);
        mvOriKeysUn.push_back(mvOriKeysUn_cp[i]);
        mvOriDepth.push_back(mvOriDepth_cp[i]);
        mvuOriRight.push_back(mvuOriRight_cp[i]);

        cCuboidTmp->InitInFrameMapObjectPointsOrder(nFeaNum);
        //cout<<"  Object "<< cCuboidTmp->mnObjectID<<" 2D features number： "<<nFeaNum;
    }

    // background features
    mvKeys = mvKeys_cp;
    mDescriptors = mDescriptors_cp.clone();
    mvDepth = mvDepth_cp;
    mvuRight = mvuRight_cp;
    mvKeysUn = mvKeysUn_cp;
    cout<<endl;
    mnDetObj = mvDetectionObjects.size();
    mvObjKeysGrid.resize(mnDetObj);
    mvpMapObjectPoints.resize(mnDetObj);
    mvbObjKeysOutlier.resize(mnDetObj);
    mvbObjKeysMatchedFlag.resize(mnDetObj);
    mvnNewConstructedObjOrders.reserve(nObjNum);
    mvInLastFrameTrackedObjOrders.reserve(nObjNum);
    mvTotalTrackedObjOrders.reserve(nObjNum);
    mvMapObjects = vector<MapObject*>(mnDetObj, static_cast<MapObject*>(NULL));
    //mvbInsertObjKeyFrameFlags = vector<bool>(mnDetObj, false);
    for(size_t i=0; i<mnDetObj; i++)
    {
        size_t nFeaNum = mvObjKeysUn[i].size();
        mvpMapObjectPoints[i] = (vector<MapObjectPoint *>(nFeaNum, static_cast<MapObjectPoint *>(NULL)));
        mvbObjKeysOutlier[i] = (vector<bool>(nFeaNum, false));
        mvbObjKeysMatchedFlag[i] = (vector<bool>(nFeaNum, false));
        AssignDetObjFeasToGrid(i, nFeaNum);
    }
    cout<<endl;
}



int Frame::FindDetectionObject(DetectionObject* mCuboidTmp)
{
    int order = -1;
    for(size_t i=0; i<this->mvDetectionObjects.size();i++)
    {
        if(this->mvDetectionObjects[i]==NULL)
            continue;
        DetectionObject* cCuboid = this->mvDetectionObjects[i];
        if(cCuboid->mnObjectID == mCuboidTmp->mnObjectID)
        {
            order = i;
            break;
        }
    }
    return order;
}



///Convert MOTS image of kitti dataset to imgMasK image (return value)
///ImgMask: The area with a gray value of 255 is other area, 0 is static area, and 1-50 is area of object (here the object only includes vehicle, and person is also included in other areas),
///Its value represents the object_ ID (is its own instance ID, which has been proved to be consistent with the ID of 2D detetction)
///ImgLabel: only for display
cv::Mat Frame::ReadKittiSegmentationImage(const string &strFolder, const int &nFrameId)
{
    char frame_index_c[256];
    sprintf(frame_index_c,"%06d",nFrameId);
    string MOTS_PNG_Pathname = strFolder + frame_index_c + ".png";

    cv::Mat Img_Init_MOTS = cv::imread(MOTS_PNG_Pathname, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);//cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH

    cv::Mat imgceshi = cv::imread(MOTS_PNG_Pathname,0);


    cv::Mat imgLabel(Img_Init_MOTS.rows, Img_Init_MOTS.cols, CV_8UC3);

    cv::Mat imgMask(imgLabel.rows, imgLabel.cols,CV_8UC1);/// 32位有符号整型单通道
    for(int i=0; i<Img_Init_MOTS.rows;i++)
    {
        for(int j=0;j<Img_Init_MOTS.cols;j++)
        {
            int tmp;
            tmp = Img_Init_MOTS.at<u_int16_t>(i,j);

            if(tmp==10000)
            {

                imgMask.at<u_int8_t>(i,j) = 255;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }

            else if(tmp == 0)
            {
                imgMask.at<u_int8_t>(i,j) = 0;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 255, 255);
            }

            else if(tmp >=1000 && tmp<2000)
            {
                int instance_label = tmp % 1000;
                int instance_label_temp = instance_label+1;

                imgMask.at<u_int8_t>(i,j) = instance_label_temp;

                switch(instance_label)
                {
                    case 0:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,255);
                        break;
                    case 1:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);  // red
                        break;
                    case 2:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);  // blue
                        break;
                    case 3:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,0); // cyan
                        break;
                    case 4:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(47,255,173); // green yellow
                        break;
                    case 5:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 128);
                        break;
                    case 6:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(203,192,255);
                        break;
                    case 7:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(196,228,255);
                        break;
                    case 8:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(42,42,165);
                        break;
                    case 9:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 10:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(245,245,245); // whitesmoke
                        break;
                    case 11:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,165,255); // orange
                        break;
                    case 12:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(230,216,173); // lightblue
                        break;
                    case 13:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128,128,128); // grey
                        break;
                    case 14:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,215,255); // gold
                        break;
                    case 15:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
                        break;
                    case 16:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
                        break;
                    case 17:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 18:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                    case 19:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 20:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                    case 21:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
                        break;
                    case 22:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
                        break;
                    case 23:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
                        break;
                    case 24:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
                        break;
                    case 25:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 255, 127); // chartreuse
                        break;
                    case 26:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(139, 0, 0);  // darkblue
                        break;
                    case 27:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(60, 20, 220);  // crimson
                        break;
                    case 28:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 139);  // darkred
                        break;
                    case 29:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(211, 0, 148);  // darkviolet
                        break;
                    case 30:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 144, 30);  // dodgerblue
                        break;
                    case 31:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(105, 105, 105);  // dimgray
                        break;
                    case 32:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(180, 105, 255);  // hotpink
                        break;
                    case 33:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(204, 209, 72);  // mediumturquoise
                        break;
                    case 34:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(173, 222, 255);  // navajowhite
                        break;
                    case 35:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(143, 143, 188); // rosybrown
                        break;
                    case 36:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(50, 205, 50);  // limegreen
                        break;
                    case 37:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 38:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                    case 39:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 40:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                    case 41:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
                        break;
                    case 42:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
                        break;
                    case 43:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
                        break;
                    case 44:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
                        break;
                    case 45:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
                        break;
                    case 46:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
                        break;
                    case 47:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 48:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                        //case 49:
                        //imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        //break;
                    case 50:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                }
            }

            else
            {
                imgMask.at<u_int8_t>(i,j) = 255;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }

        }
    }

//    cv::imshow("img", imgLabel);
//    cv::waitKey(0);

    return imgMask;
}
cv::Mat Frame::ReadKittiSegmentationImage(const string &strFolder, const int &nFrameId, bool rightseg)
{
    char frame_index_c[256];
    sprintf(frame_index_c,"%06d",nFrameId);
    string MOTS_PNG_Pathname = strFolder + frame_index_c + ".png";

    cv::Mat Img_Init_MOTS = cv::imread(MOTS_PNG_Pathname, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);//cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH

    cv::Mat imgceshi = cv::imread(MOTS_PNG_Pathname,0);


    cv::Mat imgLabel(Img_Init_MOTS.rows, Img_Init_MOTS.cols, CV_8UC3);

    cv::Mat imgMask(imgLabel.rows, imgLabel.cols,CV_8UC1);
    for(int i=0; i<Img_Init_MOTS.rows;i++)
    {
        for(int j=0;j<Img_Init_MOTS.cols;j++)
        {
            int tmp;

            tmp = Img_Init_MOTS.at<u_int16_t>(i,j);
            if(tmp==10000)
            {

                imgMask.at<u_int8_t>(i,j) = 255;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);

                if (rightseg){
                    for (int k = 0; k < 50; ++k) {
                        if (j-k>0)
                            imgMask.at<u_int8_t>(i,j-k) = 255;
                        if (j+k<imgMask.cols)
                            imgMask.at<u_int8_t>(i,j+k) = 255;
                    }
                }
            }

            else if(tmp == 0)
            {
                imgMask.at<u_int8_t>(i,j) = 0;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 255, 255);
            }

            else if(tmp >=1000 && tmp<2000)
            {
                int instance_label = tmp % 1000;
                int instance_label_temp = instance_label+1;

                imgMask.at<u_int8_t>(i,j) = instance_label_temp;
                if (rightseg){
                    for (int k = 0; k < 50; ++k) {
                        if (j-k>0)
                            imgMask.at<u_int8_t>(i,j-k) = instance_label_temp;
                        if (j+k<imgMask.cols)
                            imgMask.at<u_int8_t>(i,j+k) = instance_label_temp;
                    }
                }

                switch(instance_label)
                {
                    case 0:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,255);
                        break;
                    case 1:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);  // red
                        break;
                    case 2:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);  // blue
                        break;
                    case 3:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,0); // cyan
                        break;
                    case 4:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(47,255,173); // green yellow
                        break;
                    case 5:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 128);
                        break;
                    case 6:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(203,192,255);
                        break;
                    case 7:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(196,228,255);
                        break;
                    case 8:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(42,42,165);
                        break;
                    case 9:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 10:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(245,245,245); // whitesmoke
                        break;
                    case 11:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,165,255); // orange
                        break;
                    case 12:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(230,216,173); // lightblue
                        break;
                    case 13:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128,128,128); // grey
                        break;
                    case 14:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,215,255); // gold
                        break;
                    case 15:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
                        break;
                    case 16:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
                        break;
                    case 17:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 18:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                    case 19:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 20:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                    case 21:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
                        break;
                    case 22:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
                        break;
                    case 23:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
                        break;
                    case 24:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
                        break;
                    case 25:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 255, 127); // chartreuse
                        break;
                    case 26:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(139, 0, 0);  // darkblue
                        break;
                    case 27:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(60, 20, 220);  // crimson
                        break;
                    case 28:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 139);  // darkred
                        break;
                    case 29:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(211, 0, 148);  // darkviolet
                        break;
                    case 30:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 144, 30);  // dodgerblue
                        break;
                    case 31:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(105, 105, 105);  // dimgray
                        break;
                    case 32:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(180, 105, 255);  // hotpink
                        break;
                    case 33:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(204, 209, 72);  // mediumturquoise
                        break;
                    case 34:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(173, 222, 255);  // navajowhite
                        break;
                    case 35:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(143, 143, 188); // rosybrown
                        break;
                    case 36:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(50, 205, 50);  // limegreen
                        break;
                    case 37:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 38:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                    case 39:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        break;
                    case 40:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                    case 41:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(225, 228, 255);  // mistyrose
                        break;
                    case 42:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(128, 0, 0);  // navy
                        break;
                    case 43:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(35, 142, 107);  // olivedrab
                        break;
                    case 44:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(45, 82, 160);  // sienna
                        break;
                    case 45:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(30,105,210); // chocolate
                        break;
                    case 46:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,255,0);  // green
                        break;
                    case 47:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(34, 34, 178);  // firebrick
                        break;
                    case 48:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(240, 255, 240);  // honeydew
                        break;
                        //case 49:
                        //imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(250, 206, 135);  // lightskyblue
                        //break;
                    case 50:
                        imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                        break;
                }
            }

            else
            {
                imgMask.at<u_int8_t>(i,j) = 255;
                imgLabel.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }

        }
    }

    return imgMask;
}



///Read the virtual kitti semantic segmentation image.
/// The pixel value=trackID+1. If it is 0, it means it is not a car (temporarily considered static)
cv::Mat Frame::ReadVirtualKittiSegmentationImage(const string &strFolder, const int &nFrameId)
{
    char frame_index_c[256];
    sprintf(frame_index_c,"%06d",nFrameId);
    string MOTS_PNG_Pathname = strFolder + frame_index_c + ".pgm";
    cv::Mat Img_Init_MOTS = cv::imread(MOTS_PNG_Pathname, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH);//cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH
    return Img_Init_MOTS;
}

cv::Mat Frame::ReadVirtualKittiForwardOpticalFlow(const string &strFolder, const int &nFrameID)
{
    char frame_index_c[256];
    sprintf(frame_index_c,"%05d",nFrameID);
    string Folder = strFolder + "/Camera_0/flow_"+ frame_index_c +".png";

    cv::Mat imgForwardOpticalInit = cv::imread(Folder, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);//cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH

    int width = imgForwardOpticalInit.cols;
    int height = imgForwardOpticalInit.rows;

    double flow_u = 0;
    double flow_v = 0;

    cv::Mat imgForwardOpticalFinal(imgForwardOpticalInit.rows, imgForwardOpticalInit.cols, CV_64FC2);
    cv::Vec3w x;

    for(int i=0; i< height; i++)
    {
        for(int j=0; j<width; j++)
        {
            x = imgForwardOpticalInit.at<cv::Vec3w>(i,j);
            if(x(0)==0)
            {
                flow_u = 0;
                flow_v = 0;
            }
            // 计算光流
            else{
                flow_u = (2.0 /(pow(2,16) -1.0) * x(2) -1)*(width-1);
                flow_v = (2.0 /(pow(2,16) -1.0) * x(1) -1)*(height-1);
            }
            imgForwardOpticalFinal.at<cv::Vec2d>(i,j) = cv::Vec2d(flow_u, flow_v);
        }
    }
    return imgForwardOpticalFinal;
}


vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};


cv::Ptr<cv::Tracker> createTrackerByName(string trackerType)
{
    cv::Ptr<cv::Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = cv::TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = cv::TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = cv::TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = cv::TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = cv::TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = cv::TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = cv::TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = cv::TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}


void Frame::Online2DObjectTracking(cv::MultiTracker* multiTracker, vector<cv::Ptr<cv::Tracker>> vTrackers, vector<DetectionObject*>& vDetectionObjects)
{
    cv::Mat frame;
    if(mnId == 0) // First frame
    {
        frame = mRawImg;

        vector<cv::Rect> bboxes;
        cv::selectROIs("MultiTracker", frame, bboxes, true, false); // showCrosshair, fromCenter

        // 等选完结束了再开始viewer线程
        if(EbStartViewerWith2DTracking == false)
            EbStartViewerWith2DTracking = true;


        if(bboxes.size() < 1)
        {
            cout<<"There is no tracking object!"<<endl;
            assert(0);
        }
        string trackerType = "CSRT";// {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
        for(size_t i=0; i < bboxes.size(); i++)
            multiTracker->add(vTrackers[i], frame, cv::Rect2d(bboxes[i]));
        for(size_t i=0; i<bboxes.size(); i++)
        {
            Eigen::Vector3d scale = EeigUniformObjScale;
            Eigen::Vector3d initPosition = EeigInitPosition;
            Eigen::Vector3d initRotation = EeigInitRotation;
            DetectionObject *raw_cuboid = new DetectionObject(mnId, i+3, bboxes[i], scale, initPosition, initRotation);
            vDetectionObjects.push_back(raw_cuboid);
        }
    }
    else{
        frame =mRawImg;
        multiTracker->update(frame);
        for(unsigned i=0; i< multiTracker->getObjects().size(); i++)
        {

            Eigen::Vector3d scale = EeigUniformObjScale;
            Eigen::Vector3d initPosition = Eigen::Vector3d::Zero();
            Eigen::Vector3d initRotation = Eigen::Vector3d::Zero();
            DetectionObject *raw_cuboid = new DetectionObject(mnId, i+3, multiTracker->getObjects()[i], scale, initPosition, initRotation);
            vDetectionObjects.push_back(raw_cuboid);
        }
    }
}

void Frame::OfflineDetectObject(vector<DetectionObject*>& vDetectionObjects)
{
    vDetectionObjects.clear();

    if(EvOfflineAllObjectDetections.size() == 0)
    {
        cout<<"Object tracking preprocessing information was not read!"<<endl;
        exit(0);
    }

    std::vector<Eigen::Matrix<double, 1, 24>> pred_frame_objects;
    pred_frame_objects = EvOfflineAllObjectDetections[ORB_SLAM2::EnStartFrameId];

    vDetectionObjects.reserve(pred_frame_objects.size());
    for (size_t i = 0; i < pred_frame_objects.size(); i++)
    {
        //Eigen::Matrix<double, 1, 24> One_object_temp;
        auto One_object_temp = pred_frame_objects[i];
        int type_id_temp = int(One_object_temp [17]);
        int truth_id = int(One_object_temp[1]);


        switch(EnSLOTMode)
        {
            case 3:
            {
                if(type_id_temp == 1)
                {
                    DetectionObject *raw_cuboid = new DetectionObject(mnId, One_object_temp);
                    vDetectionObjects.push_back(raw_cuboid);
                }
                break;
            }

            case 2:
            {
                if(truth_id == ORB_SLAM2::EnSelectTrackedObjId)
                {
                    DetectionObject *raw_cuboid = new DetectionObject(mnId, One_object_temp);
                    vDetectionObjects.push_back(raw_cuboid);
                }
                break;
            }

            case 4:
            {
                if(type_id_temp == 1)
                //if(truth_id == ORB_SLAM2::EnSelectTrackedObjId && type_id_temp == 1)
                {
                    DetectionObject *raw_cuboid = new DetectionObject(mnId, One_object_temp);
                    vDetectionObjects.push_back(raw_cuboid);
                }
                break;
            }
        }

    }
}


void Frame::AssignFeaturesToGrid()
{

    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++) {

            mGrid[i][j].clear();
            mGrid[i][j].reserve(nReserve);
        }

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}


void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    mSETcw = Converter::toSE3Quat(Tcw);

    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;

    mTwc = cv::Mat::eye(4, 4, mTcw.type());
    mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
    mOw.copyTo(mTwc.rowRange(0, 3).col(3));
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates

    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}


bool Frame::isInFrustum(MapObjectPoint *pMP, const size_t &nOrder, float viewingCosLimit)
{
    MapObject* pMO = mvMapObjects[nOrder];
    if(pMO==NULL)
        assert(0);
    pMP->mbTrackInView = false;

    cv::Mat Poj = pMP->GetInObjFramePosition();
    g2o::SE3Quat Tco = pMO->GetCFInFrameObjState(mnId).pose;
    const cv::Mat Pcj = Converter::toCvMat(Tco * Converter::toVector3d(Poj));

    const float &PcjX = Pcj.at<float>(0);
    const float &PcjY= Pcj.at<float>(1);
    const float &PcjZ = Pcj.at<float>(2);
    if(PcjZ<0)
        return false;
    const float invz = 1.0f/PcjZ;
    const float u=fx*PcjX*invz+cx;
    const float v=fy*PcjY*invz+cy;
    if(!this->isInBBox(nOrder, u,v)) // 判断投影点是否在对应BBox内
        return false;

    //g2o::ObjectState gObjState = pMO->GetCFInFrameObjState(mnId); //求出oPcj
    cv::Mat Poc = Converter::toCvMat(Tco.inverse().translation());
    const cv::Mat oPcj = Poj-Poc;

    const float dist = cv::norm(oPcj); // oPcj是否满足距离要求
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    if(dist<minDistance || dist>maxDistance)
        return false;

    cv::Mat Pn = pMP->GetNormal(); // oPcj是否满足方向要求
    const float viewCos = oPcj.dot(Pn)/dist;
    if(viewCos<viewingCosLimit)
        return false;

    const int nPredictedLevel = pMP->PredictScale(dist,this);
    pMP->mbTrackInView = true;

    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;

    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

bool Frame::isInBBox(const size_t &nOrder, const float &u, const float &v)
{
    DetectionObject* pDet = mvDetectionObjects[nOrder];
    if(pDet==NULL)
        assert(0);
    double minX = pDet->mrectBBox.x;
    double minY = pDet->mrectBBox.y;
    double maxX = minX + pDet->mrectBBox.width;
    double maxY = minY + pDet->mrectBBox.height;
    return (u>=minX && u<maxX && v>=minY && v<maxY);
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

void Frame::AssignDetObjFeasToGrid(const size_t &objOrder, const size_t &objFeaNum)
{
    int nReserve = 0.5f*objFeaNum/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    mvObjKeysGrid[objOrder].resize(FRAME_GRID_COLS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
    {
        mvObjKeysGrid[objOrder][i].resize(FRAME_GRID_ROWS);
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
        {
            if(nReserve == 0)
                mvObjKeysGrid[objOrder][i][j].reserve(1);
            else
                mvObjKeysGrid[objOrder][i][j].reserve(nReserve);
        }
    }



    for(size_t i=0;i<objFeaNum;i++)
    {
        const cv::KeyPoint &kp = mvObjKeysUn[objOrder][i];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mvObjKeysGrid[objOrder][nGridPosX][nGridPosY].push_back(i);
    }
}



vector<size_t> Frame::GetObjectFeaturesInArea(const int& nDetObjOrder, const size_t& nDetObjFeaNum, const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(nDetObjFeaNum);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));

    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));

    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;


    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {

        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mvObjKeysGrid[nDetObjOrder][ix][iy];
            if(vCell.empty())
                continue;


            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvObjKeysUn[nDetObjOrder][vCell[j]];

                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}


int Frame::GetCloestFeaturesInArea(const int& nDetObjOrder, const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    int vIndices = -1;

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));

    if (nMaxCellX < 0)
        return vIndices;


    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;


    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);
    float min_dist = -1;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mvObjKeysGrid[nDetObjOrder][ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {

                const cv::KeyPoint &kpUn = mvObjKeysUn[nDetObjOrder][vCell[j]];

                if (bCheckLevels)
                {
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }


                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;


                if (fabs(distx) < r && fabs(disty) < r)
                {
                    float dist = distx * distx + disty * disty;

                    if (min_dist == -1 || dist < min_dist)
                    {
                        min_dist = dist;
                        vIndices = vCell[j];
                    }
                }
            }
        }
    }

    return vIndices;
}




bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}
void Frame::UndistortObjKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvTempObjKeysUn=mvTempObjKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(mvTempObjKeys.size(),2,CV_32F);
    for(int i=0; i<mvTempObjKeys.size(); i++)
    {
        mat.at<float>(i,0)=mvTempObjKeys[i].pt.x;
        mat.at<float>(i,1)=mvTempObjKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvTempObjKeysUn.resize(mvTempObjKeys.size());
    for(int i=0; i<mvTempObjKeys.size(); i++)
    {
        cv::KeyPoint kp = mvTempObjKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvTempObjKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

void Frame::ComputeObjStereoMatches()
{
    if (mvTempObjKeys.size()<=0) return;
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        auto &vObjKeys= mvTempObjKeys;
        auto &vObjKeysRight =  mvTempObjKeysRight;
        auto &vObjKeysUn = mvTempObjKeysUn;
        vector<float> &vuObjRight = mvuTempObjKeysRight;
        vector<float> &vObjDepth = mvTempObjDepth;
        auto &vObjDescriptors = mTempObjPointsDescriptors;
        auto &vObjDescriptorsRight = mTempObjPointsDescriptorsRight;

        vuObjRight = vector<float>(vObjKeys.size(),-1.0f);
        vObjDepth = vector<float>(vObjKeys.size(),-1.0f);
        vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
        for(int i=0; i<nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = vObjKeysRight.size();

        for(int iR=0; iR<Nr; iR++)
        {
            const cv::KeyPoint &kp = vObjKeysRight[iR];
            const float &kpY = kp.pt.y;

            const float r = 2.0f*mvScaleFactors[vObjKeysRight[iR].octave];
            const int maxr = ceil(kpY+r);
            const int minr = floor(kpY-r);

            for(int yi=minr;yi<=maxr;yi++)
                vRowIndices[yi].push_back(iR);
        }

        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf/minZ;

        // For each left keypoint search a match in the right image
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(vObjKeys.size());

        for(int iL=0; iL<int(vObjKeys.size()); iL++)
        {
            const cv::KeyPoint &kpL = vObjKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if(vCandidates.empty())
                continue;

            const float minU = uL-maxD;
            const float maxU = uL-minD;

            if(maxU<0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = vObjDescriptors.row(iL);
            // Compare descriptor to right keypoints

            for(size_t iC=0; iC<vCandidates.size(); iC++)
            {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = vObjKeysRight[iR];

                if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                    continue;

                const float &uR = kpR.pt.x;

                if(uR>=minU && uR<=maxU)
                {
                    const cv::Mat &dR = vObjDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }
            // Subpixel match by correlation
            if(bestDist<thOrbDist)
            {
                // coordinates in image pyramid at keypoint scale

                const float uR0 = vObjKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x*scaleFactor);
                const float scaledvL = round(kpL.pt.y*scaleFactor);
                const float scaleduR0 = round(uR0*scaleFactor);

                // sliding window search
                const int w = 5;

                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
                IL.convertTo(IL,CV_32F);

                IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2*L+1);

                const float iniu = scaleduR0+L-w;
                const float endu = scaleduR0+L+w+1;
                if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                {
                    continue;
                }

                for(int incR=-L; incR<=+L; incR++)
                {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                    IR.convertTo(IR,CV_32F);
                    IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                    float dist = cv::norm(IL,IR,cv::NORM_L1);
                    if(dist<bestDist)
                    {
                        bestDist =  dist;
                        bestincR = incR;
                    }

                    vDists[L+incR] = dist;
                }

                if(bestincR==-L || bestincR==L)
                {
                    continue;
                }

                const float dist1 = vDists[L+bestincR-1];
                const float dist2 = vDists[L+bestincR];
                const float dist3 = vDists[L+bestincR+1];

                const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

                if(deltaR<-1 || deltaR>1)
                {
                    continue;
                }
                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

                float disparity = (uL-bestuR);

                if(disparity>=minD && disparity<maxD)
                {
                    if(disparity<=0)
                    {
                        disparity=0.01;
                        bestuR = uL-0.01;
                    }
                    vObjDepth[iL]=mbf/disparity;
                    vuObjRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int,int>(bestDist,iL));
                }
            }
        }
        sort(vDistIdx.begin(),vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size()/2].first;
        const float thDist = 1.5f*1.4f*median;

        for(int i=vDistIdx.size()-1;i>=0;i--)
        {
            if(vDistIdx[i].first<thDist)
                break;
            else
            {
                vuObjRight[vDistIdx[i].second]=-1;
                vObjDepth[vDistIdx[i].second]=-1;
            }
        }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

cv::Mat Frame::UnprojectStereodynamic(const size_t& nDetObjOrder, const int &i, bool flag_world_frame)
{
    const float z = mvObjPointDepth[nDetObjOrder][i];
    if(z>0)
    {
        const float u = mvObjKeysUn[nDetObjOrder][i].pt.x;
        const float v = mvObjKeysUn[nDetObjOrder][i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        //cout<<x3Dc<<endl;

        if(flag_world_frame==1)
        {
            return mRwc*x3Dc+mOw;
        }
        else{
            return x3Dc;
        }
    }
    else
        return cv::Mat();
}

void Frame::DetectYOLO(cv::Mat &imColor, vector<DetectionObject *> &vDetectionObjects,Detector* YOLODetector, DS::DeepSort* deepSort, cv::Mat left, cv::Mat right) {

    auto t1 = std::chrono::steady_clock::now();

    std::vector<std::vector<Detection>> result = YOLODetector->Run(imColor, EfConfThres, EfIouThres);
    std::vector<DS::DetectBox> det; // Detection -> DS::DetectBox
    if(result.size() != 0)
    {
        det.reserve(result[0].size());
        for (auto &m: result[0])
        {
            if(m.class_idx == 2 || m.class_idx == 7 ) // car 或 truck
            {
                // Detection -> DetectBox
                DS::DetectBox temp(m.bbox.x, m.bbox.y, m.bbox.x+m.bbox.width, m.bbox.y+m.bbox.height, m.score, m.class_idx);
                det.push_back(temp);
            }
        }
    }

    deepSort->sort(imColor, det);
    for(auto &temp: det) // 得到vDetectionObjects
    {

        DS::DetectBox temp1 = temp;
        //To prevent the temp from exceeding the image range, the temp needs to be modified
        temp.x1 = max(float(0), temp1.x1);
        temp.y1 = max(float(0), temp1.y1);
        temp.x2 = min(float(mRawImg.cols-1), temp1.x2);
        temp.y2 = min(float(mRawImg.rows-1), temp1.y2);


        cv::Rect bbox(temp.x1, temp.y1, temp.x2- temp.x1, temp.y2-temp.y1);
        bool initflag=0;
        Eigen::Vector3d scale = EeigUniformObjScale;
        Eigen::Vector3d initPosition = Eigen::Vector3d::Zero();
        Eigen::Vector3d initRotation = Eigen::Vector3d::Zero();

        if(1)
        OfflineObjectPoseInit(bbox,&initPosition,&initRotation,&scale);
        if(initPosition[0]!=0) initflag=1;
        DetectionObject *raw_cuboid = new DetectionObject(mnId, temp.trackID, bbox, scale, initPosition, initRotation);
        raw_cuboid->mInitflag = initflag;
        vDetectionObjects.push_back(raw_cuboid);
    }
    cout<<endl;



    int th1 = 5, th2 = 5, th3 = 5, th4 = 5;
    for(size_t i=0; i < vDetectionObjects.size(); i++)
    {
        cv::Rect tmp = vDetectionObjects[i]->mrectBBox;
        if (tmp.width<th3+th1 || tmp.height<th2+th4){
            vDetectionObjects[i] =  static_cast<DetectionObject*>(NULL);
            continue;
        }
        int id = vDetectionObjects[i]->mnObjectID;
        if (id>255)
            id = id - 255;
        cv::Mat m0 = mMaskImg(cv::Rect(tmp.x, tmp.y, tmp.width, tmp.height));
        m0.setTo(255);
        cv::Mat m1 = mMaskImg(cv::Rect(tmp.x + th1, tmp.y + th2, tmp.width-th3-th1, tmp.height - th2 - th4));
        m1.setTo(id+1);
        int x_right = (tmp.x-80>0)? tmp.x-80 : 0;
        int width_right =(tmp.width+70+x_right<mMaskImgRight.cols)? tmp.width+70 : mMaskImgRight.cols-x_right;
        cv::Mat m2 = mMaskImgRight(cv::Rect(x_right, tmp.y, width_right, tmp.height));
        m2.setTo(255);
        cv::Mat m3 = mMaskImgRight(cv::Rect(x_right + th1, tmp.y + th2, width_right-th3-th1, tmp.height - th2 - th4));
        m3.setTo(id+1);
    }

    ExtractObjORB(left, right, vDetectionObjects);
    auto t2 = std::chrono::steady_clock::now();
    cout<<"Object Dense Points extraction: "<<std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count()<<endl;
}

void OpencvORBDetector(cv::Mat im, cv::Mat ObjMask, vector<cv::KeyPoint> &kp, cv::Mat &descriptor){
    //输入参数：nfeatures, scalefactor,nlevels,edgethreshold
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1000, 1.2, 8, 19);
    detector->detectAndCompute(im, ObjMask, kp, descriptor);
}
void Frame::ExtractObjORB(cv::Mat left, cv::Mat right, vector<DetectionObject*> &vDetectionObjects) {
    if (!vDetectionObjects.size()>0) return;
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;

    cv::Mat LeftObjMask = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    cv::Mat RightObjMask = cv::Mat::zeros(mRawImg.rows, mRawImg.cols, CV_8UC1);
    for (size_t m = 0; m < mMaskImg.rows; m++) {
        for (size_t n = 0; n < mMaskImg.cols; n++) {
            if (mMaskImg.at<uchar>(m, n) != 0 && mMaskImg.at<uchar>(m, n) != 255) {
                LeftObjMask.at<uchar>(m, n) = 255;
            }
            if (mMaskImgRight.at<uchar>(m, n) != 0 && mMaskImgRight.at<uchar>(m, n) != 255) {
                RightObjMask.at<uchar>(m, n) = 255;
            }
        }
    }
//    detector->detect(left,keypoints_1,LeftObjMask);
//    selectMax(3,left,keypoints_1);
//    detector->compute(left,keypoints_1,descriptors_1);
    thread threadOpencvORBLeft(OpencvORBDetector,left, LeftObjMask, ref(keypoints_1), ref(descriptors_1));
    thread threadOpencvORBRight(OpencvORBDetector,right, RightObjMask, ref(keypoints_2), ref(descriptors_2));
    threadOpencvORBLeft.join();
    threadOpencvORBRight.join();
//    cv::Mat showimg(LeftObjMask);
//    cv::drawKeypoints(LeftObjMask, keypoints_1, showimg);
//    cv::imshow("ObjectKeys", showimg);
//    cv::waitKey(0);
//    cv::drawKeypoints(RightObjMask, keypoints_2, showimg);
//    cv::imshow("ObjectKeysRight",showimg);
//    cv::waitKey(0);

    mvTempObjKeys = keypoints_1;
    mvTempObjKeysRight = keypoints_2;
    mTempObjPointsDescriptors = descriptors_1.clone();
    mTempObjPointsDescriptorsRight = descriptors_2.clone();

}

float bbox_IoU(cv::Rect a, cv::Rect b){

    int xA = max(a.x,b.x);
    int yA = max(a.y,b.y);
    int xB = min(a.x+a.width,b.x+b.width);
    int yB = min(a.y+a.height,b.y+b.height);

    int interArea = (xB-xA+1)*(yB-yA+1);
    if (interArea<=0) return 0;

    int aArea = a.area();
    int bArea = b.area();

    float iou = float(interArea) / float(aArea+bArea-interArea);

    return iou;
}
void Frame::OfflineObjectPoseInit(cv::Rect YOLODet, Eigen::Vector3d *initPosition, Eigen::Vector3d *initRotation, Eigen::Vector3d *scale) {

    if(EvOfflineAllObjectDetections.empty())
    {
        cout<<"Object tracking preprocessing information was not read!"<<endl;
        exit(0);
    }

    std::vector<Eigen::Matrix<double, 1, 24>> pred_frame_objects;
    pred_frame_objects = EvOfflineAllObjectDetections[ORB_SLAM2::EnStartFrameId];
    float best_iou = 0;
    Eigen::Matrix<double,1,24> best_obj;
    for (auto obj:pred_frame_objects) {
        if (obj[17]!=1) continue;
        cv::Rect offlineBox = cv::Rect(obj[5], obj[6], obj[7], obj[8]);
        float iou = bbox_IoU(YOLODet,offlineBox);
        cout<<iou<<" ";
        if (iou>best_iou){
            best_iou = iou;
            best_obj = obj;
        }
    }
    cout<<"    ";
    if (best_iou>0.6){
        *initPosition = Eigen::Vector3d(best_obj[12], best_obj[13], best_obj[14]);
        *initRotation = Eigen::Vector3d(best_obj[19], best_obj[15], best_obj[19]);
        *scale = Eigen::Vector3d (best_obj[9], best_obj[10], best_obj[11]);
    }

}
} //namespace ORB_SLAM
