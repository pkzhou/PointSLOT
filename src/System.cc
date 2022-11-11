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
#include "System.h"
#include "Converter.h"
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrame.h"
#include "ObjectLocalMapping.h"
#include "MapObject.h"
#include "DetectionObject.h"
#include "ObjectKeyFrame.h"


#include <unistd.h>
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";



    if(mSensor!=STEREO)
        assert(0);

    cout<<strSettingsFile<<endl;
    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    //bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    bool bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);


    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    // object localmapping thread
    if(EnSLOTMode == 2 || EnSLOTMode == 3 ||EnSLOTMode == 4)
    {
        mpObjectLocalMapping = new ObjectLocalMapping();
        mptObjectLocalMapping= new thread(&ObjectLocalMapping::Run, mpObjectLocalMapping);
    }

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    if(1)
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
        //Initialize the Viewer thread and launch
    if (bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }
    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);
    mpTracker->SetObjectLocalMapper(mpObjectLocalMapping);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    if(EnSLOTMode == 2 || EnSLOTMode == 3 ||EnSLOTMode == 4)
        mpObjectLocalMapping->SetTracker(mpTracker);



    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4)
        mpObjectLocalMapping->RequestFinish();

    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA() || ((EnSLOTMode == 2 || EnSLOTMode == 3 || EnSLOTMode == 4) && !mpObjectLocalMapping->isFinished()))
    {
        usleep(5000);
    }

//    if(mpViewer)
//        pangolin::BindToContext("SLOT1: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(15) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(15) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();




    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw; // 相机， 此时目标是多少？
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);




        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;


    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}
void System::SaveObjectDetectionKITTI(const string &OTPath){
    vector<MapObject*> vMapObjs = mpTracker->AllObjects;
    if(vMapObjs.size() == 0)
        assert(0);

    // kitti上目标很多，并且不是每一帧都会有目标，
    // 策略是遍历每一帧，还是遍历每一个3D目标
    // 结果是需要每一帧的检测结果
    // 但是没办法遍历每一帧，只能遍历每一个3D目标
    // 每个3D目标

    for(size_t i=0; i<mpTracker->mlFrameTimes.size(); i++) // 遍历所有图像写文件
    {
        char frame_id[16];
        sprintf(frame_id, "%06d", i);
        string filename = OTPath + frame_id + ".txt";
        ofstream fo;
        fo.open(filename.c_str());
        // fixed 常和setprecision()结合使用，fixed不用科学计算法来表示一个数，那么在输出的时候就用std::fixed来看更细的值
        // 而setprecision(2)就代表在小数点后面保留几位有效数字
        fo.close();
    }
    for(size_t i=0; i<vMapObjs.size(); i++) {
        MapObject *mOb = vMapObjs[i];
        if (mOb == NULL)
            assert(0);
        std::map<ObjectKeyFrame *, g2o::ObjectState> mOKF = mOb->GetCFInAllKFsObjStates();
        std::map<long unsigned int, pair<ObjectKeyFrame*, g2o::SE3Quat>> mRelPose = mOb->mlRelativeFramePoses;
        for(auto lit=mRelPose.begin(), lend=mRelPose.end();lit!=lend;lit++)
        {
            int frameid = lit->first;
            DetectionObject* mDet =  mOb->mmmDetections[frameid];
            auto package = lit->second;
            ObjectKeyFrame* pKF = package.first;
            g2o::SE3Quat RelPose = package.second;
            g2o::SE3Quat KFPose = Converter::toSE3Quat(pKF->GetPose());
            g2o::SE3Quat CurrentPose = RelPose * KFPose; //recover from relative pose
            cv::Mat Tco = Converter::toCvMat(CurrentPose);
            cv::Mat tco = Tco.rowRange(0,3).col(3);

            float roll = CurrentPose.toXYZPRYVector()[3];
            float pitch = CurrentPose.toXYZPRYVector()[4];
            float yaw = CurrentPose.toXYZPRYVector()[5];

            char sframeid[16];
            sprintf(sframeid, "%06d", frameid);
            string filename = OTPath + sframeid +".txt";
            ofstream fo;
            fo.open(filename.c_str(),ios::app);

            // 一定都是车吗？
            // 包括：type truncated occluded alpha bbox(left top right bottom) dimensions location rotation_y score
            // 3D bounding box 测试需要的是h w l t ry

            fo<<"Car"<<" "<< float(mDet->mdTruncated/2) << " "<<float(mDet->mdOcculuded)<<" "<< float(mDet->mdAlpha) <<
              " "<<mDet->mrectBBox.x <<" "<<mDet->mrectBBox.y<<" "<<mDet->mrectBBox.x + mDet->mrectBBox.width<<" "<<
              mDet->mrectBBox.y+mDet->mrectBBox.height<<" "<<mDet->mScale[1]<<" "<<mDet->mScale[2]<<" "<<mDet->mScale[0]<< // height(y1) width(z2) length(x0)
              " "<<tco.at<float>(0)<<" "<<tco.at<float>(1)+mDet->mScale[1]/2<<" "<<tco.at<float>(2)<<" "<<pitch<<" "<<1<<endl;


            fo.close();
        }
    }
    cout << endl << "object detection results saved!" << endl;
}
void System::SaveObjectDetectionResultsInCameraFrame(const string &OTPath){
    vector<MapObject*> vMapObjs = mpTracker->AllObjects;
    if(vMapObjs.size() == 0)
        assert(0);

    // kitti上目标很多，并且不是每一帧都会有目标，
    // 策略是遍历每一帧，还是遍历每一个3D目标
    // 结果是需要每一帧的检测结果
    // 但是没办法遍历每一帧，只能遍历每一个3D目标
    // 每个3D目标

    for(size_t i=0; i<mpTracker->mlFrameTimes.size(); i++) // 遍历所有图像写文件
    {
        char frame_id[16];
        sprintf(frame_id, "%06d", i);
        string filename = OTPath + frame_id + ".txt";
        ofstream fo;
        fo.open(filename.c_str());
        // fixed 常和setprecision()结合使用，fixed不用科学计算法来表示一个数，那么在输出的时候就用std::fixed来看更细的值
        // 而setprecision(2)就代表在小数点后面保留几位有效数字
        fo.close();
    }

    for(size_t i=0; i<vMapObjs.size(); i++)
    {
        MapObject* mOb = vMapObjs[i];
        if(mOb == NULL)
            assert(0);

        // 写一个迭代器
        // FIXME 没有用mlRelativeFramePoses！！！
        std::map<long unsigned int, g2o::ObjectState> ObjPoses = mOb->GetCFInAllFrameObjStates();
        for(std::map<long unsigned int, g2o::ObjectState>::iterator it = ObjPoses.begin(); it!= ObjPoses.end(); it++)
        {
            int frameid = it->first;
            if(!mOb->mmmDetections.count(frameid))
                assert(0);
            DetectionObject* mDet =  mOb->mmmDetections[frameid];
            cv::Mat Tco = Converter::toCvMat(it->second.pose);
            cv::Mat tco = Tco.rowRange(0,3).col(3);
            // ry 该怎么求，应该是把旋转转成欧拉角，然后取沿着y轴方向的即可
            // 欧拉角和四元数之间不是太懂，后面看看
            float roll = it->second.pose.toXYZPRYVector()[3];
            float pitch = it->second.pose.toXYZPRYVector()[4];
            float yaw = it->second.pose.toXYZPRYVector()[5];

            char sframeid[16];
            sprintf(sframeid, "%06d", frameid);
            string filename = OTPath + sframeid +".txt";
            ofstream fo;
            fo.open(filename.c_str(),ios::app);

            // 一定都是车吗？
            // 包括：type truncated occluded alpha bbox(left top right bottom) dimensions location rotation_y score
            // 3D bounding box 测试需要的是h w l t ry

            fo<<"Car"<<" "<< float(mDet->mdTruncated/2) << " "<<float(mDet->mdOcculuded)<<" "<< float(mDet->mdAlpha) <<
              " "<<mDet->mrectBBox.x <<" "<<mDet->mrectBBox.y<<" "<<mDet->mrectBBox.x + mDet->mrectBBox.width<<" "<<
              mDet->mrectBBox.y+mDet->mrectBBox.height<<" "<<mDet->mScale[1]<<" "<<mDet->mScale[2]<<" "<<mDet->mScale[0]<< // height(y1) width(z2) length(x0)
              " "<<tco.at<float>(0)<<" "<<tco.at<float>(1)<<" "<<tco.at<float>(2)<<" "<<pitch<<" "<<1<<endl;


            fo.close();

        }
    }
    cout << endl << "object detection results saved!" << endl;

}

void System::SaveTrajectoryKITTICameraAndObject(const string &filenameCamera, const string &filenameObject)
    {
        cout << endl << "Saving camera trajectory to " << filenameCamera << " ..." << endl;
        if(mSensor==MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filenameCamera.c_str());
        f << fixed;

        ofstream fo;
        fo.open(filenameObject.c_str());
        fo << fixed;


        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();

        vector<MapObject*> vMapObjs = mpTracker->AllObjects;
        if(vMapObjs.size() == 0)
            assert(0);
        MapObject* mOb = vMapObjs[0];


        int frame_id = 0;

        for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
        {
            ORB_SLAM2::KeyFrame* pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

            while(pKF->isBad())
            {
                //  cout << "bad parent" << endl;
                Trw = Trw*pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw*pKF->GetPose()*Two;

            cv::Mat Tcw = (*lit)*Trw; // 相机， 此时目标是多少？
            cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
            cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);





            f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
              Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
              Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;



            if(frame_id !=0)
            {
                cv::Mat Tow = Converter::toCvMat(mOb->GetCFInFrameObjState(frame_id).pose) * Tcw;
                cv::Mat Rwo = Tow.rowRange(0,3).colRange(0,3).t();
                cv::Mat two = -Rwo*Tow.rowRange(0,3).col(3);

                fo << setprecision(9) << Rwo.at<float>(0,0) << " " << Rwo.at<float>(0,1)  << " " << Rwo.at<float>(0,2) << " "  << two.at<float>(0) << " " <<
                   Rwo.at<float>(1,0) << " " << Rwo.at<float>(1,1)  << " " << Rwo.at<float>(1,2) << " "  << two.at<float>(1) << " " <<
                   Rwo.at<float>(2,0) << " " << Rwo.at<float>(2,1)  << " " << Rwo.at<float>(2,2) << " "  << two.at<float>(2) << endl;
            }
            frame_id++;

        }
        f.close();
        fo.close();
        cout << endl << "trajectory saved!" << endl;
    }





int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
