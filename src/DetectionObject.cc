//
// Created by liuyuzhen on 2021/3/28.
//
#include "DetectionObject.h"
#include "Parameters.h"
#include "Converter.h"
using namespace  std;
namespace ORB_SLAM2
{
    DetectionObject::DetectionObject(const int &pFrameId, const int &nObjectID, const bool &bDynamicFlag, const double &dMonoDynaVal,
            const double &dStereoDynaVal, const double &dTruncated, const double &dOcculuded, const double &dAlpha,  const cv::Rect &rectBBox, const Eigen::Vector4d & eigBBox,
            const Eigen::Vector3d &eigScale, const Eigen::Vector3d &Position, const double &dRotY, const double &dMeasQuality):
    mnFrameID(pFrameId),mnObjectID(nObjectID), mbDynamicFlag(bDynamicFlag), mdMonoDynaVal(dMonoDynaVal), mdStereoDynaVal(dStereoDynaVal),
    mdTruncated(dTruncated), mdOcculuded(dOcculuded), mdAlpha(dAlpha), mrectBBox(rectBBox),  mScale(eigScale), mPos(Position),
    mdRotY(dRotY), mdMeasQuality(dMeasQuality), mnBALocalForTemporalWindow(-1), mnOptimizeVertexId(-1), mnVelOptimizeVertexId(-1), mbFewMOPsFlag(false), mnMatchesInliers(0),
    mbTrackOK(false), mbNeedCreateNewOKFFlag(false)
    {
    }

    DetectionObject::DetectionObject(const long unsigned int &pFrameId, const Eigen::Matrix<double, 1, 24> &eigDetection):
            mnFrameID(pFrameId), mbDynamicFlag(true), mdMonoDynaVal(0), mdStereoDynaVal(0),
            mbBBoxOutlier(false), mnBALocalForTemporalWindow(-1), mnOptimizeVertexId(-1), mnVelOptimizeVertexId(-1), mbFewMOPsFlag(false), mnMatchesInliers(0),
            mbTrackOK(false), mbNeedCreateNewOKFFlag(false)
    {
        mnObjectID = eigDetection[1];
        mdTruncated = eigDetection[2];
        mdOcculuded = eigDetection[3];
        mdAlpha = eigDetection[4];
        mrectBBox = cv::Rect(eigDetection[5], eigDetection[6], eigDetection[7], eigDetection[8]);
        //mBBox = Eigen::Vector4d(eigDetection[5]+eigDetection[7]/2, eigDetection[6]+eigDetection[8]/2, eigDetection[7], eigDetection[8]);
        /// x length， y height z width
        mScale = Eigen::Vector3d (eigDetection[9], eigDetection[10], eigDetection[11]);
        /// pos: in the camera coordinate system,
        /// note that kitti's camera coordinate system defines z facing forward, y facing down, and x facing right (overhead view)
        mPos = Eigen::Vector3d(eigDetection[12], eigDetection[13], eigDetection[14]);
        mdRotY = eigDetection[15];
        mdRotX = eigDetection[19]; // extend
        mdRotZ = eigDetection[19]; // extend
        mdMeasQuality= 1;

        if(EnDataSetNameNum==0) //kitti
        {
            if(mdTruncated==0 && mdOcculuded == 0 && mdMeasQuality > 0.7)
            {
                mbGoodDetFlag = true;
            }
            else{
                mbGoodDetFlag = false;
            }

        }
        else if(EnDataSetNameNum ==1)// virtual kitti
        {
            if(mdTruncated==0 && mdOcculuded > 0.55 && mdMeasQuality > 0.7)
            {
                mbGoodDetFlag = true;
            }
            else{
                mbGoodDetFlag = false;
            }
        }

        Eigen::Matrix<double, 9, 1> cube_pose;
        cube_pose << mPos[0], mPos[1]-mScale[1]/2, mPos[2], mdRotZ, mdRotY, mdRotX, mScale[0], mScale[1], mScale[2];

        mTruthPosInCameraFrame.fromMinimalVector(cube_pose);

//        Eigen::MatrixXd cube_corners = mTruthPosInCameraFrame.compute3D_BoxCorner();
//        cout<<mPos.transpose()<<endl<<endl;
//        cout<<cube_corners<<endl;
    }

    DetectionObject::DetectionObject(const long unsigned int &pFrameId, const int &object_id, const cv::Rect &BBox, const Eigen::Vector3d &scale, const Eigen::Vector3d &position, const Eigen::Vector3d& rotation):
            mnFrameID(pFrameId), mbDynamicFlag(true), mdMonoDynaVal(0), mdStereoDynaVal(0),
            mbBBoxOutlier(false), mnBALocalForTemporalWindow(-1), mnOptimizeVertexId(-1), mnVelOptimizeVertexId(-1), mbFewMOPsFlag(false), mnMatchesInliers(0),
            mbTrackOK(false), mbNeedCreateNewOKFFlag(false)
    {
        mnObjectID = object_id;
        mdTruncated = 0;
        mdOcculuded = 0;
        mdAlpha = 0;
        mrectBBox = BBox;
        TooFar = true;
        mScale = scale;
        mPos = position;
        mdRotY = rotation[1];
        mdRotX = rotation[0]; // extend
        mdRotZ = rotation[2]; // extend
        mdMeasQuality= 1;

        if(EnDataSetNameNum==0) //kitti
        {
            if(mdTruncated==0 && mdOcculuded == 0 && mdMeasQuality > 0.7)
            {
                mbGoodDetFlag = true;
            }
            else{
                mbGoodDetFlag = false;
            }

        }
        else if(EnDataSetNameNum ==1)// virtual kitti
        {
            if(mdTruncated==0 && mdOcculuded > 0.55 && mdMeasQuality > 0.7)
            {
                mbGoodDetFlag = true;
            }
            else{
                mbGoodDetFlag = false;
            }
        }

        Eigen::Matrix<double, 9, 1> cube_pose;
        cube_pose << mPos[0], mPos[1]-mScale[1]/2, mPos[2], mdRotZ, mdRotY, mdRotX, mScale[0], mScale[1], mScale[2];
        mTruthPosInCameraFrame.fromMinimalVector(cube_pose);
    }


    MapObject* DetectionObject::GetMapObject()
    {
        unique_lock<mutex> lock(mMutexMapObject);
        return mBelongObject;
    }


    void DetectionObject::SetMapObject(MapObject *pMO)
    {
        unique_lock<mutex> lock(mMutexMapObject);
        mBelongObject = pMO;
    }

    cv::Rect DetectionObject::GetBBoxRect()
    {
        return mrectBBox;
    }


    void DetectionObject::InitInFrameMapObjectPointsOrder(const size_t &nNum)
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        mvMapPoints = vector<MapObjectPoint*>(nNum, static_cast<MapObjectPoint*>(NULL));
    }

    void DetectionObject::AddMapObjectPoint(const int &nInDetObjMOPOrder, MapObjectPoint* pMP)
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        mvMapPoints[nInDetObjMOPOrder] = pMP;
    }

    void DetectionObject::EraseMapObjectPoint(const int &nInDetObjMOPOrder)
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        mvMapPoints[nInDetObjMOPOrder] = static_cast<MapObjectPoint*>(NULL);
    }

    void DetectionObject::SetFewMOPsFlag(bool bFewMOPsFlag)
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        mbFewMOPsFlag = bFewMOPsFlag;
    }

    /*
    vector<MapObjectPoint*> DetectionObject::GetInFrameMapObjectPoints()
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        return mvMapPoints;
    }*/

    void DetectionObject::SetDynamicFlag(const double &dMonoDynVal, const double &dStereoDynVal)
    {
        unique_lock<mutex> lock(mMutexDynVal);
        if (dMonoDynVal>0 || dStereoDynVal >0){
            mdMonoDynaVal = dMonoDynVal;
            mdStereoDynaVal = dStereoDynVal;
        }
        else return;

//        int monothre = 0.035*bbox;
//        int stereothre = 0.04*bbox;
//        if (monothre<200)
//            monothre = 200;
//        if (monothre>1000)
//            monothre = 1000;
//        if (stereothre<200)
//            stereothre = 200;
//        if (stereothre>1000)
//            stereothre = 1000;
        //if(mdMonoDynaVal>2.8 || mdStereoDynaVal > 4)
        if(mdMonoDynaVal>1 || mdStereoDynaVal > 2)
        {
            mbDynamicFlag = true;
        }
        else{
            mbDynamicFlag = false;
        }
    }

    void DetectionObject::SetDynamicFlag(bool bDynamicFlag)
    {
        unique_lock<mutex> lock(mMutexDynVal);
        mbDynamicFlag = bDynamicFlag;
    }

    bool DetectionObject::GetDynamicFlag()
    {
        unique_lock<mutex> lock(mMutexDynVal);
        return mbDynamicFlag;
    }
}
