//
// Created by liuyuzhen on 2020/5/23.
//
#include "MapObject.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"
#include "Parameters.h"
#include <Converter.h>
#include "ObjectKeyFrame.h"

//#include "KeyFrame.h"
using namespace std;
namespace ORB_SLAM2
{

long int MapObject::nNextId = 0;
MapObject::MapObject(const int &nTruthID, const bool& bDynamicFlag,  long unsigned int& nFirstFrameId,
        const bool &bScaleIsKnownAndPrecise, const Vector6d &eigVel, const g2o::ObjectState &cObjState):
        mnTruthID(nTruthID), mbDynamicFlag(bDynamicFlag), mnFirstObservationFrameId(nFirstFrameId), mbScaleIsKnownAndPreciseFlag(bScaleIsKnownAndPrecise),
        mbVelInit(false), mnSlidingWindowBAID(-1), mnObsByFrameNum(0), mbDynamicChanged(false), mbFirstObserved(true)
{
    mnId = nNextId++;
    mmInAllFrameVelocity[nFirstFrameId] = eigVel;
    mmInAllFrameStates[nFirstFrameId] = cObjState;
}

MapObject::MapObject(const int &nTruthID, const bool& bDynamicFlag,  long unsigned int& nFirstFrameId,
                     const bool &bScaleIsKnownAndPrecise, const Vector6d &eigVel, const g2o::ObjectState &cObjState, const bool &bCF):
        mnTruthID(nTruthID), mbDynamicFlag(bDynamicFlag), mnFirstObservationFrameId(nFirstFrameId), mbScaleIsKnownAndPreciseFlag(bScaleIsKnownAndPrecise),
        mbVelInit(false), mnSlidingWindowBAID(-1), mnObsByFrameNum(0), mbDynamicChanged(false), mbFirstObserved(true)
{
    mnId = nNextId++;
    //mmInAllFrameVelocity[nFirstFrameId] = eigVel;
    //mmInAllFrameStates[nFirstFrameId] = cObjState;

    mmCFAllFsObjStates[nFirstFrameId] = cObjState;
    mVirtualVelocity = Vector6d::Zero();
}



void MapObject::AddKeyFrameObservation(KeyFrame *pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexDetectionObject);
    if (mmInKeyFrameObservations.count(pKF))
        return;
    mmInKeyFrameObservations[pKF] = idx;
    mnObsByKeyFrameNum++;
}

void MapObject::AddFrameObservation(const long unsigned int& nFrameId, size_t idx)
{
    unique_lock<mutex> lock(mMutexDetectionObject);
    if(mmInFrameObservations.count(nFrameId))
        return;
    mmInFrameObservations[nFrameId] = idx;
    mnObsByFrameNum++;
}

void MapObject::EraseFrameObservation(const long unsigned int &nFrameId)
{
    unique_lock<mutex> lock(mMutexDetectionObject);
    if(mmInFrameObservations.count(nFrameId))
        mmInFrameObservations.erase(nFrameId);
}


void MapObject::AddMapObjectPoint(MapObjectPoint *pMP)
{
    unique_lock<mutex> lock(mMutexLandmarks);
    msMapObjectPoints.insert(pMP);
}

std::vector<MapObjectPoint *> MapObject::GetMapObjectPoints()
{
    unique_lock<mutex> lock(mMutexLandmarks);
    return vector<MapObjectPoint*>(msMapObjectPoints.begin(), msMapObjectPoints.end());
}

void MapObject::EraseMapObjectPoint(MapObjectPoint *PMP)
{
    unique_lock<mutex> lock(mMutexLandmarks);
    msMapObjectPoints.erase(PMP);
}

void MapObject::ClearMapObjectPoint()
{
    unique_lock<mutex> lock(mMutexLandmarks);
    msMapObjectPoints.clear();
}

unordered_map<KeyFrame *, size_t> MapObject::GetKeyFrameObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mmInKeyFrameObservations;
}

int MapObject::GetKeyFrameObsNum()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnObsByKeyFrameNum;
}

//TODO 针对与frame的相关函数
bool MapObject::GetHaveBeenOptimizedInFrameFlag()
{
    unique_lock<mutex> lock(mMutexforoptimized);
    return mbHaveBeenOptimizedInFrameFlag;
}





void MapObject::SetHaveBeenOptimizedInFrameFlag()
{
    unique_lock<mutex> lock(mMutexforoptimized);
    mbHaveBeenOptimizedInFrameFlag = true;
}

g2o::ObjectState MapObject::GetInFrameObjState(const int &frame_id)
{
    unique_lock<mutex> lock(mMutexPos);
    g2o::ObjectState cuboid_temp;
    if(mmInAllFrameStates.count(frame_id))
    {
        cuboid_temp = mmInAllFrameStates[frame_id];
    }
    else
    {
        cout<<"The object is not observed in this frame!"<<endl;
        assert(0);
    }
    return cuboid_temp;
}

g2o::ObjectState MapObject::GetCFInFrameObjState(const int &frame_id)
{
    unique_lock<mutex> lock(mMutexPos);
    g2o::ObjectState cuboid_temp;
    if(mmCFAllFsObjStates.count(frame_id))
    {
        cuboid_temp = mmCFAllFsObjStates[frame_id];
    }
    else
    {
        cout<<"The object is not observed in this frame!"<<endl;
        assert(0);
    }
    return cuboid_temp;
}


Vector6d MapObject::GetInFrameObjVelocity(const int &frame_id)
{
    unique_lock<mutex> lock(mMutexVel);
    Vector6d vel;
    vel.setZero();
    if(mmInAllFrameVelocity.count(frame_id))
    {
        vel = mmInAllFrameVelocity[frame_id];
    }
    else
    {
        cout<<"The object is not observed in this frame!"<<endl;
        assert(0);
    }

    return vel;
}

void MapObject::SetInFrameObjVelocity(Vector6d vel, int frame_id)
{
    unique_lock<mutex> lock(mMutexVel);
    mmInAllFrameVelocity[frame_id] = vel;
}

void MapObject::UpdateVelocity(const long unsigned int &nCurrentFrameId)
{
    if (mbDynamicFlag == false){
        mVirtualVelocity = Vector6d::Zero();
        return;
    }
    unique_lock<mutex> lock(mMutexVel);
    g2o::ObjectState cInFrameLatestObjState, cInCurrentFrameObjState;
    int nLatestFrameId = this->GetInSecondToLatestFrameObjState(cInFrameLatestObjState);
    if(nLatestFrameId==-1)
        return;
    if(nLatestFrameId==int(nCurrentFrameId))
        assert(0);
    cInCurrentFrameObjState = this->GetInFrameObjState(nCurrentFrameId);
    double dDT = EdT*(nCurrentFrameId - nLatestFrameId);
    if(dDT<EdMaxObjMissingDt)
    {

        g2o::SE3Quat Twl = cInFrameLatestObjState.pose;
        g2o::SE3Quat Twc = cInCurrentFrameObjState.pose;
        g2o::SE3Quat Tlc = Twl.inverse() * Twc;

        double deltaT = dDT;
        Eigen::Vector3d AngularVelocity = Tlc.log().head(3)/deltaT;
        Eigen::Vector3d LinearVeclocity = Tlc.translation()/deltaT;
        Vector6d velTmp;
        velTmp.head(3) = AngularVelocity;
        velTmp.tail(3) = LinearVeclocity;

        mVirtualVelocity = velTmp;

        if(mbVelInit)
        {
            mmInAllFrameVelocity[nCurrentFrameId] = velTmp;
        }
        else{
            mmInAllFrameVelocity[nCurrentFrameId] = velTmp;
            mmInAllFrameVelocity[nLatestFrameId] = velTmp;
            mbVelInit = true;
        }
    }

    else{
        mbVelInit = false;
        mmInAllFrameVelocity[nCurrentFrameId] = Vector6d::Zero();
        mVirtualVelocity = Vector6d::Zero();
    }
}

void MapObject::UpdateCFVelocity(const long unsigned int &nCurrentFrameId, cv::Mat camera_Tcl)
{
    if (mbDynamicFlag == false){
        mVirtualVelocity = Vector6d::Zero();
        return;
    }
    unique_lock<mutex> lock(mMutexVel);
    g2o::ObjectState cInFrameLatestObjState, cInCurrentFrameObjState;
    int nLatestFrameId = this->GetCFInSecondToLatestFrameObjState(cInFrameLatestObjState);
    if(nLatestFrameId==-1)
        return;
    if(nLatestFrameId==int(nCurrentFrameId))
        assert(0);
    cInCurrentFrameObjState = this->GetCFInFrameObjState(nCurrentFrameId);
    double dDT = EdT*(nCurrentFrameId - nLatestFrameId);

    if(dDT<EdMaxObjMissingDt)
    {

        g2o::SE3Quat Tlo1 = cInFrameLatestObjState.pose;
        g2o::SE3Quat Tco2 = cInCurrentFrameObjState.pose;

        g2o::SE3Quat Tlc = Tlo1.inverse() * Converter::toSE3Quat(camera_Tcl).inverse() * Tco2;

        double deltaT = dDT;
        Eigen::Vector3d AngularVelocity = Tlc.log().head(3)/deltaT;
        Eigen::Vector3d LinearVeclocity = Tlc.translation()/deltaT;
        Vector6d velTmp;
        velTmp.head(3) = AngularVelocity;
        velTmp.tail(3) = LinearVeclocity;
        mVirtualVelocity = velTmp;
    }
    else{
        mVirtualVelocity = Vector6d::Zero();
    }
}

g2o::ObjectState MapObject::GetInKeyFrameObjState(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexPos);
    g2o::ObjectState cuboid_temp = mmInAllKeyFrameStates[pKF].first;
    return cuboid_temp;
}

DetectionObject* MapObject::GetLatestObservation() {
    if (!mmmDetections.empty())
    {
        return (--mmmDetections.end())->second;
    }
    else{
        assert(0);
    }
}

int MapObject::GetInLatestFrameObjState(g2o::ObjectState& cuboidTmp)
{
    unique_lock<mutex> lock(mMutexPos);
    if (mmInAllFrameStates.size() != 0)
    {
        cuboidTmp = (--mmInAllFrameStates.end())->second;
        return (--mmInAllFrameStates.end())->first;
    }
    else{
        return -1;
    }
}

int MapObject::GetCFLatestFrameObjState(g2o::ObjectState& cuboidTmp)
{
    unique_lock<mutex> lock(mMutexPos);
    if (mmCFAllFsObjStates.size() != 0)
    {
        cuboidTmp = (--mmCFAllFsObjStates.end())->second;
        return (--mmCFAllFsObjStates.end())->first;
    }
    else{
        return -1;
    }
}
std::map<ObjectKeyFrame *, g2o::ObjectState> MapObject::GetCFInAllKFsObjStates() {
    unique_lock<mutex> lock(mMutexKFPos);
    return mmCFAllKFsObjStates;
}
g2o::ObjectState MapObject::GetCFObjectKeyFrameObjState(ObjectKeyFrame* OKF)
{
    unique_lock<mutex> lock(mMutexKFPos);
    if(mmCFAllKFsObjStates.count(OKF))
    {
        return mmCFAllKFsObjStates[OKF];
    }
    else{
        assert(0);
    }
}

void MapObject::SetCFObjectKeyFrameObjState(ObjectKeyFrame* OKF, const g2o::ObjectState &obj)
{
    unique_lock<mutex> lock(mMutexKFPos);
    mmCFAllKFsObjStates[OKF] = obj;
}

int MapObject::GetInSecondToLatestFrameObjState(g2o::ObjectState& cuboidTmp)
{
    unique_lock<mutex> lock(mMutexPos);
    if (mmInAllFrameStates.size() > 1)
    {
        cuboidTmp = (--(--mmInAllFrameStates.end()))->second;
        return (--(--mmInAllFrameStates.end()))->first;
    }
    else{
        return -1;
    }
}


std::map<long unsigned int, g2o::ObjectState> MapObject::GetCFInAllFrameObjStates()
{
    unique_lock<mutex> lock(mMutexPos);
    return mmCFAllFsObjStates;
}

int MapObject::GetCFInSecondToLatestFrameObjState(g2o::ObjectState& cuboidTmp)
{
    unique_lock<mutex> lock(mMutexPos);
    if (mmCFAllFsObjStates.size() > 1)
    {
        cuboidTmp = (--(--mmCFAllFsObjStates.end()))->second;
        return (--(--mmCFAllFsObjStates.end()))->first;
    }
    else{
        return -1;
    }
}

std::map<int, g2o::ObjectState> MapObject::GetInAllFrameObjStates()
{
    unique_lock<mutex> lock(mMutexPos);
    return mmInAllFrameStates;
}

void MapObject::SetInFrameObjState(const g2o::ObjectState &Pos, int frame_id)
{
    unique_lock<mutex> lock(mMutexPos);
    mmInAllFrameStates[frame_id] = Pos;
}

void MapObject::SetCFInFrameObjState(const g2o::ObjectState &Pos, int frame_id)
{
    unique_lock<mutex> lock(mMutexPos);
    mmCFAllFsObjStates[frame_id] = Pos;
}


void MapObject::EraseInFrameObjState(const long unsigned int &nFrameId)
{
    unique_lock<mutex> lock(mMutexPos);
    if(mmInAllFrameStates.count(nFrameId))
        mmInAllFrameStates.erase(nFrameId);
}

void MapObject::SetHaveBeenOptimizedInKeyFrameFlag()
{
    unique_lock<mutex> lock(mMutexforoptimized);
    mbHaveBeenOptimizedInKeyFrameFlag = true;

}
bool MapObject::GetHaveBeenOptimizedInKeyFrameFlag()
{
    unique_lock<mutex> lock(mMutexforoptimized);
    return mbHaveBeenOptimizedInKeyFrameFlag;
}


std::map<KeyFrame*, std::pair<g2o::ObjectState, bool>, cmpKeyframe, Eigen::aligned_allocator<std::pair<const KeyFrame*, std::pair<g2o::ObjectState, bool>>>> MapObject::GetInAllKeyFrameObjStates()
{
    unique_lock<mutex> lock(mMutexPos);
    return mmInAllKeyFrameStates;

}

void MapObject::SetInKeyFrameObjState(const g2o::ObjectState &Pos, KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexPos);
    mmInAllKeyFrameStates[pKF] = make_pair(Pos, true);
}

void MapObject::DynamicDetection(const bool &bDynFlag){
    if (!mbDynamicChanged) {
        mqbHistoricalDynafFlag.push(bDynFlag);
        if (mqbHistoricalDynafFlag.size()<4)
            return;

        if (mqbHistoricalDynafFlag.size() > 4) {
            mqbHistoricalDynafFlag.pop();
        }
        queue<bool> q(mqbHistoricalDynafFlag);
        bool currentflag = bDynFlag;
        while(!q.empty()){
            if (currentflag == q.front()){
                //cout<<" "<<q.front();
                q.pop();
            }
            else return;
        }
        if (mbDynamicFlag!=bDynFlag)
            mbDynamicChanged = true;
    }
}
void MapObject::SetDynamicFlag(const bool &bDynFlag)
{
    unique_lock<mutex> lock(mMutexDynVal);
    if (mbFirstObserved){
        mbDynamicFlag = bDynFlag;
        mbFirstObserved = false;
    }
    if (mbDynamicChanged){
        mbDynamicFlag = bDynFlag;
        mbDynamicChanged = false;
    }

}

bool MapObject::GetDynamicFlag()
{
    unique_lock<mutex> lock(mMutexDynVal);
    return mbDynamicFlag;
}



}
