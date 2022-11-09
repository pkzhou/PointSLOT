//
// Created by liuyuzhen on 2021/4/3.
//
#include "MapObjectPoint.h"
#include "DetectionObject.h"
#include "Frame.h"
#include "Converter.h"
#include "ObjectKeyFrame.h"
#include "ORBmatcher.h"
#include "MapObject.h"
#include<iostream>

using namespace std;
namespace ORB_SLAM2
{
long unsigned int MapObjectPoint::nNextId=0;
//普通frame
MapObjectPoint::MapObjectPoint(MapObject* BelongObject, const cv::Mat &PosInObj, const cv::Mat &InFirstCamFramePosition, Frame* pFrame, const size_t &nOrder, const size_t &idx):
mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), mnSlidingWindowBAID(-1), mnBAVertexID(-1), mnBALocalForKF(0),
mnSetBadFrameID(-1), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
mbBad(false), mnFuseCandidateForKF(0), mnVisible(1), mnFound(1), mpReplaced(static_cast<MapObjectPoint*>(NULL)),mnId(-1)
{
    pFrame->mvObjPointsDescriptors[nOrder].row(idx).copyTo(mDescriptor); // 描述子
    mOw = pFrame->GetCameraCenter();
    mMapObject = BelongObject; // 所属目标

    PosInObj.copyTo(mInObjFramePosition); // 位置
    meigInObjFramePosition = Converter::toVector3d(PosInObj);
    meigInFirstObsCamFramePosition = Converter::toVector3d(InFirstCamFramePosition);


    // 方向为 oPcj
    // FIXME 这里mNormalVector与世界系下计算的什么不同，可以直接代替吗
    cv::Mat oPcj = mInObjFramePosition - Converter::toCvMat(mMapObject->GetCFInFrameObjState(pFrame->mnId).pose.inverse().translation());
    mNormalVector = oPcj/cv::norm(oPcj);


    // 最大最小距离计算
    const float dist = cv::norm(oPcj);
    const int level = pFrame->mvObjKeysUn[nOrder][idx].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;
    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    mpRefObjKF = static_cast<ObjectKeyFrame*>(NULL);// 参考关键帧

    mnId = nNextId++;
    create = 1;


}
// keyframe
MapObjectPoint::MapObjectPoint(MapObject* pMO, const cv::Mat &PosInObj, const cv::Mat &InFirstCamFramePosition, ObjectKeyFrame* pKF):
        mnFirstKFid(pKF->mnId), mnFirstFrame(pKF->mnFrameId), mnSlidingWindowBAID(-1), mnBAVertexID(-1), mnBALocalForKF(0),
        mnSetBadFrameID(-1), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mfMinDistance(0), mfMaxDistance(0),
        mbBad(false), mnFuseCandidateForKF(0), mnVisible(1), mnFound(1), mpReplaced(static_cast<MapObjectPoint*>(NULL)),mnId(-1)
{
    mMapObject = pMO;

    PosInObj.copyTo(mInObjFramePosition);
    meigInObjFramePosition = Converter::toVector3d(PosInObj);
    meigInFirstObsCamFramePosition = Converter::toVector3d(InFirstCamFramePosition);

    mNormalVector = cv::Mat::zeros(3,1,CV_32F); // 方向置为0 , 最大最远距离也置为0

    mpRefObjKF = pKF;

    mnId = nNextId++;

    create = 2;
}

MapObject* MapObjectPoint::GetMapObject()
{
    unique_lock<mutex> lock(mMutexObject);
    return mMapObject;
}

cv::Mat MapObjectPoint::GetInObjFramePosition()
{
    unique_lock<mutex> lock(mMutexPos);
    return mInObjFramePosition;
}

Eigen::Vector3d MapObjectPoint::GetInObjFrameEigenPosition()
{
    unique_lock<mutex> lock(mMutexPos);
    return meigInObjFramePosition;
}

void MapObjectPoint::SetInObjFramePosition(const cv::Mat &Pos)
{
    unique_lock<mutex> lock(mMutexPos);
    meigInObjFramePosition = Converter::toVector3d(Pos);
    Pos.copyTo(mInObjFramePosition);
}

/*
void MapObjectPoint::AddInFrameObservation(const size_t& pFrameId, const size_t& nObjOrder, const size_t &idx)
{
    if(mInFrameObservations.count(pFrameId))
        return;
    mInFrameObservations[pFrameId] = make_pair(nObjOrder, idx);
    nInFrameObsNum++;
}*/

/*
void MapObjectPoint::EraseInFrameObservation(size_t pFrameId)
{
    if(mInFrameObservations.count(pFrameId))
    {
        mInFrameObservations.erase(pFrameId);
        nInFrameObsNum--;
    }
}*/


map<ObjectKeyFrame*, size_t> MapObjectPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}


float MapObjectPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

float MapObjectPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

cv::Mat MapObjectPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}


int MapObjectPoint::PredictScale(const float &currentDist, ObjectKeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapObjectPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}


cv::Mat MapObjectPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapObjectPoint::GetIndexInKeyFrame(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

int MapObjectPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

MapObjectPoint* MapObjectPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapObjectPoint::Replace(MapObjectPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;
    if(pMP->GetMapObject() != this->GetMapObject())
        assert(0);
    int nvisible, nfound;
    map<ObjectKeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true; // TODO 这四个属性都是干嘛用的
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }
    for(map<ObjectKeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++) // 当前点的所有观测
    {
        ObjectKeyFrame* pKF = mit->first;
        if(!pMP->IsInKeyFrame(pKF))//如果新的点不在关键帧中, 则用新的点替换旧点的观测
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        else//如果新的点已经在关键帧中, 直接去掉旧的点的观测
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);// 可见属性, 找到属性是干嘛的 TODO 在于不在关键帧中次数是不是不一样
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();// 计算该点的描述子
    mMapObject->EraseMapObjectPoint(this);

    // 从地图中将该点删除
    //mpMap->EraseMapPoint(this);
}

void MapObjectPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapObjectPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

// 更新这个点的平均观测方向和深度
void MapObjectPoint::UpdateNormalAndDepth()
{
    map<ObjectKeyFrame*,size_t> observations;
    ObjectKeyFrame* pRefKF;
    cv::Mat Poj;
    {
        unique_lock<mutex> lock1(mMutexFeatures); // 先把该地图点的原始信息保存下来
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF= mpRefObjKF;  //该点的参考关键帧是什么
        Poj = mInObjFramePosition.clone();
    }
    if(observations.empty())
        return;
    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F); // 算出该点所有观测的平均观测方向
    int n=0;
    for(map<ObjectKeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        ObjectKeyFrame* pKF = mit->first;
        cv::Mat Poc = pKF->GetCameraCenter();
        cv::Mat oPcj = mInObjFramePosition - Poc; // Poj - Poc
        normal = normal + oPcj/cv::norm(oPcj);
        n++;
    }
    cv::Mat oPcj = Poj - pRefKF->GetCameraCenter(); // 参考关键帧时刻的oPcj
    const float dist = cv::norm(oPcj);
    const int level = pRefKF->mvObjKeysUn[observations[pRefKF]].octave; // 参考关键帧时刻的金字塔尺度
    const float levelScaleFactor = pRefKF->mvScaleFactors[level]; // 层数越高, 尺度因子越大
    const int nLevels = pRefKF->mnScaleLevels; // 总层数, 8层
    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor; // 最大距离 = 参考关键时刻(点到相机)的距离 * 参考关键帧时刻的尺度因子 TODO 关键是依赖于参考关键帧
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // 最小距离 = 最大距离/(预定义的最大尺度因子 1.2^7)
        mNormalVector = normal / n; // 平均观测方向 = 所有观测的oPcj向量方向的平均值
    }
}


bool MapObjectPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

bool MapObjectPoint::IsInKeyFrame(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}


void MapObjectPoint::AddObservation(ObjectKeyFrame* pKF, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF)) // 如果观测中有该关键帧, 则跳过
        return;
    mObservations[pKF]=idx;
    if (pKF->mvuObjKeysRight[idx] >= 0) // 右目有合适的匹配点就+2
        nObs += 2;
    else
        nObs++;
}

void MapObjectPoint::EraseObservation(ObjectKeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if (pKF->mvuObjKeysRight[idx] >= 0)
                nObs -= 2;
            else
                nObs--;
            mObservations.erase(pKF);
            if(mpRefObjKF==pKF)
                mpRefObjKF=mObservations.begin()->first;
            if(nObs<=2)
                bBad=true;
        }
    }
    if(bBad)
        SetBadFlag();
}

void MapObjectPoint::SetBadFlag()
{
    map<ObjectKeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    // 依次从(观测到该mappoint)的关键帧中删去:
    for(map<ObjectKeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)// 遍历obs(该mappoints生前的所有观测), 依次从(观测到该mappoint)的关键帧中删去
    {
        ObjectKeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }
    // 从目标中删去该mappoint:
    mMapObject->EraseMapObjectPoint(this);

    // mpMap->EraseMapPoint(this);
}

float MapObjectPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

// 计算地图点的描述子
void MapObjectPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;
    map<ObjectKeyFrame*,size_t> observations; // 该点目前所有关键帧观测
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations = mObservations;
    }
    if(observations.empty())
        return;
    vDescriptors.reserve(observations.size());
    for(map<ObjectKeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)// 该点在所有观测关键帧中的描述子
    {
        ObjectKeyFrame* pKF = mit->first;
        if(!pKF->isBad()) //
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }
    if(vDescriptors.empty())
        return;

    const size_t N = vDescriptors.size(); // 计算这些描述子互相之间的距离: 第一行表示0 0-1 0-2 0-3 ... 0-(N-1); 第二行表示1-0 0 1-2 ... 1-(N-1)
    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end()); // 对每一行进行有小到大排序
        int median = vDists[0.5*(N-1)]; // 取出中位数

        if(median<BestMedian) // 找到所有行的最小中位数, 对应的描述子即为该地图点的描述子
        {
            BestMedian = median;
            BestIdx = i;
        }
    }
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}


}