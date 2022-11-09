//
// Created by liuyuzhen on 2021/4/3.
//

#ifndef ORB_SLAM2_MAPOBJECTPOINT_H
#define ORB_SLAM2_MAPOBJECTPOINT_H
//#include"KeyFrame.h"
//#include"Frame.h"
//#include"Map.h"
#include<opencv2/core/core.hpp>
#include<mutex>
#include<map>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
namespace ORB_SLAM2
{
class KeyFrame;
class Map;
class Frame;
class MapObject;
class ObjectKeyFrame;
class MapObjectPoint
{
// 函数
public:
    //MapObjectPoint(bool IsTriangulated, MapObject* BelongObject, cv::Mat PosInObj, Map* pMap, Frame* pFrame);
    MapObjectPoint(MapObject* BelongObject, const cv::Mat &PosInObj, const cv::Mat &InFirstCamFramePosition, Frame* pFrame, const size_t &nOrder, const size_t &idx);
    MapObjectPoint(MapObject* pMO, const cv::Mat &PosInObj, const cv::Mat &InFirstCamFramePosition, ObjectKeyFrame* pKF);

    MapObject* GetMapObject();
    cv::Mat GetInObjFramePosition();
    Eigen::Vector3d GetInObjFrameEigenPosition();
    cv::Mat GetDescriptor();
    int GetIndexInKeyFrame(ObjectKeyFrame *pKF);
    void SetInObjFramePosition(const cv::Mat &Pos);
    //void AddInFrameObservation(const size_t& pFrameId, const size_t& nObjOrder, const size_t &idx);
    //void EraseInFrameObservation(size_t pFrameId);

    void AddObservation(ObjectKeyFrame* pOKF, const size_t &idx);
    void EraseObservation(ObjectKeyFrame* pKF);
    map<ObjectKeyFrame*, size_t> GetObservations();
    int Observations();
    bool IsInKeyFrame(ObjectKeyFrame *pKF);
    void Replace(MapObjectPoint* pMP);// 替代
    MapObjectPoint* GetReplaced();
    void SetBadFlag();


// 变量
public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    cv::Mat mDescriptor;
    MapObject* mMapObject;
    cv::Mat mInObjFramePosition;
    Eigen::Vector3d meigInObjFramePosition;
    Eigen::Vector3d meigInFirstObsCamFramePosition;
    int mnSlidingWindowBAID;
    int mnBAVertexID;
    long unsigned int mnBALocalForKF;
    int mnSetBadFrameID;
    int create;


    // localmapping 需要
    map<ObjectKeyFrame*, size_t> mObservations;
    int nObs;

    // tracking 局部地图用
    long unsigned int mnTrackReferenceForFrame; // 没有含义防止重复添加
    long unsigned int mnLastFrameSeen;
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;

    // 变量
    float mfMinDistance; // 最大最小观测距离, 与金字塔尺度有关
    float mfMaxDistance;
    cv::Mat mNormalVector; // 该MapPoint平均观测方向
    cv::Mat mOw;
    bool mbBad; // bad属性
    long unsigned int mnFuseCandidateForKF;
    ObjectKeyFrame* mpRefObjKF; // 这个点是哪一关键帧建立的, 这一帧就是该点的参考关键帧
    int mnVisible;
    int mnFound;
    MapObjectPoint* mpReplaced;



//  新添加, 还不知道是否有用
public:
    // 函数
    float GetMaxDistanceInvariance();
    float GetMinDistanceInvariance();
    cv::Mat GetNormal();
    int PredictScale(const float &currentDist, ObjectKeyFrame* pKF);
    int PredictScale(const float &currentDist, Frame* pF);
    void ComputeDistinctiveDescriptors();
    bool isBad();
    void UpdateNormalAndDepth();
    float GetFoundRatio();
    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);



protected:
    mutex mMutexObject;
    mutex mMutexPos;
    mutex mMutexFeatures;


};
}


#endif //ORB_SLAM2_MAPOBJECTPOINT_H
