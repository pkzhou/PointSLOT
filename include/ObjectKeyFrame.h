//
// Created by liuyuzhen on 2021/9/9.
//

#ifndef ORB_SLAM2_OBJECTKEYFRAME_H
#define ORB_SLAM2_OBJECTKEYFRAME_H
#include<mutex>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "g2o_Object.h"


using std::vector;
namespace ORB_SLAM2{


class Frame;
class MapObject;
class DetectionObject;
class MapObjectPoint;


class ObjectKeyFrame {

public:
    /// 函数
    ObjectKeyFrame(Frame &pFrame, const size_t &nInFrameDetObjOrder, bool bFirstObserved);

    void AddMapObjectPoint(MapObjectPoint* pMOP,  const size_t &idx);
    vector<MapObjectPoint*> GetMapObjectPointMatches();
    MapObjectPoint* GetMapObjectPoint(const size_t &idx);
    void ReplaceMapPointMatch(const size_t &idx, MapObjectPoint* pMP);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapObjectPoint* pMP);

    // 共视图相关
    void UpdateConnections();
    void AddConnection(ObjectKeyFrame *pKF, const int &weight);
    void UpdateBestCovisibles();
    void AddChild(ObjectKeyFrame *pKF);
    std::set<ObjectKeyFrame*> GetChilds();
    ObjectKeyFrame* GetParent();
    vector<ObjectKeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    vector<ObjectKeyFrame*> GetVectorCovisibleKeyFrames();
    vector<ObjectKeyFrame*> GetVectorCovisibleLast15KeyFrames();
    void EraseConnection(ObjectKeyFrame* pKF);
    void SetBadFlag();

    // 投影相关
    bool IsInBBox(const float &u, const float &v) const;


    // pose相关
    cv::Mat GetRotation();
    cv::Mat GetTranslation();
    void SetPose(const g2o::SE3Quat &cObjState);
    cv::Mat GetCameraCenter();
    cv::Mat GetPose();

    /// ****************************变量
    static long unsigned int nNextId;
    static std::map<int, int> mNextObjKFId;
    long unsigned int mnId;
    long unsigned int mnFrameId;
    long unsigned int mnObjId;
    long unsigned int mObjTrackId;

    const cv::Mat mDescriptors;
    bool mbFirstObserved;

    MapObject* mpMapObjects;
    DetectionObject* mpDetectionObject;
    vector<MapObjectPoint*> mvpMapObjectPoints;
    vector<cv::KeyPoint> mvObjKeysUn;
    vector<float> mvuObjKeysRight;
    vector<float> mvObjPointDepth;

    // 新添加, 不知道有没有用
    // 相机内参
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;

    // 基本属性
    bool mbBad;
    const int mnScaleLevels; // 尺度因子相关
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;
    long unsigned int mnFuseTargetForKF;
    bool mbNotErase; // TODO 啥意思
    bool mbToBeErased;
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;
    vector<vector<std::vector<std::size_t>>> mvObjKeysGrid;




    // pose 相关, 只存目标在相机系下的pose
    cv::Mat mTco, mToc, mPoc;
    Eigen::Vector3d mScale;


    // 优化相关
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // 共视图相关
    std::map<ObjectKeyFrame*,int> mConnectedKeyFrameWeights; // 本帧所有的共视关键帧与对应权重(即相同目标点数)
    std::vector<ObjectKeyFrame*> mvpOrderedConnectedKeyFrames;  // 本帧所有超过阈值的共视关键帧(已经排好序了)
    std::vector<int> mvOrderedWeights; // 本帧所有超过阈值的共视关键帧的权重 (与mvpOrderedConnectedKeyFrames是一一对应的)
    bool mbFirstConnection;
    ObjectKeyFrame* mpParent; // 父关键帧 = (与本关键帧)权重最大的关键帧, TODO 为什么只执行一次
    std::set<ObjectKeyFrame*> mspChildrens; // 子关键帧, TODO

    // tracking 用到
    long unsigned int mnTrackReferenceForFrame;

public:
    // 函数
    vector<size_t> GetObjectFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    bool isBad();
    void ChangeParent(ObjectKeyFrame *pKF);
    void EraseChild(ObjectKeyFrame *pKF);
    int GetWeight(ObjectKeyFrame *pKF);
    cv::Mat GetPoseInverse();

protected:
    std::mutex mMutexFeatures;
    std::mutex mMutexConnections;
    std::mutex mMutexPose;

};
}

#endif //ORB_SLAM2_OBJECTKEYFRAME_H
