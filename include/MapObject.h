//
// Created by liuyuzhen on 2020/5/23.
//

#ifndef ORB_SLAM2_MAPOBJECT_H
#define ORB_SLAM2_MAPOBJECT_H
#include "KeyFrame.h"
#include <unordered_map>
//#include "Frame.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>
#include <opencv2/core.hpp>
#include <g2o_Object.h>
#include <Eigen/StdVector>
#include <queue>
namespace ORB_SLAM2 {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    class MapPoint;
    class Map;
    class KeyFrame;
    class Frame;
    //class DetectionObject;
    class MapObjectPoint;
    class ObjectKeyFrame;
    struct cmpKeyframe
    { //sort frame based on ID
        bool operator()(const KeyFrame *a, const KeyFrame *b) const
        {
            return a->mnId < b->mnId;
        }
    };

    class MapObject {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        MapObject(const int &nTruthID, const bool& bDynamicFlag,  long unsigned int& nFirstFrameId,
                  const bool &bScaleIsKnownAndPrecise, const Vector6d &eigVel, const g2o::ObjectState &cObjState);
        MapObject(const int &nTruthID, const bool& bDynamicFlag,  long unsigned int& nFirstFrameId,
                const bool &bScaleIsKnownAndPrecise, const Vector6d &eigVel, const g2o::ObjectState &cObjState, const bool &bCF);

        // 基本属性
        long int mnId;// 表示目标的个数, 其实这个后面应该作为object的真实id
        static long int nNextId;
        int mnTruthID; // 临时用来关联的,因为现在是读临时文件
        bool mbDynamicFlag;
        bool mbDynamicChanged;
        bool mbFirstObserved;
        queue<bool> mqbHistoricalDynafFlag;
        double mdMonoDynaVal;
        double mdStereoDynaVal;
        long unsigned int mnFirstObservationFrameId;
        bool mbScaleIsKnownAndPreciseFlag;
        bool mbVelInit;
        bool mbPoseInit;


        // 优化相关
        std::map<int,int> mmBAFrameIdAndObjVertexID;
        bool mbHaveBeenOptimizedInFrameFlag = false;
        bool mbHaveBeenOptimizedInKeyFrameFlag = false;
        int mnSlidingWindowBAID;

        // pose vel相关
        //std::map<int, g2o::ObjectState, std::less<int>, Eigen::aligned_allocator<std::pair<const int, g2o::ObjectState>>> mmInAllFrameStates;
        ///********目标在世界系下的状态(暂时没有使用)****************///
        std::map<int, g2o::ObjectState> mmInAllFrameStates;
        std::map<int, Eigen::Matrix<double, 6, 1>, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix<double, 6, 1>>>> mmInAllFrameVelocity; //
        std::map<KeyFrame*, std::pair<g2o::ObjectState, bool>, cmpKeyframe, Eigen::aligned_allocator<std::pair<const KeyFrame*, std::pair<g2o::ObjectState, bool>>>> mmInAllKeyFrameStates;

        std::map<long unsigned int, DetectionObject*> mmmDetections;
        //std::map<int, g2o::SE3Quat, std::less<int>, Eigen::aligned_allocator<std::pair<const int, g2o::SE3Quat>>> mmFrameIDAndRefRelPose;
        //std::map<int, KeyFrame *> mmFrameIdAndRefKeyFrame;
        //std::map<int, double> mmFrameIDAndTimestamp;
        //KeyFrame *mpLatestRefKeyFrame;

        // 新定义的pose相关
        ///*******目标在相机系下的状态(正在使用)***************///
        std::map<long unsigned int, g2o::ObjectState> mmCFAllFsObjStates; // 目标在相机系下的状态
        std::map<ObjectKeyFrame*, g2o::ObjectState> mmCFAllKFsObjStates;
        Vector6d mVirtualVelocity;
        //list<g2o::SE3Quat> mlRelativeFramePoses;
        std::map<long unsigned int, pair<ObjectKeyFrame*, g2o::SE3Quat>> mlRelativeFramePoses;


        int GetCFLatestFrameObjState(g2o::ObjectState& cuboidTmp);
        g2o::ObjectState GetCFInFrameObjState(const int &frame_id);
        std::map<long unsigned int, g2o::ObjectState> GetCFInAllFrameObjStates();
        void SetCFInFrameObjState(const g2o::ObjectState &Pos, int frame_id);

        g2o::ObjectState GetCFObjectKeyFrameObjState(ObjectKeyFrame* OKF);
        std::map<ObjectKeyFrame*, g2o::ObjectState> GetCFInAllKFsObjStates();
        void SetCFObjectKeyFrameObjState(ObjectKeyFrame* OKF, const g2o::ObjectState &obj);






        // 观测
        size_t mnObsByKeyFrameNum;
        size_t mnObsByFrameNum;
        std::map<size_t, size_t> mmInFrameObservations;
        std::unordered_map<KeyFrame *, size_t> mmInKeyFrameObservations;

        // tracking 跟踪局部地图会用到的
        long unsigned int mnLastKeyFrameId;
        ObjectKeyFrame* mpReferenceObjKF;
        std::vector<ObjectKeyFrame*> mvLocalObjectKeyFrames;
        std::vector<MapObjectPoint*> mvpLocalMapObjectPoints;






        // 地图
        std::set<MapObjectPoint*> msMapObjectPoints;
        Map *mpMap;
        list<MapObjectPoint*> mlpTemporalPoints;

    public:
        // 函数
        void SetDynamicFlag(const bool &bDynFlag);
        void DynamicDetection(const bool &bDynFlag);
        bool IfDynamicFixed();
        bool GetDynamicFlag();

        // 帧相关
        void AddFrameObservation(const long unsigned int &nFrameId, size_t idx);
        void EraseFrameObservation(const long unsigned int &nFrameId);
        bool GetHaveBeenOptimizedInFrameFlag();
        void SetHaveBeenOptimizedInFrameFlag();
        int GetInLatestFrameObjState(g2o::ObjectState& cuboidTmp);
        int GetInSecondToLatestFrameObjState(g2o::ObjectState& cuboidTmp);
        int GetCFInSecondToLatestFrameObjState(g2o::ObjectState& cuboidTmp);
        void SetInFrameObjState(const g2o::ObjectState &Pos, int frame_id);
        void EraseInFrameObjState(const long unsigned int &nFrameId);
        g2o::ObjectState GetInFrameObjState(const int &frame_id);
        std::map<int, g2o::ObjectState> GetInAllFrameObjStates();
        Vector6d GetInFrameObjVelocity(const int &frame_id);
        void SetInFrameObjVelocity(Vector6d vel, int frame_id);
        void UpdateVelocity(const long unsigned int &nCurrentFrameId);
        void UpdateCFVelocity(const long unsigned int &nCurrentFrameId, cv::Mat camera_Tcl);
        DetectionObject* GetLatestObservation();
        // 关键帧相关
        void AddKeyFrameObservation(KeyFrame *pKF, size_t idx);
        std::unordered_map<KeyFrame *, size_t> GetKeyFrameObservations();
        int GetKeyFrameObsNum();
        bool GetHaveBeenOptimizedInKeyFrameFlag();
        void SetHaveBeenOptimizedInKeyFrameFlag();
        std::map<KeyFrame*, std::pair<g2o::ObjectState, bool>, cmpKeyframe, Eigen::aligned_allocator<std::pair<const KeyFrame*, std::pair<g2o::ObjectState, bool>>>> GetInAllKeyFrameObjStates();
        void SetInKeyFrameObjState(const g2o::ObjectState &Pos, KeyFrame* pKF);
        void SetInLatestKeyFrameObjState(const g2o::ObjectState &Pos, const int &keyframe_id);
        g2o::ObjectState GetInKeyFrameObjState(KeyFrame* pKF);

        // 地图相关
        void AddMapObjectPoint(MapObjectPoint *pMP);
        void EraseMapObjectPoint(MapObjectPoint *PMP);
        void ClearMapObjectPoint();
        std::vector<MapObjectPoint*> GetMapObjectPoints();



    protected:
        std::mutex mMutexFeatures;
        std::mutex mMutexLandmarks;
        std::mutex mMutexforoptimized;
        std::mutex mMutexPos;
        std::mutex mMutexKFPos;
        std::mutex mMutexVel;
        std::mutex mMutexDynVal;
        std::mutex mMutexDetectionObject;

    };
}


#endif //ORB_SLAM2_MAPOBJECT_H
