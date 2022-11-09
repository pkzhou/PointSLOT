//
// Created by liuyuzhen on 2021/9/9.
//

#ifndef ORB_SLAM2_OBJECTLOCALMAPPING_H
#define ORB_SLAM2_OBJECTLOCALMAPPING_H
#include<mutex>
#include <vector>
#include <opencv2/opencv.hpp>


using std::mutex;
namespace ORB_SLAM2
{
    class ObjectKeyFrame;
    class MapObjectPoint;
    class Tracking;
class ObjectLocalMapping
{

public:// 函数
        ObjectLocalMapping();
        bool CheckNewObjectKeyFrames();
        bool CheckTheSameObject();
        void ProcessNewObjectKeyFrame();
        void MapObjectPointCulling();
        //bool CheckNewKeyFrames();
        void SearchInNeighbors();
        void KeyFrameCulling();
        void InsertKeyFrame(const int &nObjectId, ObjectKeyFrame *pKF);
        void InsertOneObjKeyFrame(ObjectKeyFrame *pKF);
        void SetTracker(Tracking *pTracker);
        void RequestFinish();
        bool isFinished();
        void Run();
        void SetFinish();
        bool CheckFinish();

        int KeyframesInQueue(){
            std::unique_lock<std::mutex> lock(mMutexNewKFs);
            return mlNewObjectKeyFrames.size();
        }

        //变量
        std::map<int, std::list<ObjectKeyFrame*>> mlMapNewObjectKeyFrames;
        std::list<ObjectKeyFrame*> mlNewObjectKeyFrames;
        ObjectKeyFrame* mpCurrentObjectKeyFrame;
        std::list<MapObjectPoint*> mlpRecentAddedMapObjectPoints;



protected:

    Tracking* mpTracker;
    bool mbFinishRequested;
    bool mbAbortBA;
    bool mbFinished;
    bool mbAcceptObjectKeyFrames;

    std::mutex mMutexNewKFs;
    std::mutex mMutexMapNewKFs;
    std::mutex mMutexConnections;
    std::mutex mMutexAccept;
    std::mutex mMutexFinish;


    // 函数
    void SetAcceptObjectKeyFrames(bool flag);



    //

};
}


#endif //ORB_SLAM2_OBJECTLOCALMAPPING_H
