//
// Created by liuyuzhen on 2021/9/9.
//
#include "ObjectKeyFrame.h"
#include "ObjectLocalMapping.h"
#include "ORBmatcher.h"
#include "MapObjectPoint.h"
#include "Tracking.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "MapObject.h"
#include <unordered_set>
using std::unique_lock;
using std::vector;
using std::list;
namespace ORB_SLAM2
{
static double tproc = 0;
static double tba = 0;
static int time = 0;
ObjectLocalMapping::ObjectLocalMapping():
    mbFinishRequested(false), mbAbortBA(false), mbFinished(true) , mbAcceptObjectKeyFrames(true)
{}


bool ObjectLocalMapping::CheckNewObjectKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewObjectKeyFrames.empty());
}

bool ObjectLocalMapping::CheckTheSameObject()
{
    //检查是否还有与当前关键帧相同目标的新关键帧在后面
    if(!CheckNewObjectKeyFrames())
        return false;
    else
    {
        auto lNewObjectKeyFrames = mlNewObjectKeyFrames;
        for(auto pKF : lNewObjectKeyFrames)
        {
            if (pKF->mObjTrackId == mpCurrentObjectKeyFrame->mObjTrackId)
                return true;
        }
        return false;
    }
}

void ObjectLocalMapping::InsertKeyFrame(const int &nObjectId, ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMapNewKFs);
    mlMapNewObjectKeyFrames[nObjectId].push_back(pKF);
}

void ObjectLocalMapping::InsertOneObjKeyFrame(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewObjectKeyFrames.push_back(pKF);
}

void ObjectLocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void ObjectLocalMapping::ProcessNewObjectKeyFrame()
{
    // 从缓冲队列中取出一帧关键帧
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentObjectKeyFrame = mlNewObjectKeyFrames.front(); // Tracking线程向LocalMapping中插入关键帧存在该队列中
        mlNewObjectKeyFrames.pop_front(); // 队列开头弹出
        cout<<YELLOW<<"Object Keyframe"<<mpCurrentObjectKeyFrame->mObjTrackId<<"，queue leaves"<<mlNewObjectKeyFrames.size()<<"帧!";
        cout<<"------";
        for (auto pkf : mlNewObjectKeyFrames)
        {
            cout<<pkf->mObjTrackId<<" ";
        }
        cout<<WHITE<<endl;
    }
    // 计算该关键帧特征点的Bow映射关系

    const vector<MapObjectPoint*> vpMapObjectPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches(); // 当前关键帧的所有地图点
    for(size_t i=0; i<vpMapObjectPointMatches.size(); i++)
    {
        MapObjectPoint* pMP = vpMapObjectPointMatches[i];//检验条件: pMP存在, pMP不是Bad, pMP不是当前关键帧生成的实际跟踪得到的
        if(pMP)
        {
            if(!pMP->isBad()) // 条件这个点不是个坏点
            {
                if(!pMP->IsInKeyFrame(mpCurrentObjectKeyFrame)) // 是跟踪得到的, 不是当前关键帧生成的mappoints,
                {
                        pMP->AddObservation(mpCurrentObjectKeyFrame, i);// 添加观测
                        pMP->UpdateNormalAndDepth();// 获得该点的平均观测方向和观测距离范围
                        pMP->ComputeDistinctiveDescriptors();// 加入关键帧后，更新3d点的最佳描述子
                }
                else
                {
                    mlpRecentAddedMapObjectPoints.push_back(pMP); // 该landmark是当前关键帧生成的, 则等待检查
                }
            }
        }
    }
    mpCurrentObjectKeyFrame->UpdateConnections();// 更新关键帧间的连接关系，Covisibility图和Essential图(tree)
    // 将该关键帧插入到地图中

}

void ObjectLocalMapping::MapObjectPointCulling()
{
    list<MapObjectPoint*>::iterator lit = mlpRecentAddedMapObjectPoints.begin();// 获取等待检查的mappoints
    const unsigned long int nCurrentKFid = mpCurrentObjectKeyFrame->mnId;
    const unsigned long int nCurrentObjKFid = mpCurrentObjectKeyFrame->mnObjId;
    int nThObs = 3;
    const int cnThObs = nThObs; // ???
    int count1 = 0, count2 = 0;
    while(lit!=mlpRecentAddedMapObjectPoints.end())
    {
        MapObjectPoint* pMP = *lit;
        if (pMP->mMapObject->mnTruthID!=mpCurrentObjectKeyFrame->mObjTrackId){
            lit++;
            continue;
        }
        //cout<<"点: "<<pMP->mnFirstKFid<<endl;
        if(pMP->isBad())  //  该landmark是bad,则从容器中删除
        {
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f ) // (mnFound)/mnVisible, TODO 这是一个什么条件
        {
            // 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
            // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
            //cout<<pMP->mnId<<"  ";
            count1++;
            pMP->SetBadFlag(); // 设置badflag
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        //FIXME 这里目标关键帧就存在问题了。发现多个目标时，很快就差了很多帧。
//        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        else if(((int)nCurrentObjKFid-(int)pMP->mpRefObjKF->mnObjId)>=2 && pMP->Observations()<=cnThObs)
        {
            //cout<<pMP->mnId<<"  ";
            // 从该点建立开始，到现在已经过了不小于2个关键帧, 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
            count2++;
            pMP->SetBadFlag(); // 设置badflag
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else if(((int)nCurrentObjKFid-(int)pMP->mpRefObjKF->mnObjId)>=3)
        {
            //从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点, 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else
            lit++;
    }
    //cout<<"目标3D点删除 "<<count1<<" "<<count2<<" 个"<<endl;
}

void ObjectLocalMapping::SearchInNeighbors()
{
    // 这里考虑了当前帧的特征点与过去帧的MapPoint的匹配，但过去帧观测的特征点能与当前帧的MapPoint匹配吗
    // 可以，匹配了之后相当于是公用的MapPoints，同时关联了原始帧和当前帧
    // 但是过去帧的观测如果当时没有得到匹配，但后来帧在对应的点建立了MapPoints，这样好像就没法处理了
    // 发现函数分别针对过去帧和当前帧都作了Fuse
    int nn = 10; // 得到10帧
    const vector<ObjectKeyFrame*> vpNeighKFs = mpCurrentObjectKeyFrame->GetBestCovisibilityKeyFrames(nn); // 得到权重排名前nn的共视关键帧
    vector<ObjectKeyFrame*> vpTargetKFs;
    for(vector<ObjectKeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentObjectKeyFrame->mnId)// 为了防止重复添加
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentObjectKeyFrame->mnId; // 设置pKFi->mnFuseTargetForKF 为当前关键帧的id
        const vector<ObjectKeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);// 得到2级共视关键帧, 就是1级共视关键帧的共视关键帧
        for(vector<ObjectKeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            ObjectKeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentObjectKeyFrame->mnId || pKFi2->mnId==mpCurrentObjectKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF = mpCurrentObjectKeyFrame->mnId;
        }
    }
    // 将当前帧的目标点分别与一级二级相邻帧进行融合
    ORBmatcher matcher;
    vector<MapObjectPoint*> vpMapPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches();

    /*
    cout<<YELLOW<<"融合前各帧3D点: "<<endl;
    cout<<WHITE;
    cout<<"关键帧"<<mpCurrentObjectKeyFrame->mnFrameId<<":"<<endl;
    std::unordered_set<int> hash;
    for(vector<MapObjectPoint*>::iterator vit = vpMapPointMatches.begin(), vend = vpMapPointMatches.end(); vit!=vend; vit++)
    {
        MapObjectPoint* pMP = *vit;
        if(pMP)
        {
            if(!hash.count(pMP->mnId))
                hash.insert(pMP->mnId);
            else
                assert(0);
            cout<<pMP->mnId<<" ";
        }
    }
    cout<<endl;
    for(vector<ObjectKeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKFi = *vit;
        cout<<"关键帧"<<pKFi->mnFrameId<<":"<<endl;
        vector<MapObjectPoint*> vpMapPoints = pKFi->GetMapObjectPointMatches();
        std::unordered_set<int> hasht;
        for(vector<MapObjectPoint*>::iterator it = vpMapPoints.begin(), iend = vpMapPoints.end(); it!=iend; it++)
        {
            MapObjectPoint* pMP = *it;
            if(pMP)
            {
                if(!hasht.count(pMP->mnId))
                    hasht.insert(pMP->mnId);
                else
                    assert(0);

                cout<<pMP->mnId<<" ";
            }
        }
        cout<<endl;
    }*/



    for(vector<ObjectKeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKFi = *vit;
        // 投影当前帧的MapPoints到相邻关键帧pKFi中，并判断是否有重复的MapPoints
        // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint， 那么将两个MapPoint合并（选择观测数多的）
        // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint

        // 输出下融合前的结果(一级二级关键帧和当前帧3D的融合)

        if (!vpMapPointMatches.empty())
        matcher.Fuse(pKFi,vpMapPointMatches); // 当前关键帧地图点与所有共视关键帧融合
        // 输出下融合后的结果
    }

    int points = 0;
    //将一级二级相邻帧的MapPoints分别与当前帧进行融合
    vector<MapObjectPoint*> vpFuseCandidates;// 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    for(vector<ObjectKeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        ObjectKeyFrame* pKFi = *vitKF;
        vector<MapObjectPoint*> vpMapPointsKFi = pKFi->GetMapObjectPointMatches(); // 得到该关键帧的所有地图点
        points = points + vpMapPointsKFi.size();
        for(vector<MapObjectPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapObjectPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentObjectKeyFrame->mnId) // 防止重复添加
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentObjectKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    // 输出融合前的结果(当前关键帧与一级二级关键帧拥有的3D地图点之间的融合)
    //cout<<"融合的KF数目 "<<vpTargetKFs.size()<<", 融合地图点总量 "<<points<<endl;
    if (!vpFuseCandidates.empty())
    matcher.Fuse(mpCurrentObjectKeyFrame,vpFuseCandidates); // 所有共视关键帧的地图点与当前关键帧融合
    // 输出融合前的结果


    /*
    cout<<YELLOW<<"融合后各帧3D点: "<<endl;
    cout<<WHITE;
    cout<<"关键帧"<<mpCurrentObjectKeyFrame->mnFrameId<<" :"<<endl;
    std::unordered_set<int> hash2;
    for(vector<MapObjectPoint*>::iterator vit = vpMapPointMatches.begin(), vend = vpMapPointMatches.end(); vit!=vend; vit++)
    {
        MapObjectPoint* pMP = *vit;
        if(pMP)
        {
            if(!hash2.count(pMP->mnId))
                hash2.insert(pMP->mnId);
            else
                assert(0);
            cout<<pMP->mnId<<" ";
        }
    }
    cout<<endl;
    for(vector<ObjectKeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKFi = *vit;
        cout<<"关键帧"<<pKFi->mnFrameId<<" :"<<endl;
        vector<MapObjectPoint*> vpMapPoints = pKFi->GetMapObjectPointMatches();
        std::unordered_set<int> hasht2;
        for(vector<MapObjectPoint*>::iterator it = vpMapPoints.begin(), iend = vpMapPoints.end(); it!=iend; it++)
        {
            MapObjectPoint* pMP = *it;
            if(pMP)
            {
                if(!hasht2.count(pMP->mnId))
                    hasht2.insert(pMP->mnId);
                else
                    assert(0);
                cout<<pMP->mnId<<" ";
            }
        }
        cout<<endl;
    }*/


    vpMapPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches();// 更新当前帧目标点的描述子，深度，观测主方向等属性
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapObjectPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors(); // TODO fuse里不是已经进行描述子计算了吗?
                pMP->UpdateNormalAndDepth(); // TODO 关键是依赖于参考关键帧前面是否对参考关键帧进行了修改?
            }
        }
    }
    // 更新当前帧的MapPoints后更新与其它帧的连接关系， 即：更新covisibility图
    mpCurrentObjectKeyFrame->UpdateConnections();
}

// 其90%以上的MapPoints能被其他关键帧（至少3个）观测到，则认为该关键帧为冗余关键帧。
void ObjectLocalMapping::KeyFrameCulling()
{
    vector<ObjectKeyFrame*> vpLocalKeyFrames = mpCurrentObjectKeyFrame->GetVectorCovisibleKeyFrames();
    // 对所有的局部关键帧进行遍历
    for(vector<ObjectKeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKF = *vit;
        if(pKF->mbFirstObserved == true)
            continue;
        const vector<MapObjectPoint*> vpMapPoints = pKF->GetMapObjectPointMatches(); // 提取每个共视关键帧的MapPoints
        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)// 遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到
        {
            MapObjectPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(pKF->mvObjPointDepth[i]>pKF->mThDepth || pKF->mvObjPointDepth[i]<0) // 对于双目，仅考虑近处的MapPoints，不超过mbf * 35 / fx
                        continue;
                    nMPs++;
                    if(pMP->Observations()>thObs) // MapPoints至少被三个关键帧观测到
                    {
                        const int &scaleLevel = pKF->mvObjKeysUn[i].octave;
                        const map<ObjectKeyFrame*, size_t> observations = pMP->GetObservations();
                        // 判断该MapPoint是否同时被三个关键帧观测到
                        int nObs=0;
                        for(map<ObjectKeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            ObjectKeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvObjKeysUn[mit->second].octave;
                            if(scaleLeveli<=scaleLevel+1)// 尺度约束，要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
                            {
                                nObs++;
                                if(nObs>=thObs)// 已经找到三个同尺度的关键帧可以观测到该MapPoint，不用继续找了
                                    break;
                            }
                        }
                        if(nObs>=thObs)// 该MapPoint至少被三个关键帧观测到
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

void ObjectLocalMapping::SetAcceptObjectKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptObjectKeyFrames=flag;
}

void ObjectLocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool ObjectLocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void ObjectLocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    cout<<"ObjectLocalMappping Time----proc: "<<tproc/time*1000<<", BA: "<<tba/time*1000<<endl;
}

bool ObjectLocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void ObjectLocalMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        if (CheckNewObjectKeyFrames()) //  遍历所有的目标关键帧, 队列不为空
        {
            auto t1 = std::chrono::steady_clock::now();
            ProcessNewObjectKeyFrame(); // 队列中取出一帧, 作为当前关键帧处理
            MapObjectPointCulling(); // 处理mlpRecentAddedMapObjectPoints中的地图点

            // 相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints
            // CreateNewMapObjects(); // 提升效果不明显
            if (!CheckTheSameObject())//如果队列为空,已经处理完队列中的最后的一个关键帧或者队列中无与当前帧相同目标
            {
                SearchInNeighbors(); // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
                // BA
                // 最好ba需要有多个关键帧才进行
                auto t2 = std::chrono::steady_clock::now();
                if (mpCurrentObjectKeyFrame->GetVectorCovisibleKeyFrames().size() > 8)// ORB-SLAM是地图里的关键帧
                {
                    Optimizer::ObjectLocalBundleAdjustment(mpCurrentObjectKeyFrame, false);
                    auto t3 = std::chrono::steady_clock::now();
                    tproc = tproc + std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                    tba = tba + std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
                    time++;
                }

                // 剔除关键帧
                KeyFrameCulling();
            }
            // 将帧加入闭环检测
        }

        if(CheckFinish())
            break;
        usleep(3000);
        //break;
    }
    SetFinish();
}

}