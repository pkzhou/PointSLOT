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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Parameters.h"
#include "Converter.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "Tracking.h"
#include "MapObject.h"

#include <unistd.h>
//#include<mutex>

namespace ORB_SLAM2
{
static double tproc = 0;
static double tba = 0;
static int time = 0;
LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void Determin_chosealgorithm(int total_object_num, vector<bool>* firstoptimizeforme, vector<bool>* secondoptimizeforme)
{
    bool object0 = false;
    bool object1 = false;
    if(object0==true)
    {
        firstoptimizeforme->push_back(true);
        secondoptimizeforme->push_back(false);
    }
    else{
        firstoptimizeforme->push_back(false);
        secondoptimizeforme->push_back(true);
    }

    if(object1==true)
    {
        firstoptimizeforme->push_back(true);
        secondoptimizeforme->push_back( false);
    }
    else{
        firstoptimizeforme->push_back(false);
        secondoptimizeforme->push_back( true);
    }

}


/// 决定optimize_id, optimize_id2, need_decoupled的函数
/// 紧耦合, 松耦合直接决定: optimize_id, optimize_id2, need_decoupled
/// 切换耦合: 根据firstoptimizeforme与secondoptimizeforme决定optimize_id, optimize_id2, need_decoupled
void Determin_optimizeid(int *chose_algorithm, int total_object_num, vector<bool> firstoptimizeforme, vector<bool> secondoptimizeforme, bool* need_decoupled, vector<bool>* optimize_id, vector<bool>* optimize_id2)
{
    /// 1. 如果总共目标数为1: 即只有静态结构, 则采用切换耦合
    if(total_object_num==1)
    {
        *chose_algorithm = 0;
    }
    /// 2. 分配空间: 第一次, 第二次优化的objects: optimize_id, optimiza_id2
    optimize_id->reserve(total_object_num);
    optimize_id2->reserve(total_object_num);
    /// 3. 根据选择的算法决定: optimize_id, optimiza_id2,
    /// need_decoupled(在松耦合和切换耦合下为ture)
    switch(*chose_algorithm)
    {
        /// 3.1 紧耦合: 则optimize_id = [true true ... true],
        /// optimize_id2 = [false false ... false], need_decoupled = false
        case 0:{
            for(int i=0; i<total_object_num;i++)
            {
                optimize_id->push_back(true);
                optimize_id2->push_back(false);
            }
            *need_decoupled = false;
            break;
        }
        /// 3.2 松耦合: 则optimize_id = [true false ... false],
        /// optimize_id2 = [false true ... true], need_decoupled = true
        case 1:{
            for(int i=0; i<total_object_num;i++)
            {
                if(i==0)
                {
                    optimize_id->push_back(true);
                    optimize_id2->push_back(false);
                }
                else
                {
                    optimize_id2->push_back(true);
                    optimize_id->push_back(false);
                }
            }
            *need_decoupled = true;
            break;
        }
        /// 3.3 切换耦合: 则 optimize_id = firstoptimizeforme,
        /// optimize_id2 = secondoptimizeforme
        case 2:{
            /// 3.3.1 决定optimize_id: = firstoptimizeforme,
            /// 决定firstnum: 其为第一次优化object的个数(firstoptimizeforme的true的个数)
            int firstnum = 0;
            for(size_t i=0;i<firstoptimizeforme.size();i++)
            {
                bool x = firstoptimizeforme[i];
                optimize_id->push_back(x);
                if(x==true)
                {
                    firstnum++;
                }
            }
            /// 3.3.2 决定need_decoupled: 如果firstnum与总目标数相等, 则为false
            if(firstnum==total_object_num)
                *need_decoupled = false;
            else
                *need_decoupled = true;
            /// 3.3.3 决定optimize_id2: = secondoptimizeforme
            for(size_t i=0;i<secondoptimizeforme.size();i++)
            {
                bool x = secondoptimizeforme[i];
                optimize_id2->push_back(x);
            }
            break;
        }
    }
}
/// 理解多线程
/// 线程2: 回调函数为: localmapping中的run函数, 一直都处于在while中,
/// 首先setacceptkeyframes函数将标志位mbAcceptKeyFrames置为false,
/// 表示不接收插入关键帧(tracking线程回调用insert函数插入), 在tracking线程的needkeyframe函数中会对该标志位进行判定
/// 判定是否插入关键帧到mlNewKeyFrames队列;
/// 然后checknewkeyframes函数判定该队列是否为空, 不为空则处理; 为空(.....), 则把mbAcceptKeyFrames置为true
/// 休息3000us(等待tracking线程那边插入关键帧), 然后循环置标志位为false; 也就是说当localmapping在处理关键帧时
/// 是不准trackig线程插入关键帧, 只有它休息的时候可以插
void LocalMapping::Run ()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        /// 1. 告诉Tracking，LocalMapping正处于繁忙状态, LocalMapping线程处理的关键帧都是Tracking线程发过的
        /// 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快 TODO??
        SetAcceptKeyFrames(false); // 设置mbAcceptKeyFrames为false

        // Check if there are keyframes in the queue
        /// 2. 等待处理的关键帧列表不为空
        if(CheckNewKeyFrames())
        {
            auto t1 = std::chrono::steady_clock::now();
            /// 2.1 计算关键帧特征点的BoW映射，将关键帧插入地图 VI-A keyframe insertion
            //// 增加的部分主要为, 将object以及动态点与keyframe互相绑定,
            /// 意思就是可以直接得到这个object这些动态点被多少keyframe观测到
            ProcessNewKeyFrame();

            /// 2.2 剔除ProcessNewKeyFrame函数中引入的不合格MapPoints VI-B recent map points culling
            /// mlpRecentAddedMapPoints中的mappoints进行处理
            /// 若目标landmarks也在mlpRecentAddedMapPoints， 则也会被检查， 目前先不放入
            MapPointCulling();

            // Triangulate new MapPoints
            /// 2.3 相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints VI-C new map points creation
            CreateNewMapPoints();


            /// 2.5 已经处理完队列中的最后的一个关键帧
            if(!CheckNewKeyFrames())//如果队列为空
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                /// 2.5.1 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
                ///有动有静
                SearchInNeighbors();
            }

            mbAbortBA = false; // 在tracking线程中也在设置这个变量为什么不加锁
            auto t2 = std::chrono::steady_clock::now();
            /// 2.6 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            /// 进行localba
            if(!CheckNewKeyFrames() && !stopRequested())// mlNewKeyFrames队列不为空 且 没有停止要求信号mbStopRequested
            {
                /// 2.6.1 Local BA: VI-D
                if(mpMap->KeyFramesInMap()>2)
                {
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap); // TODO mbAbortBA到底什么意思
                    mbAbortBA = false;
                    auto t3 = std::chrono::steady_clock::now();
                    tproc = tproc + std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                    tba = tba + std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
                    time++;
                }

                // Check redundant local Keyframes
                /// 2.6.2 检测并剔除当前帧相邻的关键帧中冗余的关键帧: VI-E,
                /// 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到, trick!
                /// Tracking中先把关键帧交给LocalMapping线程,
                /// 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                /// 在这里再删除冗余的关键帧
                KeyFrameCulling();
            }
            /// 2.7 将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop()) // 如果队列为空, mbStopRequested为真 且mbNotStop为假
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish()) // mbStopped为真,且 mbFinishRequested为假, 就一直休眠
            {
                usleep(3000);
            }
            if(CheckFinish()) // 若mbFinishRequested为真则跳出来
                break;
        }
        /// 4. ?? TODO
        ResetIfRequested();  // 重设mlNewKeyFrames.clear(); mlpRecentAddedMapPoints.clear();

        // Tracking will see that Local Mapping is busy
        /// 5. 设置可以接收关键帧标志位
        SetAcceptKeyFrames(true); // 设置mbAcceptKeyFrames为真

        /// 6. ?? TODO
        if(CheckFinish()) // 若mbFinishRequested为真, 则跳出来
            break;

        usleep(3000);
    }

    SetFinish();// 设置 mbFinished mbStopped 均为真
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}


/// 处理列表中的关键帧
/// (1) 计算Bow，加速三角化新的MapPoints
/// (2) 关联当前关键帧到静态MapPoints(将双目产生的landmarks放入检查队列)，
/// 并更新MapPoints的平均观测方向和观测距离范围
/// (3) 关联当前关键帧到目标landmarks与目标， 并更新目标landmarks的平均观测方向和观测距离范围
/// (4) 插入关键帧，更新Covisibility图和Essential图
void LocalMapping::ProcessNewKeyFrame()
{
    /// 1. 从缓冲队列中取出一帧关键帧
    /// Tracking线程向LocalMapping中插入关键帧存在该队列中
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 从列表中获得一个等待被插入的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    /// 2. 计算该关键帧特征点的Bow映射关系
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    /// 3. 跟踪局部地图过程中新匹配上的静态MapPoints和当前关键帧绑定
    /// 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配，
    /// 但没有对这些匹配上的MapPoints与当前帧进行关联
    /// 3.1 得到当前关键帧的所有动态静态点
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        /// 3.2 遍历所有静态点
        /// 3.3 检验条件: pMP存在, pMP不是Bad, pMP不是当前关键帧生成的实际跟踪得到的
        MapPoint* pMP = vpMapPointMatches[i];
        /// (1) 条件1:
        if(pMP)
        {
            /// (2) 条件2:
            if(!pMP->isBad())
            {
                /// (3) 条件3: 不是当前关键帧生成的mappoints, 则是跟踪得到的
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    /// 3.4.1 添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    /// 3.4.2 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    /// 3.4.3 加入关键帧后，更新3d点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else
                {
                    /// 3.4.4 该landmark是当前关键帧生成的: 将双目或RGBD跟踪过程中新插入的MapPoints放入
                    /// mlpRecentAddedMapPoints，等待检查
                    /// 此外， CreateNewMapPoints函数中通过三角化也会生成MapPoints,
                    /// 这些MapPoints都会经过MapPointCulling函数的检验
                        mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }


    // Update links in the Covisibility Graph
    /// 5. 更新关键帧间的连接关系，Covisibility图和Essential图(tree)
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    /// 6. 将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/// 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
/// 检查mlpRecentAddedMapPoints中的点
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    /// 1. 获取等待检查的mappoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    /// 2.????TODO
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    /// debug
    //cout<<RED<<"Mappointculling start:  "<<endl;
    cout<<WHITE<<endl;
    //int dy_badnum = 0;
    /// 3. 遍历mlpRecentAddedMapPoints中的所有landmark: 包括动态和静态
    int count1 = 0, count2 = 0;
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        /// 3.1 判断1: 该landmark是bad,则从容器中删除
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {   /// 判断2: 将不满足VI-B条件的MapPoint剔除
            /// VI-B 条件1：
            /// 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
            /// IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
            /// 并且设置badflag
            pMP->SetBadFlag();
            count1++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            /// 判断3：将不满足VI-B条件的MapPoint剔除
            /// VI-B 条件2：从该点建立开始，到现在已经过了不小于2个关键帧
            /// 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格

            pMP->SetBadFlag();
            count2++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            /// 判断4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            /// 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }

    //cout<<"静态3D点删除 "<<count1<<" "<<count2<<" 个"<<endl;

}

/// 相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    /// 1. ???TODO 设置nn
    int nn = 10;
    if(mbMonocular)
        nn=20;
    /// 2. 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    /// 3. 建立ORBmatcher类
    ORBmatcher matcher(0.6,false);

    /// 4. 获取当前关键帧的pose信息, 相机内参
    /// 4.1 旋转
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    /// 4.2 平移
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    /// 4.3 相机中心点世界系位置
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
    /// 4.4 相机内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    /// 4.5 ???TODO
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    /// 5. 遍历相邻关键帧vpNeighKFs, 分别与当前关键帧匹配, 并进行三角化
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        /// 5.1 TODO 什么意思 check newkeyframe
        if(i>0 && CheckNewKeyFrames())
            return;

        /// 5.2 遍历相邻关键帧
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        /// 5.3 获取邻接的关键帧在世界坐标系中的坐标: Ow2
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        /// 5.4 计算当前关键这与邻接关键帧之间的相机位移: vBaseline, 向量
        cv::Mat vBaseline = Ow2-Ow1;
        /// 5.5 得到向量长度: 基线长度
        const float baseline = cv::norm(vBaseline);

        /// 5.6 分单目双目情况讨论: 判断相机运动的基线是不是足够长, 不够长则不生成3D点
        if(!mbMonocular)
        {
            /// (1) 如果是双目相机或者RGBD, 则关键帧间距太小时则不生成3D点
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            /// (2) 是单目相机, 则判断ratioBaselineDepth, 太小也不生成
            /// 1) 获取邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            /// 2) 计算baseline与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            /// 3) 如果特别远(比例特别小)，那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        /// 5.7 根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        /// 5.8 两关键帧进行特征点匹配: 通过极线约束限制匹配时的搜索范围, 结果在vMatchedIndices
        /// vMatchedIndices: <特征点在当前关键帧的序号, 匹配特征点在邻接关键帧的序号>
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        /// 5.9 获取邻接关键帧的pose, 相机内参等
        /// 5.9.1 旋转
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        /// 5.9.2 平移
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        /// 5.9.3 位姿矩阵Tcw2
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));
        /// 5.9.4 相机内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        /// 5.10 对每对匹配通过三角化生成3D点, 和Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            /// 5.10.1 遍历每对匹配点
            /// 5.10.1.1 当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            /// 5.10.1.2 当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;
            /// 5.10.1.3 当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            /// 5.10.1.4 mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0; /// 判断当前关键帧是不是双目
            /// 5.10.1.5 当前匹配在邻接关键帧中的特征点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            /// 5.10.1.6 mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0; /// 判断邻接关键帧是不是双目

            // Check parallax between rays
            /// 5.10.2 利用匹配点反投影得到视差角
            /// 5.10.2.1 特征点反投影, 投影到相机系
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
            /// 5.10.2.2 由相机坐标系转到世界坐标系，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
            /// 5.10.2.3 初始化cosParallaxStereo, cosParallaxStereo1, cosParallaxStereo2
            /// 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            /// 5.10.2.4 如果是双目, 则利用双目得到视差角
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
            /// 5.10.2.5 得到双目观测的视差角, TODO 为什么是这么算???
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            /// 5.10.3 三角化恢复3D点
            cv::Mat x3D;
            /// 5.10.3.1 条件:
            /// cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998) 表明视差角正常
            /// cosParallaxRays<cosParallaxStereo表明视差角很小
            /// 视差角度小时用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                /// 5.10.3.2 线性三角化
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;
                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                /// 如果关键帧1 双目有效, 直接用它双目三角化
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                /// 如果关键帧2 双目有效,  则用它双目三角化
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else /// 都不满足, 则不三角化
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            /// 5.10.4 三角化完成, 进行检验
            /// 5.10.4.1 检测生成的3D点是否在相机前方
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            /// 5.10.4.2 计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            /// 5.10.4.3 计算3D点在另一个关键帧下的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                // 基于卡方检验计算出的阈值（假设测量有一个一个像素的偏差）
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            /// 5.10.4.4 检查尺度连续性
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);
            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);
            if(dist1==0 || dist2==0)
                continue;
            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];
            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // ratioDist*ratioFactor < ratioOctave 或 ratioDist/ratioOctave > ratioFactor表明尺度变化是连续的
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;
            // Triangulation is succesfull
            /// 5.10.5 三角化生成3D点成功，构造成MapPoint
            /// 5.10.5.1 构造mappoints
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            /// 5.10.5.2 为该MapPoint添加属性
            /// a.观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);
            /// b.该MapPoint的描述子
            pMP->ComputeDistinctiveDescriptors();
            /// c.该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();
            /// 5.10.5.3 将该landmark加入地图
            mpMap->AddMapPoint(pMP);
            /// 5.10.5.5 将新产生的点放入检测队列: mlpRecentAddedMapPoints
            /// 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);
            nnew++;
        }
    }
}



void LocalMapping::SearchInNeighbors()
{
    /// 步骤1：获得当前关键帧在covisibility图中权重排名前nn的邻接关键帧
    /// 找到当前关键帧的一级相邻与二级相邻关键帧放入： vpTargetKFs
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            // 这个地方有问题, 应该标记pKFi2, 为什么没有标记
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        }
    }

    /// 步骤2： 静态点
    /// (1) 将当前帧的静态点分别与一级二级相邻帧(的MapPoints)进行融合
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); //
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        /// 投影当前帧的MapPoints到相邻关键帧pKFi中，并判断是否有重复的MapPoints
        /// 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint， 那么将两个MapPoint合并（选择观测数多的）
        /// 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
        matcher.Fuse(pKFi,vpMapPointMatches);

        // TODO 为啥不做pKFi共视图的更新, 不做每帧地图点的描述子和观测方向,最大\最小观测距离的重新计算
    }

    /// 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    /// (2) 将一级二级相邻帧的MapPoints分别与当前帧（的MapPoints）进行融合
    ///  遍历每一个一级邻接和二级邻接关键帧
    int allpoints = 0;
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); \
    vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        allpoints = allpoints + vpMapPointsKFi.size();
        /// 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), \
        vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    int nFuse = matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    //cout<<"融合的KF数目 "<<vpTargetKFs.size()<<", 融合地图点总量 "<<allpoints<<", 实际fuse数量 "<<nFuse<<endl;
    /// (3) 更新当前帧MapPoints的描述子，深度，观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }
    /// (4) 更新更新当前帧的MapPoints后更新与其它帧的连接关系， 即：更新covisibility图
    mpCurrentKeyFrame->UpdateConnections();

}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped) //
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
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

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
    cout<<"LocalMappping Time----proc: "<<tproc/time*1000<<", BA: "<<tba/time*1000<<endl;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
