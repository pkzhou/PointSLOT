//
// Created by liuyuzhen on 2021/9/9.
//

#include "ObjectKeyFrame.h"
#include "MapObjectPoint.h"
#include "DetectionObject.h"
#include "MapObject.h"
#include "Frame.h"
#include "Converter.h"
#include "Parameters.h"
namespace ORB_SLAM2
{

long unsigned int ObjectKeyFrame::nNextId=0;
map<int,int> ObjectKeyFrame::mNextObjKFId;

ObjectKeyFrame::ObjectKeyFrame(Frame &pFrame, const size_t &nInFrameDetObjOrder, bool bFirstObserved):
mnFrameId(pFrame.mnId), mDescriptors(pFrame.mvObjPointsDescriptors[nInFrameDetObjOrder].clone()),
fx(pFrame.fx), fy(pFrame.fy), cx(pFrame.cx), cy(pFrame.cy), invfx(pFrame.invfx), invfy(pFrame.invfy),
mbf(pFrame.mbf), mb(pFrame.mb), mThDepth(pFrame.mThDepth),
mnMinX(pFrame.mnMinX), mnMinY(pFrame.mnMinY), mnMaxX(pFrame.mnMaxX), mnMaxY(pFrame.mnMaxY), mK(pFrame.mK),
mbBad(false), mnScaleLevels(pFrame.mnScaleLevels), mfScaleFactor(pFrame.mfScaleFactor), mfLogScaleFactor(pFrame.mfLogScaleFactor), mvScaleFactors(pFrame.mvScaleFactors), mvLevelSigma2(pFrame.mvLevelSigma2),
mvInvLevelSigma2(pFrame.mvInvLevelSigma2), mnFuseTargetForKF(0), mbNotErase(false), mbToBeErased(false),
mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(pFrame.mfGridElementWidthInv), mfGridElementHeightInv(pFrame.mfGridElementHeightInv),
mnBALocalForKF(0), mnBAFixedForKF(0), mbFirstConnection(true),  mpParent(NULL), mnTrackReferenceForFrame(0), mbFirstObserved(bFirstObserved)
{
    mnId=nNextId++;
    MapObject* pMO = pFrame.mvMapObjects[nInFrameDetObjOrder];
    DetectionObject* pDetObj = pFrame.mvDetectionObjects[nInFrameDetObjOrder];
    if(pMO == NULL || pDetObj ==NULL)
        assert(0);
    mObjTrackId = pMO->mnTruthID;
    if (mNextObjKFId.count(mObjTrackId)){
        mnObjId = mNextObjKFId[mObjTrackId];
        mNextObjKFId[mObjTrackId] = mNextObjKFId[mObjTrackId] + 1;
    }
    else{
        mnObjId = 0;
        mNextObjKFId.insert(make_pair(mObjTrackId, 0 + 1));
    }
    cout << YELLOW << "Object Keyframe" << mObjTrackId << "  First created: " << mbFirstObserved << " mnObjId: " << mnObjId << WHITE << endl;

    mpMapObjects = pMO;
    mpDetectionObject = pDetObj;
    g2o::SE3Quat x = pMO->GetCFInFrameObjState(pFrame.mnId).pose; // 设置状态, 注意输入Tco
    SetPose(x);
    mScale = pMO->GetCFInFrameObjState(pFrame.mnId).scale;


    mvpMapObjectPoints = pFrame.mvpMapObjectPoints[nInFrameDetObjOrder];
    mvObjKeysUn = pFrame.mvObjKeysUn[nInFrameDetObjOrder];
    mvObjPointDepth = pFrame.mvObjPointDepth[nInFrameDetObjOrder];
    mvuObjKeysRight = pFrame.mvuObjKeysRight[nInFrameDetObjOrder];

    mvObjKeysGrid = pFrame.mvObjKeysGrid[nInFrameDetObjOrder];

//    for (int i = 0; i < mvpMapObjectPoints.size(); ++i) {
//        auto pMP = mvpMapObjectPoints[i];
//        if (!pMP)
//            continue;
//        if (pMP->create==1) assert(0);
//        if (pMP->create!=1&&pMP->create!=2) assert(0);
//        if (pMP->mnId==-1) assert(0);
//    }
}


void ObjectKeyFrame::SetPose(const g2o::SE3Quat &cObjPose) // 要求输入的是Tco, 目标到camera的变换
{
    unique_lock<mutex> lock(mMutexPose);
    mTco = Converter::toCvMat(cObjPose);
    cv::Mat Rco = mTco.rowRange(0,3).colRange(0,3);
    cv::Mat tco = mTco.rowRange(0,3).col(3);
    cv::Mat Roc = Rco.t();
    mPoc = -Roc*tco;
    mToc = cv::Mat::eye(4,4,mTco.type());
    Roc.copyTo(mToc.rowRange(0,3).colRange(0,3));
    mPoc.copyTo(mToc.rowRange(0,3).col(3));
    //mScale = cObjState.scale;
    //mObjState = cObjState;
    //cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    //Cw = mToc*center;
}

cv::Mat ObjectKeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return mPoc.clone();
}

cv::Mat ObjectKeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTco.clone();
}


void ObjectKeyFrame::AddMapObjectPoint(MapObjectPoint* pMOP,  const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapObjectPoints[idx]=pMOP;
    if (pMOP->create==1) assert(0);
    if (pMOP->create!=1&&pMOP->create!=2) assert(0);
}


vector<MapObjectPoint*> ObjectKeyFrame::GetMapObjectPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapObjectPoints;
}

// 按照权重排序
void ObjectKeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,ObjectKeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<ObjectKeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<ObjectKeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    mvpOrderedConnectedKeyFrames = vector<ObjectKeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

vector<ObjectKeyFrame*> ObjectKeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

void ObjectKeyFrame::EraseConnection(ObjectKeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }
    if(bUpdate)
        UpdateBestCovisibles();
}

void ObjectKeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mbFirstObserved == true)// 首帧图像直接返回
            return;
        else if(mbNotErase)// TODO ?? 干嘛的
        {
            mbToBeErased = true;
            return;
        }
    }
    // 删除共视关键帧们与自己的联系(即共视的权重等)
    for(map<ObjectKeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);
    // 删除自己观测的地图点与自己的联系(即从地图点的观测中抹去自己)
    for(size_t i=0; i<mvpMapObjectPoints.size(); i++)
        if(mvpMapObjectPoints[i])
            mvpMapObjectPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);
        mConnectedKeyFrameWeights.clear();// 清空自己与其它共视关键帧之间的联系
        mvpOrderedConnectedKeyFrames.clear();
        set<ObjectKeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);
        // 为自己的子关键帧,找新的父关键帧
        while(!mspChildrens.empty())
        {
            bool bContinue = false;
            int max = -1;
            ObjectKeyFrame* pC;
            ObjectKeyFrame* pP;
            for(set<ObjectKeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)// 遍历每一个子关键帧，让它们更新它们指向的父关键帧
            {
                ObjectKeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;
                vector<ObjectKeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();// 子关键帧遍历每一个它的共视关键帧
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<ObjectKeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(*spcit == NULL) /// 可能有问题
                            continue;
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }
            if(bContinue)
            {
                pC->ChangeParent(pP);// 因为父节点死了，并且子节点找到了新的父节点，子节点更新自己的父节点
                sParentCandidates.insert(pC);// 因为子节点找到了新的父节点并更新了父节点，那么该子节点升级，作为其它子节点的备选父节点
                mspChildrens.erase(pC);// 该子节点处理完毕
            }
            else
                break;
        }
        // 如果还有子节点没有找到新的父节点
        if(!mspChildrens.empty())
            for(set<ObjectKeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);// 直接把父节点的父节点作为自己的父节点
            }
        mpParent->EraseChild(this);
        // 自己此时(目标在相机系下的位姿) * 父关键帧时刻(相机在目标系下的位姿) 这能得到一个什么东西?
        // mTcp = mTco*mpParent->GetPoseInverse();
        mbBad = true;
    }
    // 地图中删除该关键帧
    // mpMap->EraseKeyFrame(this);
    // BoW删除该关键帧
    // mpKeyFrameDB->erase(this);
}

cv::Mat ObjectKeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mToc.clone();
}

void ObjectKeyFrame::ChangeParent(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

void ObjectKeyFrame::EraseChild(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

int ObjectKeyFrame::GetWeight(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}


void ObjectKeyFrame::AddConnection(ObjectKeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF)) // 如果本关键帧的共视关键帧没有pKF, 就加进去
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight; // 不同则更新
        else
            return;
    }
    UpdateBestCovisibles(); // 需要重新排序
}


void ObjectKeyFrame::AddChild(ObjectKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}


set<ObjectKeyFrame*> ObjectKeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

ObjectKeyFrame* ObjectKeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

void ObjectKeyFrame::UpdateConnections()
{
    map<ObjectKeyFrame*,int> KFcounter;
    vector<MapObjectPoint*> vpMP; // 当前关键帧所有目标地图点先存起来
    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapObjectPoints;
    }

    for(vector<MapObjectPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++) // 找出来也观测到自己帧地图点关键帧, 并统计他们观测到(和自己帧)相同点的数量
    {
        MapObjectPoint* pMP = *vit;
        if(!pMP)
            continue;
        if(pMP->isBad()) // 判断点是不是坏的
            continue;

        map<ObjectKeyFrame*,size_t> observations = pMP->GetObservations(); //该点的所有观测, 第二个是索引
        for(map<ObjectKeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }
    if(KFcounter.empty())
        return;

    int nmax=0;
    ObjectKeyFrame* pKFmax=NULL;
    int th = 10;
    vector<pair<int,ObjectKeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<ObjectKeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax) // 找到最佳共视关键帧与(同地图点)数量
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th) // 如果共视程度大于阈值, 则更新共视帧和本帧之间的权重
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty()) // 如果前面共视程度均小于阈值,就只更新最多共视的那帧与自己的权重
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());// 满足阈值的关键帧,按照共视点数排序
    list<ObjectKeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++) // 权重与关键帧分开
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights = KFcounter; // 本帧所有的共视关键帧与对应权重(即相同目标点数)
        mvpOrderedConnectedKeyFrames = vector<ObjectKeyFrame*>(lKFs.begin(),lKFs.end()); // 本帧所有超过阈值的共视关键帧(已经排好序了)
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end()); // 本帧所有超过阈值的共视关键帧的权重

        if(mbFirstConnection && !mbFirstObserved) // ??
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();// 父关键帧 = (与本关键帧)权重最大的关键帧, TODO 为什么只执行一次
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}

vector<ObjectKeyFrame*> ObjectKeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<ObjectKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

cv::Mat ObjectKeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTco.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat ObjectKeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTco.rowRange(0,3).col(3).clone();
}

bool ObjectKeyFrame::IsInBBox(const float &u, const float &v) const
{
    double minX = mpDetectionObject->mrectBBox.x;
    double minY = mpDetectionObject->mrectBBox.y;
    double maxX = mpDetectionObject->mrectBBox.width + minX;
    double maxY = mpDetectionObject->mrectBBox.height + minY;
    return (u>=minX && u<maxX && v>=minY && v<maxY);
}

vector<size_t> ObjectKeyFrame::GetObjectFeaturesInArea(const float &x, const float  &y, const float  &r) const
{
    vector<size_t> vIndices;
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;
    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;
    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mvObjKeysGrid[ix][iy];
            if(vCell.empty())
                continue;
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvObjKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}

MapObjectPoint* ObjectKeyFrame::GetMapObjectPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapObjectPoints[idx];
}


void ObjectKeyFrame::ReplaceMapPointMatch(const size_t &idx, MapObjectPoint* pMP)
{
    mvpMapObjectPoints[idx] = pMP;
}

void ObjectKeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapObjectPoints[idx]=static_cast<MapObjectPoint*>(NULL);
}

void ObjectKeyFrame::EraseMapPointMatch(MapObjectPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapObjectPoints[idx]=static_cast<MapObjectPoint*>(NULL);
}

bool ObjectKeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

}