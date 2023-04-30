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

    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentObjectKeyFrame = mlNewObjectKeyFrames.front();
        mlNewObjectKeyFrames.pop_front();
//        cout<<YELLOW<<"Object Keyframe"<<mpCurrentObjectKeyFrame->mObjTrackId<<"，queue leaves"<<mlNewObjectKeyFrames.size()<<"帧!";
//        cout<<"------";
//        for (auto pkf : mlNewObjectKeyFrames)
//        {
//            cout<<pkf->mObjTrackId<<" ";
//        }
//        cout<<WHITE<<endl;
    }


    const vector<MapObjectPoint*> vpMapObjectPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches();
    for(size_t i=0; i<vpMapObjectPointMatches.size(); i++)
    {
        MapObjectPoint* pMP = vpMapObjectPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentObjectKeyFrame))
                {
                        pMP->AddObservation(mpCurrentObjectKeyFrame, i);
                        pMP->UpdateNormalAndDepth();
                        pMP->ComputeDistinctiveDescriptors();
                }
                else
                {
                    mlpRecentAddedMapObjectPoints.push_back(pMP);
                }
            }
        }
    }
    mpCurrentObjectKeyFrame->UpdateConnections();
}

void ObjectLocalMapping::MapObjectPointCulling()
{
    list<MapObjectPoint*>::iterator lit = mlpRecentAddedMapObjectPoints.begin();
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
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f ) // (mnFound)/mnVisible,
        {

            count1++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }

//        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        else if(((int)nCurrentObjKFid-(int)pMP->mpRefObjKF->mnObjId)>=2 && pMP->Observations()<=cnThObs)
        {
            //cout<<pMP->mnId<<"  ";
            count2++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else if(((int)nCurrentObjKFid-(int)pMP->mpRefObjKF->mnObjId)>=3)
        {
            lit = mlpRecentAddedMapObjectPoints.erase(lit);
        }
        else
            lit++;
    }

}

void ObjectLocalMapping::SearchInNeighbors()
{

    int nn = 10;
    const vector<ObjectKeyFrame*> vpNeighKFs = mpCurrentObjectKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<ObjectKeyFrame*> vpTargetKFs;
    for(vector<ObjectKeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentObjectKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentObjectKeyFrame->mnId;
        const vector<ObjectKeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<ObjectKeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            ObjectKeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentObjectKeyFrame->mnId || pKFi2->mnId==mpCurrentObjectKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF = mpCurrentObjectKeyFrame->mnId;
        }
    }

    ORBmatcher matcher;
    vector<MapObjectPoint*> vpMapPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches();

    /*
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
        cout<<pKFi->mnFrameId<<":"<<endl;
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

        if (!vpMapPointMatches.empty())
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    int points = 0;

    vector<MapObjectPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    for(vector<ObjectKeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        ObjectKeyFrame* pKFi = *vitKF;
        vector<MapObjectPoint*> vpMapPointsKFi = pKFi->GetMapObjectPointMatches();
        points = points + vpMapPointsKFi.size();
        for(vector<MapObjectPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapObjectPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentObjectKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentObjectKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    if (!vpFuseCandidates.empty())
    matcher.Fuse(mpCurrentObjectKeyFrame,vpFuseCandidates);


    vpMapPointMatches = mpCurrentObjectKeyFrame->GetMapObjectPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapObjectPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    mpCurrentObjectKeyFrame->UpdateConnections();
}

void ObjectLocalMapping::KeyFrameCulling()
{
    vector<ObjectKeyFrame*> vpLocalKeyFrames = mpCurrentObjectKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<ObjectKeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        ObjectKeyFrame* pKF = *vit;
        if(pKF->mbFirstObserved == true)
            continue;
        const vector<MapObjectPoint*> vpMapPoints = pKF->GetMapObjectPointMatches();
        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapObjectPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(pKF->mvObjPointDepth[i]>pKF->mThDepth || pKF->mvObjPointDepth[i]<0)
                        continue;
                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvObjKeysUn[i].octave;
                        const map<ObjectKeyFrame*, size_t> observations = pMP->GetObservations();

                        int nObs=0;
                        for(map<ObjectKeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            ObjectKeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvObjKeysUn[mit->second].octave;
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
        if (CheckNewObjectKeyFrames())
        {
            auto t1 = std::chrono::steady_clock::now();
            ProcessNewObjectKeyFrame();
            MapObjectPointCulling();


            // CreateNewMapObjects();
            if (!CheckTheSameObject())
            {
                SearchInNeighbors();
                // BA
                auto t2 = std::chrono::steady_clock::now();
                if (mpCurrentObjectKeyFrame->GetVectorCovisibleKeyFrames().size() > 8)
                {
                    Optimizer::ObjectLocalBundleAdjustment(mpCurrentObjectKeyFrame, false);
                    auto t3 = std::chrono::steady_clock::now();
                    tproc = tproc + std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                    tba = tba + std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
                    time++;
                }

                KeyFrameCulling();
            }

        }

        if(CheckFinish())
            break;
        usleep(3000);
        //break;
    }
    SetFinish();
}

}