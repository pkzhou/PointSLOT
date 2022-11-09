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

#include "ORBmatcher.h"
#include "Parameters.h"
#include "Converter.h"
#include "DetectionObject.h"
#include "g2o_Object.h"
#include "MapObject.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"
#include"MapObjectPoint.h"
#include"TwoViewReconstruction.h"
#include "gms_matcher.h"
#include "MapObjectPoint.h"
#include "ObjectKeyFrame.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/tracking.hpp>

#include<limits.h>


#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;

using namespace cv;
namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_HIGH_FORDYNAMIC = 130;
const int ORBmatcher::RADIUS_FORDYNAMIC = 5;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;// th不为1.0的话, bFactor为真, 搜索半径就乘以th这个倍数得到新的搜索半径
    int t1 =0,t2 =0,t3 =0,t4 =0,t5 =0;//,t6 =0;
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;
        t1++;
        if(pMP->isBad())
            continue;
        t2++;
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        t3++;
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        //cout<<"最佳距离: "<<bestDist<<endl;
        if(bestDist<=TH_HIGH)
        {
            t4++;
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
            t5++;
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }
    //cout<<"静态局部3D跟踪测试: "<<t1<<" "<<t2<<" "<<t3<<" "<<t4<<" "<<t5<<endl;
    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &F, const size_t &nOrder, const vector<MapObjectPoint*> &vpMapPoints, const float th)
{
    if(F.mvDetectionObjects[nOrder]==NULL)
        assert(0);

    // TODO 为什么跟踪的点如此的少, 描述子距离为什么这么大???  58 跟踪6个


    int nmatches=0;
    const bool bFactor = th!=1.0;// th不为1.0的话, bFactor为真, 搜索半径就乘以th这个倍数得到新的搜索半径
    int t1 =0,t2 =0,t3 =0,t4 =0,t5 =0;//t6 =0;
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapObjectPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView) // 被标记点和不在视野中的调过
            continue;
        t1++;
        if(pMP->isBad())
            continue;
        t2++;
        const int &nPredictedLevel = pMP->mnTrackScaleLevel; // 决定搜索半径r
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        if(bFactor)
            r*=th;
        //const vector<size_t> vIndices = F.GetObjectFeaturesInArea(nOrder, 50, pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);
        //const vector<size_t> vIndices = F.GetObjectFeaturesInArea(nOrder, 50, pMP->mTrackProjX,pMP->mTrackProjY,20,nPredictedLevel-1,nPredictedLevel);
        //cout<<pMP->mnId<<" 绝:"<<pMP->mTrackProjX<<" "<<pMP->mTrackProjY<<endl;
        const vector<size_t> vIndices = F.GetObjectFeaturesInArea(nOrder, 50, pMP->mTrackProjX,pMP->mTrackProjY,5,nPredictedLevel-1,nPredictedLevel+1);

        if(vIndices.empty()) // 找到pMP投影点附近的点
            continue;
        t3++;
        const cv::Mat MPdescriptor = pMP->GetDescriptor();



        //const int* p = MPdescriptor.ptr<int32_t>();
        //cout<<"打印描述子:"<<MPdescriptor.rows<<endl;
        //cout<<MPdescriptor<<endl;

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)// Get best and second matches with near keypoints
        {
            const size_t idx = *vit;
            //cout<<"序号2: "<<idx<<endl;
            if(!F.isInBBox(nOrder, F.mvObjKeysUn[nOrder][idx].pt.x, F.mvObjKeysUn[nOrder][idx].pt.y)) // 匹配点要在方框中
                continue;

            if(F.mvpMapObjectPoints[nOrder][idx]) // 匹配点已经有一个比较好的地图点也跳过
                if(F.mvpMapObjectPoints[nOrder][idx]->Observations()>0)
                    continue;

            if(F.mvuObjKeysRight[nOrder][idx]>0) // 匹配点的右观测也要满足pMP的尺度
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuObjKeysRight[nOrder][idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mvObjPointsDescriptors[nOrder].row(idx); // 最大最小描述子距离及对应的尺度
            const int dist = DescriptorDistance(MPdescriptor,d);
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvObjKeysUn[nOrder][idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvObjKeysUn[nOrder][idx].octave;
                bestDist2=dist;
            }
        }
        //cout<<"最佳距离: "<<bestDist<<endl;
        if(bestDist<=TH_HIGH_FORDYNAMIC)// Apply ratio to second match (only if best and second are in the same scale level)
        {
            t4++;
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
            t5++;
            F.mvpMapObjectPoints[nOrder][bestIdx]=pMP;
            nmatches++;
        }
    }
    //cout<<"目标局部3D跟踪测试: "<<t1<<" "<<t2<<" "<<t3<<" "<<t4<<" "<<t5<<endl;
    return nmatches;
}



float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    //const float factor = 1.0f/HISTO_LENGTH;
    const float factor  = HISTO_LENGTH/360.0f;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}


int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    //const float factor = 1.0f/HISTO_LENGTH;
    const float factor  = HISTO_LENGTH/360.0f;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    //const float factor = 1.0f/HISTO_LENGTH;
    const float factor  = HISTO_LENGTH/360.0f;


    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    //const float factor = 1.0f/HISTO_LENGTH;
    const float factor  = HISTO_LENGTH/360.0f;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    if(EbDrawStaticFeatureMatches)
    {
        if (nmatches != 0)
        {
            std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
            keypoints_1 = pKF1->mvKeysUn;
            keypoints_2 = pKF2->mvKeysUn;
            cv::Mat img_goodmatch;
            string init_name = EstrDatasetFolder + "/image_0/";
            char frame1_id[24];
            char frame2_id[24];
            int fr1 = pKF1->mnFrameId;
            int fr2 = pKF2->mnFrameId;
            sprintf(frame1_id, "%06d", fr1);
            sprintf(frame2_id, "%06d", fr2);
            string strimg1 = init_name + frame1_id + ".png";
            string strimg2 = init_name + frame2_id + ".png";
            cv::Mat img_1 = cv::imread(strimg1);
            cv::Mat img_2 = cv::imread(strimg2);
            std::vector<cv::DMatch> good_matches;
            for (int i = 0; i < nmatches; i++)
            {
                cv::DMatch *match = new cv::DMatch(vMatchedPairs[i].first, vMatchedPairs[i].second, 0.1);
                good_matches.push_back(*match);
            }
            cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
            cv::imwrite("/home/liuyuzhen/match1.png", img_goodmatch);
        }
    }

    return nmatches;
}


int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();
    int t1=0, t2=0, t3=0, t4=0,t5=0, t6=0, t7=0, t8=0;
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;
        t1++;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF)) // 该点的观测是否包括该关键帧
            continue;
        t2++;
        cv::Mat p3Dw;

        p3Dw = pMP->GetWorldPos();

        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;
        t3++;
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;
        t4++;
        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 得到最远距离
        const float minDistance = pMP->GetMinDistanceInvariance(); // 得到最近距离?
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;
        t5++;
        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // view angle也不懂

        if(PO.dot(Pn)<0.5*dist3D)
            continue;
        t6++;
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        if(vIndices.empty())
            continue;
        t7++;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF); // 找到描述子距离最匹配的

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            t8++;
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations()) // 根据哪个点的观测更多, 留哪一个
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }
    //cout<<"Fuse静态测试: 关键帧"<<pKF->mnFrameId<<" "<<vpMapPoints.size()<<" "<<t1<<" "<<t2<<" "<<t3<<" "<<t4<<" "<<t5<<" "<<t6<<" "<<t7<<" "<<t8<<endl;
    return nFused;
}

int ORBmatcher::Fuse(ObjectKeyFrame *pKF, const vector<MapObjectPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rco = pKF->GetRotation(); // 得到当前帧的目标相机系下的pose
    cv::Mat tco = pKF->GetTranslation();
    cv::Mat Poc = pKF->GetCameraCenter();
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused=0;
    const int nMPs = vpMapPoints.size();

    int t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0, t8=0;
    for(int i=0; i<nMPs; i++)
    {
        MapObjectPoint* pMP = vpMapPoints[i];
        if(!pMP)
            continue;
        t1++;
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF)) // 该点的观测是否包括该关键帧
            continue;
        t2++;
        cv::Mat Poj = pMP->GetInObjFramePosition();
        cv::Mat Pcj = Rco * Poj + tco;
        if(Pcj.at<float>(2)<0.0f) // 相机系位置需要为正
            continue;
        t3++;
        const float invz = 1/Pcj.at<float>(2);
        const float x = Pcj.at<float>(0)*invz;
        const float y = Pcj.at<float>(1)*invz;
        const float u = fx*x+cx;
        const float v = fy*y+cy;
        const float ur = u-bf*invz;// 得到右图的投影点,有什么用
        if(!pKF->IsInBBox(u, v))// 该点是否在目标的2D框里面
             continue;
        t4++;
        cv::Mat oPcj = Poj-Poc; // 该点与相机之间的距离
        const float dist3D = cv::norm(oPcj); // 应该dist3D = cv::norm(cPcj)同样成立
        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 得到最远距离
        const float minDistance = pMP->GetMinDistanceInvariance(); // 得到最近距离?
        if(dist3D<minDistance || dist3D>maxDistance ) // 距离需要在尺度范围内
            continue;
        t5++;
        cv::Mat Pn = pMP->GetNormal(); // Viewing angle must be less than 60 deg, TODO 注意这个方向是在目标系下
        if(oPcj.dot(Pn)<0.5*dist3D)
            continue;
        t6++;
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF); // 预测尺度
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // 确定搜索半径
        const vector<size_t> vIndices = pKF->GetObjectFeaturesInArea(u,v,radius); // 然后在投影范围找可能的点
        if(vIndices.empty())
            continue;
        t7++;
        const cv::Mat dMP = pMP->GetDescriptor();// 描述子匹配
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            const cv::KeyPoint &kp = pKF->mvObjKeysUn[idx];
            const int &kpLevel= kp.octave;
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) // 如果与预测尺度相差较大
                continue;
            if(pKF->mvuObjKeysRight[idx]>=0) // 双目点
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuObjKeysRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8) // 比较下带信息矩阵的重投影误差有多大
                    continue;
            }
            else // 单目点
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            const int dist = DescriptorDistance(dMP,dKF); // 找到描述子距离最匹配的点
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW) // 找到了这样满足条件的点
        {
            t8++;
            MapObjectPoint* pMPinKF = pKF->GetMapObjectPoint(bestIdx); // 取出那个点
            if(pMPinKF) // 若点已经存在
            {
                if(!pMPinKF->isBad()) // 判断点是好是坏
                {
                    if(pMPinKF->Observations()>pMP->Observations()) // 根据哪个点的观测更多, 留哪一个
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else // 若点还不存在
            {
                pMP->AddObservation(pKF,bestIdx); // 给点加观测
                pKF->AddMapObjectPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }
    //cout<<"Fuse目标3D点测试: 关键帧"<<pKF->mnFrameId<<" "<<nMPs<<" "<<t1<<" "<<t2<<" "<<t3<<" "<<t4<<" "<<t5<<" "<<t6<<" "<<t7<<" "<<t8<<endl;
    return nFused;
}




int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    //const float factor = 1.0f/HISTO_LENGTH;
    const float factor  = HISTO_LENGTH/360.0f;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//TODO 为啥不计算描述子来计算,我觉得还可以计算描述子距离直接匹配.
/// 思路： 首先利用LK光流预测上一帧的目标ORB特征点在当前关键帧的像素位置，
/// 然后在该像素位置的附近搜索最匹配（距离最近）的（当前帧提取的）目标ORB特征点，
/// 然后若上一帧的目标特征点对应存在目标landmark，则顺势将此landmark赋给当前帧。该方法依赖于LK光流的精度，
/// 以及前后两帧是否都提取到了同一ORB特征点，若两帧的提取前后不一致，则也匹配不成功
void ORBmatcher::SearchByOpticalFlowTracking(const Frame &LastFrame,  const int &th, Frame &CurrentFrame, vector<int> &vnAllObjMatchesNum)
{
    if (LastFrame.mRawImg.rows == 0 || LastFrame.mRawImg.cols == 0)
    {
        cout<<"上一帧没有图像？"<<endl;
        exit(1);
    }
    //bool bTrack = false;
    vnAllObjMatchesNum.resize(LastFrame.mvDetectionObjects.size(), -1);

    for(size_t nInLastFrameDetObjOrder=0; nInLastFrameDetObjOrder <LastFrame.mnDetObj; nInLastFrameDetObjOrder++)
    {
        DetectionObject* LastDetObj = LastFrame.mvDetectionObjects[nInLastFrameDetObjOrder];
        if(LastDetObj == NULL)
            continue;
        int nInCurrentFrameDetObjOrder = CurrentFrame.FindDetectionObject(LastDetObj);
        if(nInCurrentFrameDetObjOrder == -1)
            continue;
        DetectionObject* CurrentDetObj = CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder];// 当前帧的cuboid观测
        size_t nCurrentFeaNum = CurrentFrame.mvObjKeys[nInCurrentFrameDetObjOrder].size();
        //vector<MapObjectPoint*> vpLastMOPs =  LastDetObj->GetInFrameMapObjectPoints();// 准备上一帧该object有地图点的关键点集合:vpInLastFrameCorresObjPts, 与(在帧里的)序号vnInLastFrameExsitMOPOrders
        vector<MapObjectPoint*> vpLastMOPs =  LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder];
        vector<cv::KeyPoint> vInLastFrameObjPts = LastFrame.mvObjKeysUn[nInLastFrameDetObjOrder];
        size_t nLastFeaNum = vInLastFrameObjPts.size();
        vector<DMatch> vDMacthes;
        vector<cv::Point2f> vLastPointPts, vCurrentPointPts;
        vLastPointPts.reserve(nLastFeaNum);
        vCurrentPointPts.reserve(nLastFeaNum);
        for(size_t n=0; n<nLastFeaNum; n++)
        {
            //if(vpLastMOPs[n] ==NULL)
              //  continue;
           vLastPointPts.push_back(vInLastFrameObjPts[n].pt);
        }
        std::vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(LastFrame.mRawImg, CurrentFrame.mRawImg, vLastPointPts, vCurrentPointPts, status, error);//, cv::Size(21, 21), 3


        int nMatchNum = 0;
        float fObjMeanDeltaX= 0;
        float fObjMeanDeltaY = 0;
        float fObjMeanDeltaNorm = 0;
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i])
            {
                nMatchNum++;
                fObjMeanDeltaX += vCurrentPointPts[i].x - vLastPointPts[i].x;
                fObjMeanDeltaY += vCurrentPointPts[i].y - vLastPointPts[i].y;
            }
        }

        if(nMatchNum == 0)
        {
            cout<<"this "<<endl;
            continue;
        }

        fObjMeanDeltaX /= float(nMatchNum);
        fObjMeanDeltaY /= float(nMatchNum);
        float mean_move_norm = sqrt(fObjMeanDeltaX * fObjMeanDeltaX + fObjMeanDeltaY * fObjMeanDeltaY);
        fObjMeanDeltaNorm = mean_move_norm;
        fObjMeanDeltaX /= mean_move_norm;
        fObjMeanDeltaY /= mean_move_norm;// 归一化， 平均移动方向

        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i])
            {
                /// 2.8.1 计算匹配点的移动方向: direct_x, direct_y, 移动距离: move_norm
                float direct_x = vCurrentPointPts[i].x - vLastPointPts[i].x;
                float direct_y = vCurrentPointPts[i].y - vLastPointPts[i].y;
                float move_norm = sqrt(direct_x * direct_x + direct_y * direct_y);
                direct_x /= move_norm;
                direct_y /= move_norm;

                // if mean_flow is very small, comparing angle is meaningless.
                /// 2.8.2 条件1: 如果平均移动距离和该特征点移动距离都较大, 但是二者的移动方向相差较多, 则舍弃该匹配点
                /// 该条件暂未使用
                /// 上面计算出了特征点的平均移动方向, 现在来比较他们每个与平均移动方向的角度是否大于30度,大于了就直接continue跳过(cos30=0.85)
                if(0)
                {
                    if (fObjMeanDeltaNorm > 20 && move_norm > 20)//TODO 如果两帧跟踪的光流移动的很小也不用比较光流.
                    {
                        if ((direct_x * fObjMeanDeltaX + direct_y * fObjMeanDeltaY) < 0.85) //cos45'=0.7 cos30=0.86  0.85  or 0.80
                        {
                            vCurrentPointPts[i] = cv::Point2f(0, 0); // ?
                            continue;
                        }
                    }
                }

                int nLastOctave = LastFrame.mvObjKeys[nInLastFrameDetObjOrder][i].octave;
                float radius = th * CurrentFrame.mvScaleFactors[nLastOctave]; // 搜索半经想一下



                int bestIdx2 = -1;
                float bestDist;
                bool bUseDescriptorFlag = true;
                if(bUseDescriptorFlag)
                {
                    vector<std::size_t> vIndices2;
                    vIndices2 = CurrentFrame.GetObjectFeaturesInArea(nInCurrentFrameDetObjOrder, nCurrentFeaNum, vCurrentPointPts[i].x,vCurrentPointPts[i].y,RADIUS_FORDYNAMIC,0,7);// 在预测点附近搜索备选ORB特征点： 搜索半径：RADIUS_FORDYNAMIC(自己设置的比较小), 在所有(0-7)的金字塔层级搜索
                    if(vIndices2.empty())
                        continue;

                    cv::Mat dMP;// 计算描述子距离, 为什么ORB-SLAM是计算地图点的描述子
                    dMP = LastFrame.mvObjPointsDescriptors[nInLastFrameDetObjOrder].row(i);
                    int bestDist = 256;
                    for(vector<std::size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)// 遍历备选ORB特征点
                    {
                        const std::size_t i2 = *vit;
                        if(CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][i2])// 如果该路标点已经匹配过了， 则跳过
                            continue;
                        const cv::Mat &d = CurrentFrame.mvObjPointsDescriptors[nInCurrentFrameDetObjOrder].row(i2);// 找到最近的描述子距离
                        const int dist = DescriptorDistance(dMP, d);
                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }
                    if(bestDist<=TH_HIGH_FORDYNAMIC)// 条件： 要求最近描述子距离小于阈值， 则找到匹配点
                        continue;
                }
                else{
                    bestIdx2 = CurrentFrame.GetCloestFeaturesInArea(nInCurrentFrameDetObjOrder, vCurrentPointPts[i].x,
                                                                    vCurrentPointPts[i].y, radius, nLastOctave - 1, nLastOctave + 1);
                    bestDist = 20;
                }
                if(bestIdx2 == -1)
                    continue;
                if(CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][bestIdx2])
                    continue;
                cv::DMatch match = cv::DMatch(nInLastFrameDetObjOrder, bestIdx2, bestDist); // distance 是描述子距离
                vDMacthes.push_back(match);
            }
        }

        // PNPRANSAC去除误匹配
        cv::Mat ransacInliersFlag;
        vector<pair<size_t, size_t>> vInlierMatches;
        size_t nMatchesNum = 0;
        ForObjectPnPRANSAC(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vDMacthes, ransacInliersFlag);
        // 地图点传递
        for(int l = 0; l<ransacInliersFlag.rows; l++)
        {
            //if(bTrack == false)
              //  bTrack = true;
            int nOr = ransacInliersFlag.at<int>(l);
            pair<size_t, size_t> vMatchTmp = make_pair(vDMacthes[nOr].queryIdx, vDMacthes[nOr].trainIdx);
            MapObjectPoint* pMP = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder][vMatchTmp.first];
            CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][vMatchTmp.second] = pMP;
            CurrentDetObj->AddMapObjectPoint(vMatchTmp.second, pMP);
            //pMP->AddInFrameObservation(CurrentFrame.mnId, nInCurrentFrameDetObjOrder, vMatchTmp.second);
            vInlierMatches.push_back(vMatchTmp);
            nMatchesNum++;
        }

        vnAllObjMatchesNum.push_back(nMatchesNum);
        if(1)
            DrawInFrameFeatureMatches(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vInlierMatches, true);// 画出当前object的匹配结果
    }

    return;
}

void ORBmatcher::ForObjectGMSFilter(DetectionObject* cLastCuboidTmp, DetectionObject* cCurrentCuboidTmp,
        vector<cv::KeyPoint> vLastkeyPoints, vector<KeyPoint> vCurrentKeyPoints, const vector<DMatch> &matches,
        vector<DMatch>& good_matches)
{
    int width1 = int(cLastCuboidTmp->mrectBBox.width);
    int height1 = int(cLastCuboidTmp->mrectBBox.height);
    double left1= cLastCuboidTmp->mrectBBox.x;
    double top1 = cLastCuboidTmp->mrectBBox.y;
    int width2 = int(cCurrentCuboidTmp->mrectBBox.width);
    int height2 = int(cCurrentCuboidTmp->mrectBBox.height);
    double left2= cCurrentCuboidTmp->mrectBBox.x;
    double top2 = cCurrentCuboidTmp->mrectBBox.y;

    vector<DMatch> forGMSDMatches;
    for(auto &k: matches)
    {
        cv::Point2f point1, point2;
        point1.x = vLastkeyPoints[k.queryIdx].pt.x - left1;
        point1.y = vLastkeyPoints[k.queryIdx].pt.y - top1;
        point2.x = vCurrentKeyPoints[k.trainIdx].pt.x - left2;
        point2.y = vCurrentKeyPoints[k.trainIdx].pt.y - top2;
        if(point1.x <0 || point1.y<0 || point2.x < 0 || point2.y <0)
            continue;
        vLastkeyPoints[k.queryIdx].pt = point1;
        vCurrentKeyPoints[k.trainIdx].pt = point2;
        forGMSDMatches.push_back(k);

    }
    std::vector<bool> vbInliers;
    gms_matcher gms(vLastkeyPoints, width1, height1, vCurrentKeyPoints, width2, height2, forGMSDMatches);
    gms.GetInlierMask(vbInliers, false, false);
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            good_matches.push_back(forGMSDMatches[i]);
        }
    }
}

void ORBmatcher::ForObjectORBFilter(const Mat &descriptors_1, const Mat &descriptors_2, const vector<DMatch> &matches, vector<DMatch> &good_matches)
{
    auto min_max = minmax_element(matches.begin(), matches.end(),
                              [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
}

void ORBmatcher::TwoFrameObjectPointsBruceMatching(const Frame &LastFrame, const Frame& CurrentFrame, const int &nInLastFrameDetObjOrder, const int &nInCurrentFrameDetObjOrder, const bool & bUseGMSFlag, vector<DMatch> &matches)
{
    // 1. 采用其他方法进行匹配: 暴力匹配 + GMS去除
    Mat descriptors_1, descriptors_2;
    std::vector<KeyPoint> keypoints_1 = LastFrame.mvObjKeysUn[nInLastFrameDetObjOrder];
    std::vector<KeyPoint> keypoints_2 = CurrentFrame.mvObjKeysUn[nInCurrentFrameDetObjOrder];
    std::vector<MapObjectPoint*> vpInLastFrameMOPs = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder];
    for(size_t m=0; m< keypoints_1.size(); m++)
    {
        descriptors_1.push_back(LastFrame.mvObjPointsDescriptors[nInLastFrameDetObjOrder].row(m));
    }
    for(size_t m=0; m<keypoints_2.size(); m++)
    {
        descriptors_2.push_back(CurrentFrame.mvObjPointsDescriptors[nInCurrentFrameDetObjOrder].row(m));
    }
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);// 暴力匹配

    vector<DMatch> vMOPMathes;
    for(auto &k: matches)
    {
        if(vpInLastFrameMOPs[k.queryIdx] == NULL)
        {
            continue;
        }
        vMOPMathes.push_back(k);
    }
    matches.clear();
    matches = vMOPMathes;

    //cout<<"采用暴力匹配点数: "<<matches.size()<<endl;
    if(1)
    {
        Mat matORBMatch;
        drawMatches(LastFrame.mRawImg, keypoints_1, CurrentFrame.mRawImg, keypoints_2, matches, matORBMatch);
        imshow("Bruce Match", matORBMatch);
        waitKey(0);
    }

    // 如果不用GMS筛选, 用基本的阈值策略进行筛选是否可行?


    if(bUseGMSFlag)
    {
        vector<DMatch> good_matches;
        ForObjectGMSFilter(LastFrame.mvDetectionObjects[nInLastFrameDetObjOrder], CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder], keypoints_1, keypoints_2, vMOPMathes, good_matches);// 通过matches得到 good_matches
        matches.clear();
        matches = good_matches;

        if(1)
        {
            Mat gmsORBMatch;
            drawMatches(LastFrame.mRawImg, keypoints_1, CurrentFrame.mRawImg, keypoints_2, matches, gmsORBMatch);
            imshow("gms ORB match", gmsORBMatch);
            waitKey(0);
        }
        cout<<"GMS筛选后匹配点数: "<<matches.size()<<endl;
    }



}

int ORBmatcher::SearchByBruceMatching(const Frame& LastFrame, const Frame& CurrentFrame, const int &nLastOrder, const int &nCurrenOrder, vector<MapObjectPoint *> &vpMapObjectPointMatches)
{
    // 我的方法是:描述子匹配找到最佳描述子, 然后进行剔除, 剔除分成两块: 旋转和尺度

    // 输出:就是当前帧的目标地图点
    vpMapObjectPointMatches = vector<MapObjectPoint* >(CurrentFrame.mvpMapObjectPoints[nCurrenOrder].size(), static_cast<MapObjectPoint*>(NULL));

    vector<MapObjectPoint*> vpMapLast = LastFrame.mvpMapObjectPoints[nLastOrder];

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = HISTO_LENGTH/360.0f;

    int nmatches = 0;
    vector<cv::DMatch> vDMatches;

    for(size_t i=0; i<vpMapLast.size(); i++) // 遍历上一帧该目标的所有点
    {
        MapObjectPoint* pMP = vpMapLast[i]; // 到底是应该用3D点的描述子还是2D点的描述子?
        if(!pMP || pMP->isBad() || LastFrame.mvbObjKeysOutlier[nLastOrder][i]) // 要求上一帧的3D点存在,不是坏点,也不是outlier
            continue;

        const cv::Mat &dLF = LastFrame.mvObjPointsDescriptors[nLastOrder].row(i); // 2d点描述子
        int bestDist1 = 256;
        int bestIdxCF = -1;
        int bestDist2 = 256;

        for(size_t iF = 0; iF < CurrentFrame.mvObjKeysUn[nCurrenOrder].size(); iF++)//遍历当前帧该目标的所有地图点
        {
            if(vpMapObjectPointMatches[iF]) // 表明该点已经匹配过了
                continue;
            const cv::Mat &dCF = CurrentFrame.mvObjPointsDescriptors[nCurrenOrder].row(iF);
            const int dist = DescriptorDistance(dLF, dCF);
            if(dist <bestDist1)
            {
                bestDist2 = bestDist1;
                bestDist1 = dist;
                bestIdxCF = iF; // 当前帧最佳匹配点序号
            }
            else if(dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if(bestDist1 <= TH_LOW)
        {
            if(static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
            {
                vpMapObjectPointMatches[bestIdxCF] = pMP; // 传递上一帧的3D点
                const cv::KeyPoint &kp = LastFrame.mvObjKeysUn[nLastOrder][i]; // 构建匹配点集的角度直方图
                if(mbCheckOrientation)
                {
                    float rot = kp.angle - CurrentFrame.mvObjKeys[nCurrenOrder][bestIdxCF].angle;
                    if(rot <0.0)
                        rot  += 360.0f;
                    int bin =round(rot * factor);
                    //cout<<"bin: "<<bin<<" rot: "<<rot<<" factor: "<<factor<<endl;

                    if(bin==HISTO_LENGTH)
                        bin = 0;

                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(bestIdxCF);
                }

                cv::DMatch match = cv::DMatch(i, bestIdxCF, bestDist1); // distance 是描述子距离
                vDMatches.push_back(match);

                nmatches++;
            }
        }

    }

    if(mbCheckOrientation) // 角度剔除
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i == ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend = rotHist[i].size(); j<jend; j++)
            {
                vpMapObjectPointMatches[rotHist[i][j]] = static_cast<MapObjectPoint*>(NULL);
                nmatches--;
            }
        }
    }


    // 尺度筛选
    // 首先判断目标是在前进还是在后退
    if(1)
    {

    }
    //cout<<"采用暴力匹配点数: "<<nmatches<<endl;
    //display
    if(0)
    {
        Mat matORBMatch;

        vector<cv::DMatch> tmpMatches;
        for(auto &m: vDMatches)
        {
            if(vpMapObjectPointMatches[m.trainIdx] == NULL)
                continue;
            tmpMatches.push_back(m);
        }
        std::vector<KeyPoint> keypoints_1 = LastFrame.mvObjKeysUn[nLastOrder];
        std::vector<KeyPoint> keypoints_2 = CurrentFrame.mvObjKeysUn[nCurrenOrder];
        drawMatches(LastFrame.mRawImg, keypoints_1, CurrentFrame.mRawImg, keypoints_2, tmpMatches, matORBMatch);
        imshow("Bruce New Match", matORBMatch);
        waitKey(0);
    }

    return nmatches;
}





void ORBmatcher::ForObjectPnPRANSAC(const Frame& LastFrame, const Frame& CurrentFrame, const int& nInLastFrameDetObjOrder, const int& nInCurrentFrameDetObjOrder, const vector<DMatch> &vDMacthes, cv::Mat &inliers)
{
    /******3D-2D PnPRANSAC****/
    vector<pair<size_t, size_t>> vInlierMatches;
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    MapObject* pMO = LastFrame.mvMapObjects[nInLastFrameDetObjOrder];
    //g2o::ObjectState cObjSta = pMO->GetInFrameObjState(LastFrame.mnId); // Two


    for(auto &m:vDMacthes)
    {
        MapObjectPoint* pMOP = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder][m.queryIdx];
        //Eigen::Vector3d wP = cObjSta.pose * pMOP->GetInObjFrameEigenPosition();
        Eigen::Vector3d oP = pMOP->GetInObjFrameEigenPosition();
        //cout<<oP<<endl;
        //pts_3d.push_back(cv::Point3f(wP[0], wP[1], wP[2]));
        pts_3d.push_back(cv::Point3f(oP[0], oP[1], oP[2]));
        pts_2d.push_back(CurrentFrame.mvObjKeysUn[nInCurrentFrameDetObjOrder][m.trainIdx].pt);

    }
    //cout<<"目标: "<<pMO->mnTruthID<<" RANSAC前匹配点数: "<<pts_3d.size()<<endl;
    Mat rvec, tvec;
    cv::solvePnPRansac(pts_3d, pts_2d, EK, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    //cout<<" R:"<<rvec<<"  t: "<<tvec<<endl;

    //cout<<EK<<endl;
    //cout<<inliers<<endl;
    // 设置 目标状态为RANSAC出来的状态
    if(1)
    {
        cv::Mat r;
        cv::Rodrigues(rvec,r);
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        cv::cv2eigen(r,R);
        cv::cv2eigen(tvec,t);
        g2o::SE3Quat Tco(R, t);

        g2o::SE3Quat Tco2 = pMO->GetCFInFrameObjState(CurrentFrame.mnId).pose;
        Eigen::Matrix3d Kalib = ORB_SLAM2::EdCamProjMatrix;
        // 计算重投影误差: e = (u,v) - K * Tco * Poj
        double e1 = 0, e2 = 0;
        for(size_t i=0; i<pts_3d.size(); i++)
        {
            Eigen::Vector3d Pj(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
            Eigen::Vector2d zj(pts_2d[i].x, pts_2d[i].y);
            Eigen::Vector3d localpt = Tco * Pj;
            Eigen::Vector2d projected(Kalib(0, 2) + Kalib(0, 0) * localpt(0) / localpt(2), Kalib(1, 2) + Kalib(1, 1) * localpt(1) / localpt(2));
            Eigen::Vector2d error = zj - projected;
            e1 += error.norm();

            Eigen::Vector3d localpt2 = Tco2 * Pj;
            Eigen::Vector2d projected2(Kalib(0, 2) + Kalib(0, 0) * localpt2(0) / localpt2(2), Kalib(1, 2) + Kalib(1, 1) * localpt2(1) / localpt2(2));
            Eigen::Vector2d error2 = zj - projected2;
            e2 += error2.norm();

            //cout<<"err: RANSAC前: "<<error2<<endl;
            //cout<<"RANSAC后: "<<error<<endl;
        }
        //cout<<"err: RANSAC前: "<<e2<<" RANSAC后: "<<e1<<endl;



        //g2o::SE3Quat PoseErr = CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder]->mTruthPosInCameraFrame.pose.inverse() * Tco;
        //cout<<"RANSAC后误差: "<<PoseErr.toMinimalVector()<<endl;

        //g2o::SE3Quat Two;
        //Two = CurrentFrame.mSETcw.inverse() * Tco;
        //g2o::ObjectState cObjCur = pMO->GetInFrameObjState(CurrentFrame.mnId);

        g2o::ObjectState cObjCur = pMO->GetCFInFrameObjState(CurrentFrame.mnId);
        //cout<<"原来: "<<cObjCur.pose<<endl;
        cObjCur.pose = Tco;
        //cout<<"现在: "<<cObjCur.pose<<endl;
        g2o::ObjectState Swo(CurrentFrame.mSETcw.inverse() * cObjCur.pose, cObjCur.scale);
        //pMO->SetInFrameObjState(Swo, CurrentFrame.mnId);
        //pMO->SetCFInFrameObjState(cObjCur, CurrentFrame.mnId);
    }


    //cout<<CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder]->mTruthPosInCameraFrame.pose<<endl;
    cout<<"目标: "<<pMO->mnTruthID<<"  RANSAC前匹配点数: "<<pts_3d.size()<<"  RANSAC后Inlier点数: "<<inliers.rows<<endl;
}


bool ORBmatcher::SearchByOfflineOpticalFlowTracking(const Frame &LastFrame, const vector<pair<size_t, size_t>> &vnpTrackedObjOrders, Frame &CurrentFrame, vector<int> &vnAllObjMatchesNum)
{
    if (LastFrame.mRawImg.rows == 0 || LastFrame.mRawImg.cols == 0)
    {
        cout<<"上一帧没有图像？"<<endl;
        exit(1);
    }
    bool bTrack = false;

    size_t nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder;
    for(size_t l=0; l<vnpTrackedObjOrders.size(); l++)
    {
        nInLastFrameDetObjOrder = vnpTrackedObjOrders[l].first;
        nInCurrentFrameDetObjOrder = vnpTrackedObjOrders[l].second;
        DetectionObject* cLastCuboidTmp = LastFrame.mvDetectionObjects[nInLastFrameDetObjOrder];
        DetectionObject* cCurrentCuboidTmp = CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder];// 当前帧的cuboid观测
        if(cLastCuboidTmp->mnObjectID != cCurrentCuboidTmp->mnObjectID)
            assert(0);
        size_t nCurrentFeaNum = CurrentFrame.mvObjKeys[nInCurrentFrameDetObjOrder].size();
        //vector<MapObjectPoint*> vpLastMOPs =  cLastCuboidTmp->GetInFrameMapObjectPoints();// 准备上一帧该object有地图点的关键点集合:vpInLastFrameCorresObjPts, 与(在帧里的)序号vnInLastFrameExsitMOPOrders
        vector<MapObjectPoint*> vpLastMOPs =  LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder];
        vector<cv::KeyPoint> vInLastFrameObjPts = LastFrame.mvObjKeysUn[nInLastFrameDetObjOrder];
        size_t nLastFeaNum = vInLastFrameObjPts.size();

        // 如果上一帧的目标地图点很少, 怎么办

        // method1: 离线光流跟踪
        vector<DMatch> vDMacthes;
        cv::Point2f pPointLast, pPointCurrent;
        cv::Vec2d OpticalFlow;
        for(size_t n=0; n<nLastFeaNum; n++)
        {
            if(vpLastMOPs[n] == NULL)
                continue;
            pPointLast = vInLastFrameObjPts[n].pt;
            OpticalFlow = LastFrame.mForwardOpticalImg.at<cv::Vec2d>(round(pPointLast.y), round(pPointLast.x));// 通过离线光流(imgFprwardOptical)预测上一帧特征点在当前帧的像素坐标: pPointCurrent
            pPointCurrent.x = OpticalFlow(0) + pPointLast.x;
            pPointCurrent.y = OpticalFlow(1) + pPointLast.y;

            //vpLastMOPs[n]->mTrackProjX = pPointCurrent.x;
            //vpLastMOPs[n]->mTrackProjY = pPointCurrent.y;

            //cout<<"赋值: "<<vpLastMOPs[n]->mnId<<" "<<pPointCurrent.x<<" "<<pPointCurrent.y<<endl;
            //cout<<vpLastMOPs[n]->mDescriptor<<endl;


            vector<std::size_t> vIndices2;
            vIndices2 = CurrentFrame.GetObjectFeaturesInArea(nInCurrentFrameDetObjOrder, nCurrentFeaNum, pPointCurrent.x,pPointCurrent.y,RADIUS_FORDYNAMIC,0,7);// 在预测点附近搜索备选ORB特征点： 搜索半径：RADIUS_FORDYNAMIC(自己设置的比较小), 在所有(0-7)的金字塔层级搜索


            if(vIndices2.empty())
                continue;

            cv::Mat dMP;// 计算描述子距离, 为什么ORB-SLAM是计算地图点的描述子
            dMP = LastFrame.mvObjPointsDescriptors[nInLastFrameDetObjOrder].row(n);

            //cout<<"跟踪上一帧描述子"<<endl;
            //cout<<   dMP<<endl;

            int bestDist = 256;
            int bestIdx2 = -1;
            for(vector<std::size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)// 遍历备选ORB特征点
            {
                const std::size_t i2 = *vit;
                //cout<<"序号1: "<<i2<<endl;
                if(CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][i2])// 如果该路标点已经匹配过了， 则跳过
                    continue;
                if(CurrentFrame.mvbObjKeysMatchedFlag[nInCurrentFrameDetObjOrder][i2] != false) // 只有逐点匹配才会用到这个, 不是逐点匹配不用
                    continue;
                const cv::Mat &d = CurrentFrame.mvObjPointsDescriptors[nInCurrentFrameDetObjOrder].row(i2);// 找到最近的描述子距离
                const int dist = DescriptorDistance(dMP, d);
                if(dist<bestDist)
                {
                    bestDist=dist;
                    bestIdx2=i2;
                }
            }
            //cout<<"best: "<<bestDist<<endl;
            if(bestDist<=TH_HIGH_FORDYNAMIC)// 条件： 要求最近描述子距离小于阈值， 则找到匹配点
            {
                cv::DMatch match = cv::DMatch(n, bestIdx2, bestDist); // distance 是描述子距离
                vDMacthes.push_back(match);
                CurrentFrame.mvbObjKeysMatchedFlag[nInCurrentFrameDetObjOrder][bestIdx2] = true;
            }
        }
        //cout<<"目标: "<<cCurrentCuboidTmp->mnObjectID<<" 跟踪点数: "<<vDMacthes.size()<<endl;

        // method2 : 暴力匹配 + GMS, 这里是否可以考虑在光流跟踪的基础上再暴力匹配
        size_t nMinRansacNum = 5;
        if(vDMacthes.size()<nMinRansacNum) // 采用暴力匹配
        {
            //CurrentFrame.mvbObjKeysMatchedFlag[nInCurrentFrameDetObjOrder] = vector<bool>(CurrentFrame.mvObjKeysUn[nInCurrentFrameDetObjOrder].size(), false);
            vector<cv::DMatch> vORBMatches;
            TwoFrameObjectPointsBruceMatching(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, true, vORBMatches);
            vDMacthes.clear(); // 完全是为了画图
            vDMacthes = vORBMatches;
            cout<<"不够, 重新采用暴力匹配点数: "<<vDMacthes.size();
            if(vDMacthes.size()<nMinRansacNum)// 如果暴力匹配的数量还是很少, 说明确实没有什么匹配
            {
                cout<<RED<<"  警告: 该目标跟踪点数过少!"<<endl<<WHITE;
                continue;
            }
            cout<<endl;
        }

        // PNPRANSAC去除误匹配
        cv::Mat ransacInliersFlag;
        vector<pair<size_t, size_t>> vInlierMatches;
        size_t nMatchesNum = 0;
        ForObjectPnPRANSAC(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vDMacthes, ransacInliersFlag);


        // 地图点传递
        for(int l = 0; l<ransacInliersFlag.rows; l++)
        //for(int l = 0; l<vDMacthes.size(); l++)
        {
            if(bTrack == false)
                bTrack = true;
            int nOr = ransacInliersFlag.at<int>(l);
            //nOr = l;
            pair<size_t, size_t> vMatchTmp = make_pair(vDMacthes[nOr].queryIdx, vDMacthes[nOr].trainIdx);
            MapObjectPoint* pMP = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder][vMatchTmp.first];
            CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][vMatchTmp.second] = pMP;
            cCurrentCuboidTmp->AddMapObjectPoint(vMatchTmp.second, pMP);
            //pMP->AddInFrameObservation(CurrentFrame.mnId, nInCurrentFrameDetObjOrder, vMatchTmp.second);
            vInlierMatches.push_back(vMatchTmp);
            nMatchesNum++;

            //break;
            //cout<<vMatchTmp.first<<" "<<vMatchTmp.second<<endl;
        }
        //cout<<"目标 "<<cLastCuboidTmp->mnObjectID<<" : RANSAC后Inlier点数"<<nMatchesNum<<endl;
        vnAllObjMatchesNum[l] = nMatchesNum;
        cout<<"目标"<<cLastCuboidTmp->mnObjectID<<" 跟踪上一帧3D点数: "<<nMatchesNum<<endl;
        if(nMatchesNum>10)
            cCurrentCuboidTmp->mbTrackOK = true;
        else
            cCurrentCuboidTmp->mbTrackOK = false;
        if(1)
            DrawInFrameFeatureMatches(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vInlierMatches, true);// 画出当前object的匹配结果
    }
    cout<<endl;
    return bTrack;
}

int ORBmatcher::SearchByOfflineOpticalFlowTracking(const Frame &LastFrame, Frame &CurrentFrame, const int &nInLastFrameDetObjOrder, const int& nInCurrentFrameDetObjOrder)
{
    size_t nMinNum = 10;

    DetectionObject* cLastCuboidTmp = LastFrame.mvDetectionObjects[nInLastFrameDetObjOrder];
    DetectionObject* cCurrentCuboidTmp = CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder];// 当前帧的cuboid观测
    if(cLastCuboidTmp->mnObjectID != cCurrentCuboidTmp->mnObjectID)
        assert(0);
    size_t nCurrentFeaNum = CurrentFrame.mvObjKeys[nInCurrentFrameDetObjOrder].size();
    vector<MapObjectPoint*> vpLastMOPs =  LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder];
    vector<cv::KeyPoint> vInLastFrameObjPts = LastFrame.mvObjKeysUn[nInLastFrameDetObjOrder];
    size_t nLastFeaNum = vInLastFrameObjPts.size();

    // 如果上一帧的目标地图点很少, 怎么办

    // method1: 离线光流跟踪
    vector<DMatch> vDMacthes;
    cv::Point2f pPointLast, pPointCurrent;
    cv::Vec2d OpticalFlow;
    for(size_t n=0; n<nLastFeaNum; n++)
    {
        if(vpLastMOPs[n] == NULL) // 如果上一帧3D点不存在就没必要匹配了
            continue;
        pPointLast = vInLastFrameObjPts[n].pt;
        OpticalFlow = LastFrame.mForwardOpticalImg.at<cv::Vec2d>(round(pPointLast.y), round(pPointLast.x));// 通过离线光流(imgFprwardOptical)预测上一帧特征点在当前帧的像素坐标: pPointCurrent
        pPointCurrent.x = OpticalFlow(0) + pPointLast.x;
        pPointCurrent.y = OpticalFlow(1) + pPointLast.y;

        vector<std::size_t> vIndices2;
        vIndices2 = CurrentFrame.GetObjectFeaturesInArea(nInCurrentFrameDetObjOrder, nCurrentFeaNum, pPointCurrent.x,pPointCurrent.y,RADIUS_FORDYNAMIC,0,7);// 在预测点附近搜索备选ORB特征点： 搜索半径：RADIUS_FORDYNAMIC(自己设置的比较小), 在所有(0-7)的金字塔层级搜索
        if(vIndices2.empty())
            continue;

        cv::Mat dMP;// 计算描述子距离, 为什么ORB-SLAM是计算地图点的描述子
        dMP = LastFrame.mvObjPointsDescriptors[nInLastFrameDetObjOrder].row(n);

        //cout<<"跟踪上一帧描述子"<<endl;
        //cout<<   dMP<<endl;

        int bestDist = 256;
        int bestIdx2 = -1;
        for(vector<std::size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)// 遍历备选ORB特征点
        {
            const std::size_t i2 = *vit;
            //cout<<"序号1: "<<i2<<endl;
            if(CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][i2])// 如果该路标点已经匹配过了， 则跳过
                continue;
            if(CurrentFrame.mvbObjKeysMatchedFlag[nInCurrentFrameDetObjOrder][i2] != false) // 只有逐点匹配才会用到这个, 不是逐点匹配不用
                continue;
            const cv::Mat &d = CurrentFrame.mvObjPointsDescriptors[nInCurrentFrameDetObjOrder].row(i2);// 找到最近的描述子距离
            const int dist = DescriptorDistance(dMP, d);
            if(dist<bestDist)
            {
                bestDist=dist;
                bestIdx2=i2;
            }
        }
        //cout<<"best: "<<bestDist<<endl;
        if(bestDist<=TH_HIGH_FORDYNAMIC)// 条件： 要求最近描述子距离小于阈值， 则找到匹配点
        {
            cv::DMatch match = cv::DMatch(n, bestIdx2, bestDist); // distance 是描述子距离
            vDMacthes.push_back(match);
            CurrentFrame.mvbObjKeysMatchedFlag[nInCurrentFrameDetObjOrder][bestIdx2] = true;
        }
    }

    if(vDMacthes.size()<nMinNum)
        return 0;

    // PNPRANSAC去除误匹配
    cv::Mat ransacInliersFlag;
    vector<pair<size_t, size_t>> vInlierMatches;
    ForObjectPnPRANSAC(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vDMacthes, ransacInliersFlag);

    // 地图点传递
    size_t nMatchesNum = 0;
    for(int l = 0; l<ransacInliersFlag.rows; l++)
        //for(int l = 0; l<vDMacthes.size(); l++)
    {

        int nOr = ransacInliersFlag.at<int>(l);
        //nOr = l;
        pair<size_t, size_t> vMatchTmp = make_pair(vDMacthes[nOr].queryIdx, vDMacthes[nOr].trainIdx);
        MapObjectPoint* pMP = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder][vMatchTmp.first];
        CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][vMatchTmp.second] = pMP;
        cCurrentCuboidTmp->AddMapObjectPoint(vMatchTmp.second, pMP);
        vInlierMatches.push_back(vMatchTmp);
        nMatchesNum++;
    }

    cout<<"目标"<<cCurrentCuboidTmp->mnObjectID<<" 跟踪上一帧3D点数: "<<nMatchesNum<<endl;

    if(0)
        DrawInFrameFeatureMatches(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vInlierMatches, true);// 画出当前object的匹配结果

    return nMatchesNum;
}


int ORBmatcher::SearchByBruceMatchingWithGMS(const Frame &LastFrame, Frame &CurrentFrame, const int &nInLastFrameDetObjOrder, const int &nInCurrentFrameDetObjOrder)
{
    size_t nMinNum = 10;
    DetectionObject* cLastCuboidTmp = LastFrame.mvDetectionObjects[nInLastFrameDetObjOrder];
    DetectionObject* cCurrentCuboidTmp = CurrentFrame.mvDetectionObjects[nInCurrentFrameDetObjOrder];// 当前帧的cuboid观测
    if(cLastCuboidTmp->mnObjectID != cCurrentCuboidTmp->mnObjectID)
        assert(0);
    // 1. 暴力匹配 with GMS
    vector<cv::DMatch> vORBMatches;
    TwoFrameObjectPointsBruceMatching(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, false, vORBMatches);
    if(vORBMatches.size()<nMinNum)
    {
        cout<<"目标暴力匹配点数太少! "<<vORBMatches.size()<<" 不足以进行RANSAC!"<<endl;
        return vORBMatches.size();
    }

    // 2. PnPRANSAC 去除误差匹配
    cv::Mat ransacInliersFlag;//  计算下PnP重投影误差:
    vector<pair<size_t, size_t>> vInlierMatches;// PNPRANSAC去除误匹配
    ForObjectPnPRANSAC(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vORBMatches, ransacInliersFlag);

    // 3. 地图点传递

    size_t nMatchesNum = 0;
    //for(int l = 0; l<ransacInliersFlag.rows; l++)
    for(int l = 0; l<vORBMatches.size(); l++)
    {
        //int nOr = ransacInliersFlag.at<int>(l);
        int nOr = l;
        pair<size_t, size_t> vMatchTmp = make_pair(vORBMatches[nOr].queryIdx, vORBMatches[nOr].trainIdx);
        MapObjectPoint* pMP = LastFrame.mvpMapObjectPoints[nInLastFrameDetObjOrder][vMatchTmp.first];
        CurrentFrame.mvpMapObjectPoints[nInCurrentFrameDetObjOrder][vMatchTmp.second] = pMP;
        cCurrentCuboidTmp->AddMapObjectPoint(vMatchTmp.second, pMP);
        vInlierMatches.push_back(vMatchTmp);
        nMatchesNum++;
    }

    cout<<"目标"<<cCurrentCuboidTmp->mnObjectID<<" 跟踪上一帧3D点数: "<<nMatchesNum<<endl;

    if(0)
        DrawInFrameFeatureMatches(LastFrame, CurrentFrame, nInLastFrameDetObjOrder, nInCurrentFrameDetObjOrder, vInlierMatches, true);// 画出当前object的匹配结果


    return nMatchesNum;
}




/// vMatchedPairs: vector<pair<size_t, size_t>>, 当前帧feature id， 上一帧 feature id
void ORBmatcher::DrawInFrameFeatureMatches(const Frame &pKF1, const Frame &pKF2, const int &nInLastFrameDetObjOrder, const int &nInCurrentFrameDetObjOrder, vector<pair<size_t, size_t> > &vMatchedPairs, bool Imwrite)
{
    if(vMatchedPairs.size() == 0)
        return;

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    keypoints_1 = pKF1.mvObjKeysUn[nInLastFrameDetObjOrder];
    keypoints_2 = pKF2.mvObjKeysUn[nInCurrentFrameDetObjOrder];
    cv::Mat img_goodmatch;
    string init_name = EstrDatasetFolder + "/image_02/";
    char frame1_id[24];
    char frame2_id[24];
    int fr1 = pKF1.mnId;
    int fr2 = pKF2.mnId;
    string strimg1, strimg2;
    // kitti 数据集
    if(ORB_SLAM2::EnDataSetNameNum == 0)
    {
        sprintf(frame1_id, "%06d", fr1);
        sprintf(frame2_id, "%06d", fr2);
        strimg1 = init_name + frame1_id + ".png";
        strimg2 = init_name + frame2_id + ".png";
    }
    // virtual kitti数据集
    else if(ORB_SLAM2::EnDataSetNameNum == 1)
    {
        sprintf(frame1_id, "%05d", fr1);
        sprintf(frame2_id, "%05d", fr2);
        strimg1 = init_name + "rgb_" + frame1_id + ".jpg";
        strimg2 = init_name + "rgb_" + frame2_id + ".jpg";
    }


    //cv::Mat img_1 = cv::imread(strimg1);
    //cv::Mat img_2 = cv::imread(strimg2);


    cv::Mat img_1 = pKF1.mRawImg;
    cv::Mat img_2 = pKF2.mRawImg;

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < vMatchedPairs.size(); i++) {
        cv::DMatch *match = new cv::DMatch(vMatchedPairs[i].first, vMatchedPairs[i].second, 0.1); // distance 是描述子距离
        good_matches.push_back(*match);
    }
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

    if(Imwrite == true)
    {
        char frame_index_c[256];
        sprintf(frame_index_c,"%06d", int(pKF1.mnId));
        string qian_name = "/home/liuyuzhen/SLOT_match_test20210324/match/";
        std::string img_name = qian_name + frame_index_c + ".png";
        cv::imwrite(img_name, img_goodmatch);
    }

    cv::imshow("Final Match", img_goodmatch);
    cv::waitKey(0);

}


int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
