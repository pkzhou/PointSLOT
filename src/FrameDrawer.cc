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

#include "FrameDrawer.h"
#include "Tracking.h"
#include "MapPoint.h"
#include "Parameters.h"
#include "DetectionObject.h"
#include "MapObject.h"
#include "Map.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));

    /*
    mvColors.push_back(cv::Scalar(255, 255, 0));
    mvColors.push_back(cv::Scalar(255, 0, 255));
    mvColors.push_back(cv::Scalar(0, 255, 255));
    mvColors.push_back(cv::Scalar(145, 30, 180));
    mvColors.push_back(cv::Scalar(210, 245, 60));
    mvColors.push_back(cv::Scalar(128, 0, 0));*/

    /*
    mvColors.push_back(cv::Scalar(230, 0, 0));
    mvColors.push_back(cv::Scalar(60, 180, 75));
    mvColors.push_back(cv::Scalar(0, 0, 255));
    mvColors.push_back(cv::Scalar(255, 0, 255));
    mvColors.push_back(cv::Scalar(255, 165, 0));
    mvColors.push_back(cv::Scalar(128, 0, 128));
    mvColors.push_back(cv::Scalar(0, 255, 255));
    mvColors.push_back(cv::Scalar(210, 245, 60));
    mvColors.push_back(cv::Scalar(250, 190, 190));
    mvColors.push_back(cv::Scalar(0, 128, 128));*/

    mvColors.push_back(cv::Scalar(0, 0, 230));
    mvColors.push_back(cv::Scalar(75, 180, 60));
    mvColors.push_back(cv::Scalar(255, 0, 0));
    mvColors.push_back(cv::Scalar(255, 0, 255));
    mvColors.push_back(cv::Scalar(0, 165, 255));
    mvColors.push_back(cv::Scalar(128, 0, 128));
    mvColors.push_back(cv::Scalar(255, 255, 0));
    mvColors.push_back(cv::Scalar(60, 245, 210));
    mvColors.push_back(cv::Scalar(190, 190, 250));
    mvColors.push_back(cv::Scalar(128, 128, 0));


}

/// 画图像窗口: 包括图像、特征点、地图、跟踪状态
cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys;
    vector<int> vMatches;
    vector<cv::KeyPoint> vCurrentKeys;
    vector<bool> vbVO, vbMap; // 被关键帧观测到过vbMap就为true， 若关键帧从未观测到过则vbVO为true
    int state;

    // 把类变量给到临时变量： (正常跟踪)特征点vCurrentKeys， (初始)特征点vCurrentKeys,  mvIniMatches: 单目初始化才会用到
    // 动态特征点mvCurrentKeysdynamic(不分初始化); 标志位:mvbVO mvbMap为true代表该特征被关键帧观测到
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;
        mIm.copyTo(im);

        // 初始化
        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;

        }
        // 正常跟踪
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        // 跟踪丢失
        else if(mState==Tracking::LOST)//TODO lost就不用赋值mappoint
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if(im.channels()<3)
        cvtColor(im,im,CV_GRAY2BGR);

    // 画出单目初始化到当前的连线： ORB-SLAM
    // 只有采用了它的单目初始化才会画
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,cv::Scalar(0,255,0));
            }
        }
    }
    // 画出当前的静态特征点
    else if(state==Tracking::OK)
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            //vbVO和vbMap分别代表有没有被关键帧观测到
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;
                if(vbMap[i]) //被关键帧观测到过
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1); // 绿色
                    mnTracked++;
                }
                else{ // 不会进到这里面,说明所有静态点都被关键帧观测到了
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1); // 蓝色
                    mnTrackedVO++;
                }
            }
        }
    }

    if(EnSLOTMode >=2) // 目标跟踪或自动驾驶模式才画帧上目标图
    {
        //FIXME 赋值过程经常会少一个元素？？？
        vector<vector<cv::KeyPoint>> vCurrentKeysdynamic(mvCurrentFrameObjKeys);
        vector<DetectionObject *> vCurrentObjects(mvDetectionObjects);
        auto vbObjMap = mvbObjMap;
        auto vbObjVO = mvbObjVO;


        mnTrackedObjectPoints=0;
        mnTrackedVOObjectPoints=0;
        const float r = 5;
        for(std::size_t i=0; i<mvDetectionObjects.size(); i++)
        {
            /// 1. 画2d bounding box.
            DetectionObject* cuboidTmp = mvDetectionObjects[i];
            if(cuboidTmp==NULL)
                continue;

                cv::Rect  bbox = cuboidTmp->GetBBoxRect();
//                bbox.x = bbox.x-50;
//                bbox.width = bbox.width+40;
                // 颜色是object id的颜色
                cv::rectangle(im, bbox, mvColors[cuboidTmp->mnObjectID % mvColors.size()], 2);
                string id = to_string(cuboidTmp->mnObjectID);
                string TrackID = "TrackID: " + id;
                auto text_color =  cv::Scalar(255, 255, 255);
                if (cuboidTmp->GetDynamicFlag())
                    text_color =  cv::Scalar(0, 0, 255);
                cv::putText(im,TrackID,cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, 3);
                // 需不需要写， 该object建立成3D的 和没有被建立成3d的 画虚线和实线
                auto pMO = cuboidTmp->GetMapObject();
                if (pMO){
                    if (!pMO->GetDynamicFlag())
                        ;
                    else{
                        for(size_t j=0; j<mvCurrentFrameObjKeys[i].size(); j++)
                        {
                            if (mvbObjMap[i][j])
                            cv::circle(im,mvCurrentFrameObjKeys[i][j].pt,3,mvColors[cuboidTmp->mnObjectID % mvColors.size()],-1);
                            if (mvbObjVO[i][j])
                            cv::circle(im,mvCurrentFrameObjKeys[i][j].pt,2,cv::Scalar(0,0,0),-1);
                        }
                    }
                }



        }
    }
    //写文字信息
    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);
    if (frame == 212)
        cv::imwrite("000212.png",im);


    return imWithInfo;

}

/// 在图上写文字信息
void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    /// 1. 建立流: s
    stringstream s;
    /// 2. 根据state状态决定s的值
    /// 2.1 nState == NO_IMAGES_YET
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    /// 2.2 nState==NOT_INITIALIZED
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        /// 2.3 nState==OK
        /// 2.3.1 SLAM MODE 模式
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            /// 2.3.2 LOCALIZATION 模式
            s << "LOCALIZATION | ";
        /// 2.3.3 获取地图中的关键帧数量: nKFs
        int nKFs = mpMap->KeyFramesInMap();
        /// 2.3.4 获取地图中的所有静态地图点数量: nMPs
        int nMPs = mpMap->MapPointsInMap();
        /// 2.3.5 写入s
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        /// 2.3.6 mnTrackedVO???TODO 这个变量啥意思
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        /// 2.4 nState==LOST
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        /// 2.5 nState==SYSTEM_NOT_READY
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    /// 3. 决定字体大小
    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);
    /// 4. 构造比原图像更大的mat: imText 多余的部分打印字体
    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    /// 5. imText上部分放原来的im 图像
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    /// 6. 打印s到多出来的部分 TODO ???
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

///将跟踪线程的数据拷贝到绘图线程: 图像、特征点、地图、跟踪状态
void FrameDrawer::Update(Tracking *pTracker)
{
    /// 1. 第一步开始上锁
    unique_lock<mutex> lock(mMutex);
    /// 2. 拷贝跟踪线程的图像到成员变量mIm
    pTracker->mImGray.copyTo(mIm);
    /// 3. 拷贝跟踪线程的静态特征点
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    //mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
    mbOnlyTracking = pTracker->mbOnlyTracking;
    frame = pTracker->mCurrentFrame.mnId;

    /// 5. 根据tracking状态拷贝mvIniKeys, mvIniMatches; mvbMap, mvbVO
    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        /// 5.1 状态为NOT_INITIALIZED, 拷贝mvIniKeys: 初始静态特征点,
        /// 拷贝mvIniMatches, 即correspondence的序号
        mvIniKeys=pTracker->mInitialFrame.mvKeys;// 这个initialframe是只有单目初始化才有
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        /// 5.2 状态为OK, 拷贝静态: mvbMap, mvbVO
        for(int i=0;i<N;i++)
        {
            /// 5.2.1 遍历当前帧的所有mappoints
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                /// 5.2.2 判断该mappoint的outlier标志位: 只需要不是outlier的
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    /// 5.2.2.1 如果该landmark的观测次数大于0, 则它是地图点
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        /// 5.2.2.2 如果观测次数为0, 则是visual odometry, TODO 为什么
                        mvbVO[i]=true;
                }
            }
        }

    }
    /// 6. 拷贝上一次跟踪状态: mLastProcessedState
    mState=static_cast<int>(pTracker->mLastProcessedState);

    /// 7. 拷贝动态目标的: 2D框,  8个顶点( TODO 在图像平面??), 目标id
    if(EbSLOTFlag)
    {
        mvDetectionObjects = pTracker->mCurrentFrame.mvDetectionObjects;
        mvCurrentFrameObjKeys = pTracker->mCurrentFrame.mvObjKeys;
        mnDetObjNum = pTracker->mCurrentFrame.mnDetObj;
        vector<vector<bool>> table(mvCurrentFrameObjKeys.size());
        mvbObjVO = table;
        mvbObjMap = table;

        for (int i = 0; i < mvDetectionObjects.size(); ++i) {
            DetectionObject* cuboid = mvDetectionObjects[i];
            if (!cuboid)
                continue;
            vector<MapObjectPoint*> pMOPs = pTracker->mCurrentFrame.mvpMapObjectPoints[i];
            vector<bool> vMap(pMOPs.size(),false);
            vector<bool> vVO(pMOPs.size(),false);
            for (int j = 0; j < pMOPs.size(); ++j)
            {
                MapObjectPoint* pMOP = pMOPs[j];
                if (pMOP)
                {
                    if (!pTracker->mCurrentFrame.mvbObjKeysOutlier[i][j])
                    {
                        if (pMOP->Observations()>0)
                        {
                            vMap[j] = true;
                        }
                        else{
                            vVO[j] = true;
                        }
                    }
                }
            }
            mvbObjVO[i] = vVO;
            mvbObjMap[i] = vMap;
        }
    }
}

} //namespace ORB_SLAM
