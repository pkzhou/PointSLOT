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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Parameters.h"
#include "MapObject.h"
#include "Converter.h"
#include "MapObjectPoint.h"
#include "Tracking.h"
#include "DetectionObject.h"
#include <pangolin/pangolin.h>
#include <mutex>


#include"Map.h"


namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

    mObjectPointSize = fSettings["Viewer.ObjectPointSize"];
    mObjectTrajectorySize = fSettings["Viewer.ObjectTrajectorySize"];

    mveigColors.push_back(Eigen::Vector3f(230, 0, 0) / 255.0);	 // red  0
    mveigColors.push_back(Eigen::Vector3f(60, 180, 75) / 255.0);   // green  1
    mveigColors.push_back(Eigen::Vector3f(0, 0, 255) / 255.0);	 // blue  2
    mveigColors.push_back(Eigen::Vector3f(255, 0, 255) / 255.0);   // Magenta  3
    mveigColors.push_back(Eigen::Vector3f(255, 165, 0) / 255.0);   // orange 4
    mveigColors.push_back(Eigen::Vector3f(128, 0, 128) / 255.0);   //purple 5
    mveigColors.push_back(Eigen::Vector3f(0, 255, 255) / 255.0);   //cyan 6
    mveigColors.push_back(Eigen::Vector3f(210, 245, 60) / 255.0);  //lime  7
    mveigColors.push_back(Eigen::Vector3f(250, 190, 190) / 255.0); //pink  8
    mveigColors.push_back(Eigen::Vector3f(0, 128, 128) / 255.0);   //Teal  9

    //TODO 8行两列????????,为了画cuboids, 注意cuboids的顶点序号
    mOtherEightEdges.resize(8, 2); // draw 8 edges except front face
    mOtherEightEdges << 2, 3, 3, 4, 4, 1, 3, 7, 4, 8, 6, 7, 7, 8, 8, 5;
    mOtherEightEdges.array() -= 1;
    mFrontFourEdges.resize(4, 2);
    mFrontFourEdges << 1, 2, 2, 6, 6, 5, 5, 1;
    mFrontFourEdges.array() -= 1;

}

/// 画地图中: 画地图点
void MapDrawer::DrawMapPoints()
{
    /// 1. 取出所有的地图点: vpMPs
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    /// 2. 取出mvpReferenceMapPoints，也即局部地图点: vpRefMPs
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();


    /// 3. 将vpRefMPs从vector容器类型转化为set容器类型，便于使用set::count快速统计
    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    /// 4. 判断: 如果没有地图点,则直接返回
    if(vpMPs.empty())
        return;

    // for AllMapPoints
    /// 5. 显示所有的地图点（不包括局部地图点），大小为2个像素，黑色
    glPointSize(mPointSize); /// 大小
    glBegin(GL_POINTS); /// 画点
    glColor3f(0.7,0.3,0.2); /// 颜色
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        /// 5.1 遍历所有地图点, 但不包括ReferenceMapPoints（局部地图点）
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        /// 5.2 画地图点
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    // for ReferenceMapPoints
    /// 6. 显示局部地图点，大小为2个像素，红色
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad()) /// 如果是bad就不画
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }
    glEnd();



}

/// 画地图中: 画出object对象
void MapDrawer::DrawMapObjects()
{
    const vector<MapObject *> vpMOs = mpMap->GetAllMapObjects();

    // 应该先把普通相机帧的pose求出来, 再去求所有普通帧的object的pose, 现在这样写其实不太合理

    for (size_t object_id = 0; object_id < vpMOs.size(); object_id++)
    {

        MapObject *pMO = vpMOs[object_id];
        std::map<int, g2o::ObjectState> allFramePosForThisObject = pMO->GetInAllFrameObjStates();
        if(allFramePosForThisObject.size() == 0)
            assert(0);

        // 画出该object历史轨迹
        glLineWidth(mGraphLineWidth * 2);
        /// 设置类型GL_LINE_STRIP, 绘制从第一个顶点到最后一个顶点依次相连的一组线段，
        /// 第n和n＋1个顶点定义了线段n，总共绘制n－1条线段
        Eigen::Vector3f box_color = mveigColors[pMO->mnTruthID % mveigColors.size()];
        if (!pMO->mbPoseInit) box_color = Eigen::Vector3f(0,0,0);
        glBegin(GL_LINE_STRIP);
        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f); // draw all edges  cyan

        for(size_t i=0; i<allFramePosForThisObject.size(); i++)
        {
            g2o::SE3Quat cubeTmp = allFramePosForThisObject[i].pose; // Twc * Tco
            glVertex3f(cubeTmp.translation()(0), cubeTmp.translation()(1),cubeTmp.translation()(2));
        }
        glEnd();


        /// 画当前cuboid
        g2o::ObjectState objPosInWorldFrame;
        pMO->GetInLatestFrameObjState(objPosInWorldFrame);
        Eigen::MatrixXd cube_corners;
        cube_corners = objPosInWorldFrame.compute3D_BoxCorner();
        glLineWidth(mGraphLineWidth * 2);
        glBegin(GL_LINES);/// 指定画线, 每两个点画一条线

        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f);
        for (int line_id = 0; line_id < mOtherEightEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0,mOtherEightEdges(line_id, 0)),
                    cube_corners(1, mOtherEightEdges(line_id, 0)),
                    cube_corners(2, mOtherEightEdges(line_id, 0)));
            glVertex3f(
                    cube_corners(0, mOtherEightEdges(line_id, 1)),
                    cube_corners(1, mOtherEightEdges(line_id, 1)),
                    cube_corners(2, mOtherEightEdges(line_id, 1)));
        }
        for (int line_id = 0; line_id < mFrontFourEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 0)),
                    cube_corners(1, mFrontFourEdges(line_id, 0)),
                    cube_corners(2, mFrontFourEdges(line_id, 0)));

            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 1)),
                    cube_corners(1, mFrontFourEdges(line_id, 1)),
                    cube_corners(2, mFrontFourEdges(line_id, 1)));
        }
        glEnd();


    }
}




void MapDrawer::DrawMapObjectsInFrame()
{
    const vector<MapObject *> vpMOs = mpMap->GetAllMapObjects();
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    // 应该先把普通相机帧的pose求出来, 再去求所有普通帧的object的pose, 现在这样写其实不太合理

    for (size_t object_id = 0; object_id < vpMOs.size(); object_id++)
    {

        MapObject *pMO = vpMOs[object_id];
        std::map<unsigned long int, g2o::ObjectState> allFramePosForThisObject = pMO->GetCFInAllFrameObjStates();
        if(allFramePosForThisObject.size() == 0)
            assert(0);
        //int end_id = (--allFramePosForThisObject.end())->first;
        //int start_id = (allFramePosForThisObject.begin())->first;
        // 画出该object历史轨迹
        //glLineWidth(mGraphLineWidth * 2);
        glPointSize(mObjectTrajectorySize);
        /// 设置类型GL_LINE_STRIP, 绘制从第一个顶点到最后一个顶点依次相连的一组线段，
        /// 第n和n＋1个顶点定义了线段n，总共绘制n－1条线段
        Eigen::Vector3f box_color = mveigColors[pMO->mnTruthID % mveigColors.size()] * 0.8;

        glBegin(GL_POINTS);
        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f); // draw all edges  cyan
        //int final_keyframe_id = 0;
        for(size_t id =0; id<vpKFs.size(); id++)
        {
            if(allFramePosForThisObject.count(vpKFs[id]->mnFrameId)) // 若关键帧(观测到目标),那么就有一个pose
            {
                g2o::SE3Quat cubeTmp = Converter::toSE3Quat(vpKFs[id]->GetPoseInverse()) *  allFramePosForThisObject[vpKFs[id]->mnFrameId].pose; // Twc * Tco
                glVertex3f(cubeTmp.translation()(0), cubeTmp.translation()(1),cubeTmp.translation()(2));
            }
        }
        glEnd();



        /// 画当前cuboid

        g2o::ObjectState objPosInWorldFrame;
        pMO->GetInLatestFrameObjState(objPosInWorldFrame);
        //objPosInWorldFrame.pose = Converter::toSE3Quat(vpKFs[final_keyframe_id]->GetPoseInverse()) * objPosInWorldFrame.pose;
        Eigen::MatrixXd cube_corners;
        cube_corners = objPosInWorldFrame.compute3D_BoxCorner();
        glLineWidth(mGraphLineWidth * 2);
        glBegin(GL_LINES);/// 指定画线, 每两个点画一条线
        Eigen::Vector3f box_color2 = mveigColors[pMO->mnTruthID % mveigColors.size()];
        glColor4f(box_color2(0), box_color2(1), box_color2(2), 1.0f);
        for (int line_id = 0; line_id < mOtherEightEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0,mOtherEightEdges(line_id, 0)),
                    cube_corners(1, mOtherEightEdges(line_id, 0)),
                    cube_corners(2, mOtherEightEdges(line_id, 0)));
            glVertex3f(
                    cube_corners(0, mOtherEightEdges(line_id, 1)),
                    cube_corners(1, mOtherEightEdges(line_id, 1)),
                    cube_corners(2, mOtherEightEdges(line_id, 1)));
        }
        for (int line_id = 0; line_id < mFrontFourEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 0)),
                    cube_corners(1, mFrontFourEdges(line_id, 0)),
                    cube_corners(2, mFrontFourEdges(line_id, 0)));

            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 1)),
                    cube_corners(1, mFrontFourEdges(line_id, 1)),
                    cube_corners(2, mFrontFourEdges(line_id, 1)));
        }
        glEnd();


        /*
        /// 7. 画动态点: 目标点, 画历史object还是画当前object
        glPointSize(mObjectPointSize);
        glBegin(GL_POINTS);

        // 一共有十个颜色, 因此只显示10个object
        box_color = mveigColors[pMO->mnTruthID % mveigColors.size()];
        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f);
        // 遍历该object的所有points
        vector<MapObjectPoint *> owned_mappoints;
        /// 2. 得到该object的所有landmarks
        owned_mappoints = pMO->GetMapObjectPoints();
        if(owned_mappoints.size() == 0)
        {
            cout<<RED<<"目标"<<pMO->mnTruthID<<"居然没有一个点"<<endl;
            cout<<WHITE;
            continue;
        }
        g2o::ObjectState Swo;




        //  得到该object的最近pose
        int latesObsFrameId = pMO->GetInLatestFrameObjState(Swo);
        if(latesObsFrameId == -1)
            continue;

        for (size_t pt_id = 0; pt_id < owned_mappoints.size(); pt_id++)
        {
            MapObjectPoint *pMOP = owned_mappoints[pt_id];
            if(pMOP==NULL)
                assert(0);
            cv::Mat posInWorldFrame;
            Eigen::Vector3d posInObjFrame = pMOP->GetInObjFrameEigenPosition();
            posInWorldFrame = Converter::toCvMat(Swo.pose.map(posInObjFrame));
            //cout<<"目标点 "<<pMOP->mnId<<" 位置: "<<posInObjFrame<<" , ";
            if (posInWorldFrame.rows == 0)
                assert(0);
            glVertex3f(posInWorldFrame.at<float>(0), posInWorldFrame.at<float>(1), posInWorldFrame.at<float>(2));
        }

        glEnd();*/

    }
}

void MapDrawer::DrawMapObjectsInCurrentFrame()
{
    unique_lock<mutex> lock(mMutexObject);
    const vector<MapObject *> vpMOs = mvMapObjects;
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    // 应该先把普通相机帧的pose求出来, 再去求所有普通帧的object的pose, 现在这样写其实不太合理

    for (size_t object_id = 0; object_id < vpMOs.size(); object_id++)
    {

        MapObject *pMO = vpMOs[object_id];
        if(pMO == nullptr) // 第一帧仅分配了容器没有建立目标
            continue;
        std::map<unsigned long int, g2o::ObjectState> allFramePosForThisObject = pMO->GetCFInAllFrameObjStates();
        if(allFramePosForThisObject.size() == 0)
            assert(0);
        int end_id = (--allFramePosForThisObject.end())->first;
        int start_id = (allFramePosForThisObject.begin())->first;
        //画出该object历史轨迹
        glLineWidth(mGraphLineWidth * 2);
        glPointSize(mObjectTrajectorySize);
        /// 设置类型GL_LINE_STRIP, 绘制从第一个顶点到最后一个顶点依次相连的一组线段，
        /// 第n和n＋1个顶点定义了线段n，总共绘制n－1条线段
        Eigen::Vector3f box_color = mveigColors[pMO->mnTruthID % mveigColors.size()] * 0.8;
        glBegin(GL_POINTS);
        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f); // draw all edges  cyan
        //int final_keyframe_id = 0;
        for(size_t id =0; id<vpKFs.size(); id++)
        {
            if(allFramePosForThisObject.count(vpKFs[id]->mnFrameId)) // 若关键帧(观测到目标),那么就有一个pose
            {
                g2o::SE3Quat cubeTmp = Converter::toSE3Quat(vpKFs[id]->GetPoseInverse()) *  allFramePosForThisObject[vpKFs[id]->mnFrameId].pose; // Twc * Tco
                glVertex3f(cubeTmp.translation()(0), cubeTmp.translation()(1),cubeTmp.translation()(2));
            }
        }
        glEnd();
        //为啥不优化的时候，地图绘制是object先在世界系下不动，过很多帧才重新回到gt附近？

        /// 画当前cuboid
        g2o::ObjectState objPosInWorldFrame;
        g2o::ObjectState objPosInCameraFrame;
        int l = pMO->GetCFLatestFrameObjState(objPosInCameraFrame);
        //cout<<"Current Object"<<pMO->mnTruthID<<" in frame "<<l<<" Position estimate "<<objPosInCameraFrame.translation().transpose()<<endl;
        int latestframeid = pMO->GetInLatestFrameObjState(objPosInWorldFrame);
        //cout<<"Object "<<pMO->mnTruthID<<", Last observation frame: "<<latestframeid<<endl;
        if (latestframeid == -1)
            continue;

        Eigen::MatrixXd cube_corners;
        cube_corners = objPosInWorldFrame.compute3D_BoxCorner();
        glLineWidth(mGraphLineWidth * 2);
        glBegin(GL_LINES);/// 指定画线, 每两个点画一条线
        Eigen::Vector3f box_color2 = mveigColors[pMO->mnTruthID % mveigColors.size()];
        glColor4f(box_color2(0), box_color2(1), box_color2(2), 1.0f);
        for (int line_id = 0; line_id < mOtherEightEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0,mOtherEightEdges(line_id, 0)),
                    cube_corners(1, mOtherEightEdges(line_id, 0)),
                    cube_corners(2, mOtherEightEdges(line_id, 0)));
            glVertex3f(
                    cube_corners(0, mOtherEightEdges(line_id, 1)),
                    cube_corners(1, mOtherEightEdges(line_id, 1)),
                    cube_corners(2, mOtherEightEdges(line_id, 1)));
        }
        for (int line_id = 0; line_id < mFrontFourEdges.rows(); line_id++)
        {
            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 0)),
                    cube_corners(1, mFrontFourEdges(line_id, 0)),
                    cube_corners(2, mFrontFourEdges(line_id, 0)));

            glVertex3f(
                    cube_corners(0, mFrontFourEdges(line_id, 1)),
                    cube_corners(1, mFrontFourEdges(line_id, 1)),
                    cube_corners(2, mFrontFourEdges(line_id, 1)));
        }
        glEnd();

        // test，绘制当前gt检测cuboid
        if (1)
        {
            DetectionObject* det = pMO->GetLatestObservation();
            if (det == nullptr)
                continue;
            g2o::ObjectState gtc = det->mTruthPosInCameraFrame;
            //cout<<"Current Object"<<det->mnObjectID<<""<<"Position gt "<<gtc.translation().transpose()<<endl;
            g2o::ObjectState gt(Converter::toSE3Quat(mCameraPose).inverse() * gtc.pose, gtc.scale);

            //gt.pose = Converter::toSE3Quat(vpKFs[final_keyframe_id]->GetPoseInverse()) * gt.pose;
            Eigen::MatrixXd cube_corners;
            cube_corners = gt.compute3D_BoxCorner();
            glLineWidth(mGraphLineWidth * 2);
            glLineStipple(2, 0x5555);
            glEnable(GL_LINE_STIPPLE);
            glBegin(GL_LINES);/// 指定画线, 每两个点画一条线
            Eigen::Vector3f box_color2 = mveigColors[pMO->mnTruthID % mveigColors.size()];
            glColor4f(box_color2(0), box_color2(1), box_color2(2), 1.0f);
            for (int line_id = 0; line_id < mOtherEightEdges.rows(); line_id++)
            {
                glVertex3f(
                        cube_corners(0,mOtherEightEdges(line_id, 0)),
                        cube_corners(1, mOtherEightEdges(line_id, 0)),
                        cube_corners(2, mOtherEightEdges(line_id, 0)));
                glVertex3f(
                        cube_corners(0, mOtherEightEdges(line_id, 1)),
                        cube_corners(1, mOtherEightEdges(line_id, 1)),
                        cube_corners(2, mOtherEightEdges(line_id, 1)));
            }
            for (int line_id = 0; line_id < mFrontFourEdges.rows(); line_id++)
            {
                glVertex3f(
                        cube_corners(0, mFrontFourEdges(line_id, 0)),
                        cube_corners(1, mFrontFourEdges(line_id, 0)),
                        cube_corners(2, mFrontFourEdges(line_id, 0)));

                glVertex3f(
                        cube_corners(0, mFrontFourEdges(line_id, 1)),
                        cube_corners(1, mFrontFourEdges(line_id, 1)),
                        cube_corners(2, mFrontFourEdges(line_id, 1)));
            }
            glEnd();
            glDisable(GL_LINE_STIPPLE);
        }

        /// 7. 画动态点: 目标点, 画历史object还是画当前object
        if(EnSLOTMode==2 || EnSLOTMode == 3 || EnSLOTMode == 4)
        {
            /// 1. 先得到当前地图的所有object
            vector<MapObject *> all_Map_objs;
            if(EbViewCurrentObject)
                all_Map_objs = mvMapObjects;
            else
                all_Map_objs = mpMap->GetAllMapObjects();


            if(all_Map_objs.size() == 0)
                return;

            glPointSize(mObjectPointSize);
            glBegin(GL_POINTS);
            for (size_t object_id = 0; object_id < all_Map_objs.size(); object_id++)
            {
                MapObject *pMO = all_Map_objs[object_id];
                if(pMO == NULL)
                    continue;
                // 一共有十个颜色, 因此只显示10个object
                Eigen::Vector3f box_color = mveigColors[pMO->mnTruthID % mveigColors.size()];
                glColor4f(box_color(0), box_color(1), box_color(2), 1.0f);
                // 遍历该object的所有points
                vector<MapObjectPoint *> owned_mappoints;
                /// 2. 得到该object的所有landmarks
                owned_mappoints = pMO->GetMapObjectPoints();
                //cout<<"目标 "<<pMO->mnTruthID<<" 拥有的3D点个数: "<<owned_mappoints.size()<<endl;
                //cout<<endl;
                if(owned_mappoints.size() == 0)
                {
//                cout<<RED<<"目标"<<pMO->mnTruthID<<"居然没有一个点"<<endl;
//                cout<<WHITE;
                    continue;
                }
                g2o::ObjectState Swo;
                //  得到该object的最近pose
                int latesObsFrameId = pMO->GetInLatestFrameObjState(Swo);
                //if(latesObsFrameId != ) // 如果最近帧不是当前帧, 说明当前帧可能没有看到该object, 则不画出该object

                for (size_t pt_id = 0; pt_id < owned_mappoints.size(); pt_id++)
                {
                    MapObjectPoint *pMOP = owned_mappoints[pt_id];
                    if(pMOP==NULL)
                        assert(0);
                    cv::Mat posInWorldFrame;
                    Eigen::Vector3d posInObjFrame = pMOP->GetInObjFrameEigenPosition();
                    posInWorldFrame = Converter::toCvMat(Swo.pose.map(posInObjFrame));
                    //cout<<"目标点 "<<pMOP->mnId<<" 位置: "<<posInObjFrame<<" , ";
                    if (posInWorldFrame.rows == 0)
                        assert(0);
                    glVertex3f(posInWorldFrame.at<float>(0), posInWorldFrame.at<float>(1), posInWorldFrame.at<float>(2));
                }

            }
            glEnd();
        }

        /*
        /// 7. 画动态点: 目标点, 画历史object还是画当前object
        glPointSize(mObjectPointSize);
        glBegin(GL_POINTS);

        // 一共有十个颜色, 因此只显示10个object
        box_color = mveigColors[pMO->mnTruthID % mveigColors.size()];
        glColor4f(box_color(0), box_color(1), box_color(2), 1.0f);
        // 遍历该object的所有points
        vector<MapObjectPoint *> owned_mappoints;
        /// 2. 得到该object的所有landmarks
        owned_mappoints = pMO->GetMapObjectPoints();
        if(owned_mappoints.size() == 0)
        {
            cout<<"目标"<<pMO->mnTruthID<<"居然没有一个点"<<endl;
            //assert(0);
            continue;
        }
        g2o::ObjectState Swo;
        //  得到该object的最近pose
        int latesObsFrameId = pMO->GetInLatestFrameObjState(Swo);
        if(latesObsFrameId == -1)
            continue;

        for (size_t pt_id = 0; pt_id < owned_mappoints.size(); pt_id++)
        {
            MapObjectPoint *pMOP = owned_mappoints[pt_id];
            if(pMOP==NULL)
                assert(0);
            cv::Mat posInWorldFrame;
            Eigen::Vector3d posInObjFrame = pMOP->GetInObjFrameEigenPosition();
            posInWorldFrame = Converter::toCvMat(Swo.pose.map(posInObjFrame));
            //cout<<"目标点 "<<pMOP->mnId<<" 位置: "<<posInObjFrame<<" , ";
            if (posInWorldFrame.rows == 0)
                assert(0);
            glVertex3f(posInWorldFrame.at<float>(0), posInWorldFrame.at<float>(1), posInWorldFrame.at<float>(2));
        }

        glEnd();*/

    }
}

void MapDrawer::UpdateCurrentMap(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutexObject);
    mvMapObjects = pTracker->mCurrentFrame.mvMapObjects;
}

void MapDrawer::DrawMapObjectsInKeyFrame(){

}




/// 画地图中: 关键帧
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    /// 1. 设置参数: 历史关键帧图标：宽度占总宽度比例为0.05
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    /// 2. 取出所有的关键帧
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    /// 3. 显示所有关键帧图标: 通过显示界面选择是否显示历史关键帧图标
    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            /// 3.1 依次取出关键帧
            KeyFrame* pKF = vpKFs[i];
            /// 3.2 转置, OpenGL中的矩阵为列优先存储
            cv::Mat Twc = pKF->GetPoseInverse().t();

            /// 3.3 https://www.jianshu.com/p/2a21a12e19d4
            /// 对于矩阵的操作都是对于矩阵栈的栈顶来操作的, 在变换之前调用glPushMatrix()的话，
            /// 就会把当前状态压入第二层，不过此时栈顶的矩阵也与第二层的相同。
            /// 当经过一系列的变换后，栈顶矩阵被修改，此时调用glPopMatrix()时，
            /// 栈顶矩阵被弹出，且又会恢复为原来的状态。
            glPushMatrix();

            /// 3.4 （由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
            /// 因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
            /// TODO ????
            glMultMatrixf(Twc.ptr<GLfloat>(0));
            /// 3.5 设置绘制图形时线的宽度
            glLineWidth(mKeyFrameLineWidth);
            /// 3.6 设置当前颜色为蓝色(关键帧图标显示为蓝色)
            glColor3f(0.0f,0.0f,1.0f);

            glColor3f(0.0f,0.0f,0.0f);
            /// 3.7 用线将下面的顶点两两相连
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();/// 结束

            glPopMatrix();
        }
    }

    /// 4. 显示所有关键帧的Essential Graph: 包括公式程度比较高的边以及spannig tree
    /// 通过显示界面选择是否显示关键帧连接关系
    if(bDrawGraph)
    {
        /// 4.1 设置绘制图形时线的宽度
        glLineWidth(mGraphLineWidth);
        /// 4.2 设置共视图连接线为绿色，透明度为0.6f
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glColor3f(0.0f,0.0f,0.0f);
        glBegin(GL_LINES); /// 还是两两连线

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            /// 4.3 共视程度比较高的共视关键帧用线连接
            /// 4.3.1 遍历每一个关键帧，得到它们共视程度比较高(权重>100)的关键帧: vCovKFs
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            /// 4.3.2 得到该关键帧的世界系下相机坐标
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    /// 4.3.3 遍历vCovKFs:
                    /// (1) 若共视关键帧的id < 当前关键帧的id, 则跳过 (只往后面, 画防止一条线画两次)
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    /// (2) 得到共视关键帧的相机位置(在世界系)
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    /// (3) 连接当前关键帧和共视关键帧
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            /// 4.4 连接最小生成树
            /// 4.4.1 得到当前关键帧的父关键帧
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {   /// 4.4.2 如果存在父关键帧, 得到父关键帧的相机位置(在世界系)
                cv::Mat Owp = pParent->GetCameraCenter();
                /// 4.4.3 连接父关键帧和当前关键帧
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            /// 4.5 连接闭环时形成的连接关系
            /// 4.5.1 得到回环关键帧集合: sLoopKFs
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                /// 4.5.2 遍历sLoopKFs, 若回环关键帧的id < 当前关键帧的id, 则跳过；
                /// (只往后面, 画防止一条线画两次)
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                /// 4.5.3 连接回环关键帧和当前关键帧
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}
/// 绘制地图中的: 当前camera 相机模型大小：宽度占总宽度比例为0.08
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    /// 1. 设置参数
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    /// 2.??
    glPushMatrix();
    /// 3. 将4*4的矩阵Twc.m右乘一个当前矩阵
    ///   （由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
    ///    因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif
    /// 4. 设置参数, 并画图
    /// 4.1 设置绘制图形时线的宽度
    glLineWidth(mCameraLineWidth);
    /// 4.2 设置当前颜色为绿色(相机图标显示为绿色)
    glColor3f(0.0f,1.0f,0.0f);

        glColor3f(0.0f,0.0f,0.0f);

    /// 4.3 用线将下面的顶点两两相连
    glBegin(GL_LINES); /// 开始
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();/// 结束

    /// 4.4 ???TODO
    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPoseAndId(const cv::Mat &Tcw, const int &nFrameId)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
    mnCurrentCameraFrameId = nFrameId;
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
