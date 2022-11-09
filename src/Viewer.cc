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

#include "Viewer.h"
#include "Parameters.h"

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

#include <mutex>
#include <unistd.h>
#include <pangolin/pangolin.h>
namespace ORB_SLAM2
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];

}

/// viewer的回调函数
void Viewer::Run()
{
    /// 1. 初始化标志位
    mbFinished = false;// 这个变量配合SetFinish函数用于指示该函数是否执行完毕
    mbStopped = false;

    if((EnSLOTMode == 2 && EnOnlineDetectionMode) || (EnSLOTMode == 1 && EnDynaSLAMMode == 1)) // 若是目标跟踪模式，同时是在线目标检测
    {
        while(EbStartViewerWith2DTracking == false) //
        {
            usleep(3000);
        }
    }


    /// 2. 创建一个窗口
    pangolin::CreateWindowAndBind("SLOT1: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    /// 3. 启动深度测试，OpenGL只绘制最前面的一层，
    /// 绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    /// 4. 在OpenGL中使用颜色混合
    glEnable (GL_BLEND);
    /// 5. 选择混合选项
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    /// 6. 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowObjects("menu.Show Objects", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);

    // Define Camera Render Object (for view / scene browsing)
    /// 7. 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
    /// 定义观测方位向量： ModelViewLookAt
    /// 观测点位置：(mViewpointX mViewpointY mViewpointZ)
    /// 观测目标位置：(0, 0, 0)
    /// 观测的方位向量：(0.0,-1.0, 0.0)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    /// 8. 定义显示面板大小，orbslam中有左右两个面板，左边显示一些按钮，右边显示图形
    /// 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
    /// 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
    /// 最后一个参数（-1024.0f/768.0f）为显示长宽比
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    /// 9. 定义openglmatrix, 相机pose
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    /// 10. window名称
    cv::namedWindow("SLOT1: Current Frame");
    /// 11. 两个标志位: bFollow, bLocalizationMode
    bool bFollow = true;
    bool bLocalizationMode = false;
    /// 12. 主循环
    while(1)
    {
        /// 12.1 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /// 12.2 得到最新的相机位姿 Twc
        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        /// 12.3 根据相机的位姿调整视角: menuFollowCamera为按钮的状态，bFollow为真实的状态
        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }
        d_cam.Activate(s_cam);

        /// 12.4 绘制地图和图像
        /// 12.4.1 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        /// 12.4.2 绘制地图中: 当前的camera
        mpMapDrawer->DrawCurrentCamera(Twc);
        /// 12.4.3 绘制地图中: 关键帧
        if(menuShowKeyFrames || menuShowGraph)
        {
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);

        }
        /// 12.4.4 TODO 绘制地图中: 地图点
        if(menuShowPoints)
        {
            // TODO 除了画本身的环境mappoint, 也把object上面的mappoint给画出来
            mpMapDrawer->DrawMapPoints();
        }
        /// 12.4.5 TODO 绘制地图中: object对象
        if(EnSLOTMode >= 2) // 目标跟踪和自动驾驶模式才画目标
        {
            if (menuShowObjects)
            {
                if(EbViewCurrentObject == true)
                    mpMapDrawer->DrawMapObjectsInCurrentFrame();
                else
                    mpMapDrawer->DrawMapObjectsInFrame();
                //mpMapDrawer->DrawMapObjects();
            }
        }

        /// 12.5 结束???TODO
        pangolin::FinishFrame();

        /// 12.6 TODO 绘制图像: 画frame

        cv::Mat im = mpFrameDrawer->DrawFrame();
        cv::imshow("SLOT1: Current Frame",im);

        /// 显示时间， 本身mT是图像的周期， 我改成50ms在debug下可能好一点
        //cv::waitKey(mT);
        cv::waitKey(EdDrawFrameWaiKeyTime);

        /// 12.7 判断是否需要重设画图:
        if(menuReset)
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->Reset();
            menuReset = false;
        }
        /// 12.8 判断是否需要停止???? TODO
        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }
        /// 12.9 判断是否完成???
        if(CheckFinish())
            break;
    }
    /// 13. 完成
    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
