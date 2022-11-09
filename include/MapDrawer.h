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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

//#include"Map.h"
//#include"MapPoint.h"
//#include"KeyFrame.h"
#include<pangolin/pangolin.h>
#include<mutex>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

namespace ORB_SLAM2
{
    class Tracking;
class Map;
class MapPoint;
class KeyFrame;
class MapObject;

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const std::string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPoseAndId(const cv::Mat &Tcw, const int &nFrameId);

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

    //TODO define by yuzhen
    void DrawMapObjectsInKeyFrame();
    void DrawMapObjectsInFrame();
    void DrawMapObjectsInCurrentFrame();
    void UpdateCurrentMap(Tracking *pTracker);
    void DrawMapObjects();


private:
    //TODO define by yuzhen
    std::vector<Eigen::Vector3f> mveigColors;

    Eigen::MatrixXd mFrontFourEdges;
    Eigen::MatrixXd mOtherEightEdges; // for object drawing
    std::vector<MapObject *> mvMapObjects;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    float mObjectPointSize;
    float mObjectTrajectorySize;


    cv::Mat mCameraPose;
    int mnCurrentCameraFrameId;

    std::mutex mMutexCamera;
    std::mutex mMutexObject;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
