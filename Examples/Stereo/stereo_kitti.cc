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

#include<iostream>
#include<algorithm>
#include<iomanip>
#include<chrono>
#include<string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include<System.h>
#include"Parameters.h"
#include "dirent.h"

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps, string image_class1, string image_class2);
int getFileNum(const std::string &path);


// 注意输入第一个参数是词包，第二个是相机参数，第三个是数据集的baseforlder
int main(int argc, char **argv) {
    if (argc < 6) {
        cerr << endl << " 需要输入参数: 词包路径, yaml文件路径, 数据集路径, 数据集名称, 两帧图像间隔时间!!!"  << endl;
        return 1;
    }

    // 读取数据类型

    if(string(argv[4]).compare(std::string("Kitti_Tracking")) == 0)
    {
        ORB_SLAM2::EnDataSetNameNum = 0;
    }
    else if(string(argv[4]).compare(std::string("Virtual_Kitti")) == 0)
    {
        ORB_SLAM2::EnDataSetNameNum = 1;
    }
    else if (string(argv[4]).compare(std::string("Kitti_Raw")) == 0)
    {
        ORB_SLAM2::EnDataSetNameNum = 2;
    }

    ORB_SLAM2::EnImgTotalNum = getFileNum(string(argv[3])+"/image_02"); // 第5个参数是总图像数目
    ORB_SLAM2::EdT = atof(argv[5]);


    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    ORB_SLAM2::EstrDatasetFolder = string(argv[3]);
    string image_path = ORB_SLAM2::EstrDatasetFolder;


    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    string yamlFile = argv[2];
    cout<<yamlFile;
    cv::FileStorage fSettings(yamlFile, cv::FileStorage::READ);
    string image_class1 = "/image_02/";
    string image_class2 = "/image_03/";
    if (int(fSettings["Camera.gray"])==1){
        image_class1 = "/image_00/";
        image_class2 = "/image_01/";
        cout<<YELLOW<<endl<<" Gray Image Input!"<<endl<<WHITE;
    }
    else{
        cout<<YELLOW<<endl<<" RGB Image Input!"<<endl<<WHITE;
    }
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps, image_class1,image_class2);

    const int nImages = vstrImageLeft.size();
    ORB_SLAM2::EnImgTotalNum = nImages;
    cout<<endl<<argv[3]<<endl;

    ORB_SLAM2::System SLAM(argv[1],yamlFile,ORB_SLAM2::System::STEREO, int(fSettings["Viewer.UseViewer"]));

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    int StartFrameId = ORB_SLAM2::EnStartFrameId;
    for(int ni = StartFrameId; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }


        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);
        ORB_SLAM2::EnStartFrameId++;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            //usleep((T-ttrack)*1e6);
            ;

        //cv::waitKey(0);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0.;

    for(int ni=StartFrameId; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");
    system(("rm -rf " + string("/home/zpk/SLOT/ORB_SLAM2/test_tracking")).c_str());
    system(("mkdir " + string("/home/zpk/SLOT/ORB_SLAM2/test_tracking")).c_str());
    SLAM.SaveObjectDetectionKITTI("/home/zpk/SLOT/ORB_SLAM2/test_tracking/");

//    system(("rm -rf " + string("/home/zpk/SLOT/ORB_SLAM2/test_tracking2")).c_str());
//    system(("mkdir " + string("/home/zpk/SLOT/ORB_SLAM2/test_tracking2")).c_str());
//    SLAM.SaveObjectDetectionResultsInCameraFrame("/home/zpk/SLOT/ORB_SLAM2/test_tracking2/");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps, string image_class1, string image_class2)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamp.txt";
    fTimes.open(strPathTimeFile.c_str());
    if(fTimes.is_open())
    {
        while(!fTimes.eof()) // 到达文件尾
        {
            string s;
            getline(fTimes,s);
            if(!s.empty())
            {
                stringstream ss; // 输入输出操作, 用来进行数据类型转换
                ss << s;
                double t;
                ss >> t;
                vTimestamps.push_back(t);
            }
        }
    }
    else{

        // 如果 没有这个文件, 就自己写
        double t = 0;
        for(size_t i = 0; i<ORB_SLAM2::EnImgTotalNum ; i++)
        {
            vTimestamps.push_back(t);
            t += ORB_SLAM2::EdT;
        }
        cout<<"没有时间戳文件,图像总数为: "<<ORB_SLAM2::EnImgTotalNum<<" 图像周期为: "<<ORB_SLAM2::EdT<<" s"<<endl;
    }



    //string strPrefixLeft = strPathToSequence + "/raw/image_02/data/";
    //string strPrefixRight = strPathToSequence + "/raw/image_03/data/";

    string strPrefixLeft = strPathToSequence + image_class1;
    string strPrefixRight = strPathToSequence + image_class2;

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";


        // 注意virtual kitti图像名字为rgb_00000.jpg
        if(ORB_SLAM2::EnDataSetNameNum == 1)
        {
            stringstream ssvirtualkitti;
            ssvirtualkitti<<setfill('0')<<setw(5)<<i;
            vstrImageLeft[i] = strPrefixLeft + "/rgb_" + ssvirtualkitti.str() + ".jpg";
            vstrImageRight[i] = strPrefixRight + "/rgb_" + ssvirtualkitti.str() + ".jpg";
        }

        if (ORB_SLAM2::EnDataSetNameNum == 2)
        {
            stringstream ssrawkitti;
            ssrawkitti<<setfill('0')<<setw(10)<<i;
            vstrImageLeft[i] = strPrefixLeft + "data/" + ssrawkitti.str() + ".png";
            vstrImageRight[i] = strPrefixRight + "data/" + ssrawkitti.str() + ".png";
        }
    }
}
int getFileNum(const std::string &path) {
    int fileNum=0;
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return fileNum;
    while((ptr=readdir(pDir))!=0){
        if(strcmp(ptr->d_name,".")!=0&&strcmp(ptr->d_name,"..")!=0 )
            fileNum++;
    }
    closedir(pDir);
    return fileNum;
}
