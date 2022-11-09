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

#include "Optimizer.h"
#include "Parameters.h"
#include "Converter.h"
#include "g2o_Object.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
//#include "LoopClosing.h"
#include "Frame.h"
#include "MapObject.h"
#include "MapObjectPoint.h"
#include "DetectionObject.h"
#include "ObjectKeyFrame.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>



#include<mutex>
static int window = 120;
namespace ORB_SLAM2
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialCorrespondences=0;
    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)//N
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;
                //cout<<"obs: "<<obs<<endl;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();//TODO 其中如果没有定义linearizeOplus()函数,则会直接调用数值求导,运算会比较慢

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;

                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<15)//15
        return 0;


    //cout<<"camera ego pose optimization edge:    "<< nInitialCorrespondences<<endl;
    //optimizer.save("./result_before.g2o");


    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        double error_last_monocular =0;
        double error_last_stereo =0;
        int monocular_edge = 0;
        int stereo_edge = 0;

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }
            /// 计算上次优化的总体误差
            else{
                error_last_monocular = error_last_monocular+e->chi2();
                monocular_edge++;
            }

            const float chi2 = e->chi2();
            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }
            /// 计算下上次优化的总体误差
            else{
                error_last_stereo = error_last_stereo + e->chi2();
                stereo_edge++;
            }

            const float chi2 = e->chi2();
            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }
    }


    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

int Optimizer::CFSE3ObjStateOptimization(Frame *pFrame, const vector<std::size_t> &vnNeedToBeOptimized, const bool &bVerbose)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    std::vector<MapObject *> pMObjects;
    std::vector<DetectionObject *> vcCuboids;
    std::vector<size_t> vnInFrameOrder;

    for(size_t i=0; i<vnNeedToBeOptimized.size(); i++)
    {
        size_t n = vnNeedToBeOptimized[i];
        MapObject* mMapObjectTmp = pFrame->mvMapObjects[n];
        DetectionObject* cCuboidTmp = pFrame->mvDetectionObjects[n];
        if(mMapObjectTmp==NULL) // 如果该目标不存在
            assert(0);
        vcCuboids.push_back(cCuboidTmp);
        pMObjects.push_back(mMapObjectTmp);
        vnInFrameOrder.push_back(n);
    }

    if(pMObjects.size()==0)
        return false;
    for(size_t i=0; i<pMObjects.size();i++)// Set Frame vertex
    {
        MapObject *Object_temp = pMObjects[i];
        Object_temp->mmBAFrameIdAndObjVertexID.clear();
        g2o::ObjectState cube_pose = Object_temp->GetCFInFrameObjState(pFrame->mnId);
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();


        vSE3->setEstimate(cube_pose.pose);
        vSE3->setId(i);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId] = i;
    }

    int nTotalEdges = 0;
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);
    // 单目
    vector<vector<g2o::EdgeSE3ProjectXYZOnlyPose*>> veMonoEdges; // 分目标存储
    vector<vector<size_t>> vnMonoPointVertex;
    veMonoEdges.resize(pMObjects.size());
    vnMonoPointVertex.resize(pMObjects.size());
    // 双目
    vector<vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>> veStereoEdges;
    vector<vector<size_t>> vnStereoPointVertex;
    vnStereoPointVertex.resize(pMObjects.size());
    veStereoEdges.resize(pMObjects.size());
    vector<std::size_t> vnInitialCorrespondences(pMObjects.size(), 0);
    for(size_t i=0; i<pMObjects.size(); i++)
    {
        MapObject* mObjectTmp  = pMObjects[i];
        //DetectionObject* cCuboidTmp = vcCuboids[i];
        size_t nInFrameDetObjOrder = vnInFrameOrder[i];
        // 设置目标点-目标， 目标点-目标-相机 的条件是：(二折需要同时满足)
        // 1. 该目标不是该帧建立
        // 2. 该目标的目标点比较多
        if(mObjectTmp->mnFirstObservationFrameId==pFrame->mnId) // 若该目标是该帧建立的, 则不应该用来优化
        {
            cout<<"目标"<<mObjectTmp->mnTruthID<<" 是当前帧"<<pFrame->mnId<<" 建立, 不应用来优化!"<<endl;
            assert(0);
        }
        if(!mObjectTmp->mmBAFrameIdAndObjVertexID.count(pFrame->mnId))
        {
            cout<<"该object的顶点设置错误！"<<endl;
            assert(0);
        }
        vector<MapObjectPoint*> vpMapPointTmp = pFrame->mvpMapObjectPoints[nInFrameDetObjOrder];
        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> veMonoEdgesTmp; // 点投影临时容器
        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>veStereoEdgesTmp;
        vector<size_t> vnMonoPointVertexTmp;
        vector<size_t> vnStereoPointVertexTmp;
        veMonoEdgesTmp.reserve(vpMapPointTmp.size());
        veStereoEdgesTmp.reserve(vpMapPointTmp.size());
        vnMonoPointVertexTmp.reserve(vpMapPointTmp.size());
        vnStereoPointVertexTmp.reserve(vpMapPointTmp.size());

        // First, set the Translation Constraints inspired by the new detection
        g2o::ObjectState InitPose = mObjectTmp->GetCFInFrameObjState(pFrame->mnId);
        g2o::EdgeTransConstraintFromDetction* e = new g2o::EdgeTransConstraintFromDetction();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mObjectTmp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
        Eigen::Vector3d obs;
        obs << InitPose.translation();
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(deltaMono);

        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix3d::Identity()*50);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
        nTotalEdges++;

        for(size_t j=0; j<vpMapPointTmp.size(); j++)
        {
            MapObjectPoint *pMP = vpMapPointTmp[j];
            if(!pMP)
                continue;
            if(pMP->mnFirstFrame == int(pFrame->mnId))// 如果该3D点是当前帧建立 就跳过，因为这个优化没有意义
                assert(0);
            if(pFrame->mvuObjKeysRight[nInFrameDetObjOrder][j]<0)// Monocular observation
            {
                vnInitialCorrespondences[i]++;
                pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][j] = false;
                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvObjKeysUn[nInFrameDetObjOrder][j];
                obs << kpUn.pt.x, kpUn.pt.y;
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();//TODO 其中如果没有定义linearizeOplus()函数,则会直接调用数值求导,运算会比较慢
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mObjectTmp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetInObjFramePosition(); // Tco * Poj
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                optimizer.addEdge(e);
                nTotalEdges++;
                veMonoEdgesTmp.push_back(e);
                vnMonoPointVertexTmp.push_back(j);
            }
            else  // Stereo observation
            {
                vnInitialCorrespondences[i]++;
                pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][j] = false;
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvObjKeysUn[nInFrameDetObjOrder][j];
                const float &kp_ur = pFrame->mvuObjKeysRight[nInFrameDetObjOrder][j];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mObjectTmp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);
                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetInObjFramePosition();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                optimizer.addEdge(e);
                nTotalEdges++;
                veStereoEdgesTmp.push_back(e);
                vnStereoPointVertexTmp.push_back(j);
            }
        }
        veStereoEdges[i] = (veStereoEdgesTmp);
        veMonoEdges[i] = (veMonoEdgesTmp);
        vnMonoPointVertex[i] = (vnMonoPointVertexTmp);
        vnStereoPointVertex[i] = (vnStereoPointVertexTmp);
    }
    if(nTotalEdges<15)//15
        return false;
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};
    vector<size_t> vnBads(pMObjects.size(), 0);
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.setVerbose(bVerbose);
        optimizer.optimize(its[it]);
        double chi2total = 0;
        vnBads = vector<size_t>(pMObjects.size(), 0);
        for(size_t i=0; i<pMObjects.size(); i++)
        {
            int nInFrameDetObjOrder = vnInFrameOrder[i];
            for(size_t j=0; j<veMonoEdges[i].size(); j++) // 单目投影
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = veMonoEdges[i][j];
                const size_t idx = vnMonoPointVertex[i][j];
                if (pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx]) // 为true, 计算该边的误差
                {
                    e->computeError();// NOTE g2o只会计算active edge的误差
                }
                else
                {
                    chi2total = e->chi2() + chi2total;
                }
                const float chi2 = e->chi2();
                if(1)
                {
                    if (chi2 > chi2Mono[it])
                    {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = true;
                        e->setLevel(1);// 设置为outlier
                        vnBads[i]++;
                    }
                    else {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = false;
                        e->setLevel(0);// 设置为inlier
                    }
                    if (it == 2)
                    {
                        e->setRobustKernel(0);// 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
                    }
                }
            }
            for(size_t j=0; j<veStereoEdges[i].size(); j++) // 双目投影
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = veStereoEdges[i][j];
                const size_t idx = vnStereoPointVertex[i][j];
                if (pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx])
                {
                    e->computeError();
                }
                else
                {
                    chi2total = e->chi2() + chi2total;
                }
                const float chi2 = e->chi2();
                if(1)
                {
                    if (chi2 > chi2Stereo[it])
                    {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = true;
                        e->setLevel(1);
                        vnBads[i]++;
                    }
                    else {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = false;
                        e->setLevel(0);
                    }
                    if (it == 2)
                    {
                        e->setRobustKernel(0);
                    }
                }
            }
        }
    }

    if(1)
    {
        for(size_t i=0; i<pMObjects.size();i++)
        {
            MapObject *Object_temp = pMObjects[i];
            DetectionObject* CuboidTmp = vcCuboids[i];
            //vector<MapObjectPoint* > vpMOPs = CuboidTmp->GetInFrameMapObjectPoints();
            vector<MapObjectPoint* > vpMOPs = pFrame->mvpMapObjectPoints[vnInFrameOrder[i]];
            g2o::VertexSE3Expmap* vObject = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId]));
            if(vObject)
            {
                g2o::ObjectState PoseBeforeOptimize = Object_temp->GetCFInFrameObjState(pFrame->mnId);
                g2o::SE3Quat PoseAfterOptimize = vObject->estimate();
                if(0)
                {
                    // 优化前后误差: 比较在相机系下的结果
                    g2o::ObjectState PoseTruth = CuboidTmp->mTruthPosInCameraFrame;
                    g2o::SE3Quat PoseErrBefore = PoseTruth.pose.inverse() * PoseBeforeOptimize.pose;
                    double E = PoseTruth.pose.inverse().toMinimalVector().norm();
                    auto deltaE1 = PoseErrBefore.toMinimalVector();
                    double Rel_error = deltaE1.norm()/E * 100;
                    cout<<Object_temp->mnTruthID<<"优化前误差："<<Rel_error<<"%, "<<PoseErrBefore.toMinimalVector().transpose()<<endl;
                    g2o::SE3Quat PoseErrAfter = PoseTruth.pose.inverse() * PoseAfterOptimize;
                    auto deltaE2 = PoseErrAfter.toMinimalVector();
                    Rel_error = deltaE2.norm()/E * 100;
                    cout<<Object_temp->mnTruthID<<"优化后误差："<<Rel_error<<"%, "<<PoseErrAfter.toMinimalVector().transpose()<<endl;
                }
                g2o::ObjectState Swo(pFrame->mSETcw.inverse() * vObject->estimate(), PoseBeforeOptimize.scale);
                Object_temp->SetInFrameObjState(Swo, pFrame->mnId);
                Object_temp->SetCFInFrameObjState(g2o::ObjectState(vObject->estimate(), PoseBeforeOptimize.scale), pFrame->mnId);
                Object_temp->SetHaveBeenOptimizedInFrameFlag();
            }
        }
    }
    return true;
}

int Optimizer::CFObjStateOptimization(Frame *pFrame, const vector<std::size_t> &vnNeedToBeOptimized, const bool &bVerbose)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    std::vector<MapObject *> pMObjects;
    std::vector<DetectionObject *> vcCuboids;
    std::vector<size_t> vnInFrameOrder;
    /*
    for(size_t i=0; i<pFrame->mvMapObjects.size(); i++)
    {
        MapObject* mMapObjectTmp = pFrame->mvMapObjects[i];
        DetectionObject* cCuboidTmp = pFrame->mvDetectionObjects[i];
        if(mMapObjectTmp==NULL) // 如果该目标不存在
            continue;
        bool bFlag1 = (cCuboidTmp->mbFewMOPsFlag == false && mMapObjectTmp->mnFirstObservationFrameId!=pFrame->mnId);
        if(bFlag1)
        {
            vcCuboids.push_back(cCuboidTmp);
            pMObjects.push_back(mMapObjectTmp);
            vnInFrameOrder.push_back(i);
        }
    }*/
    for(size_t i=0; i<vnNeedToBeOptimized.size(); i++)
    {
        size_t n = vnNeedToBeOptimized[i];
        MapObject* mMapObjectTmp = pFrame->mvMapObjects[n];
        DetectionObject* cCuboidTmp = pFrame->mvDetectionObjects[n];
        if(mMapObjectTmp==NULL) // 如果该目标不存在
            assert(0);
        vcCuboids.push_back(cCuboidTmp);
        pMObjects.push_back(mMapObjectTmp);
        vnInFrameOrder.push_back(n);
    }

    if(pMObjects.size()==0)
        return false;
    for(size_t i=0; i<pMObjects.size();i++)// Set Frame vertex
    {
        MapObject *Object_temp = pMObjects[i];
        Object_temp->mmBAFrameIdAndObjVertexID.clear();
        g2o::ObjectState cube_pose = Object_temp->GetCFInFrameObjState(pFrame->mnId);
        //g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

        // 是否需要定义新的顶点, 可以固定 pitch yaw 和 高度的
        g2o::VertexSE3Fix *vObj = new g2o::VertexSE3Fix;
        vObj->setEstimate(cube_pose.pose);
        vObj->whether_fixheight = false;
        vObj->whether_fixrollpitch = true;
        vObj->setId(i);
        vObj->setFixed(false);
        optimizer.addVertex(vObj);
        Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId] = i;
    }

    int nTotalEdges = 0;
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);
    // 单目
    vector<vector<g2o::EdgeSE3ProjectXYZOnlyPose*>> veMonoEdges; // 分目标存储
    vector<vector<size_t>> vnMonoPointVertex;
    veMonoEdges.resize(pMObjects.size());
    vnMonoPointVertex.resize(pMObjects.size());
    // 双目
    vector<vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>> veStereoEdges;
    vector<vector<size_t>> vnStereoPointVertex;
    vnStereoPointVertex.resize(pMObjects.size());
    veStereoEdges.resize(pMObjects.size());
    vector<std::size_t> vnInitialCorrespondences(pMObjects.size(), 0);
    for(size_t i=0; i<pMObjects.size(); i++)
    {
        MapObject* mObjectTmp  = pMObjects[i];
        //DetectionObject* cCuboidTmp = vcCuboids[i];
        size_t nInFrameDetObjOrder = vnInFrameOrder[i];
        // 设置目标点-目标， 目标点-目标-相机 的条件是：(二折需要同时满足)
        // 1. 该目标不是该帧建立
        // 2. 该目标的目标点比较多
        if(mObjectTmp->mnFirstObservationFrameId ==pFrame->mnId)
        {
            cout<<"目标"<<mObjectTmp->mnTruthID<<" 是当前帧"<<pFrame->mnId<<" 建立, 不应用来优化!"<<endl;
            assert(0);
        }


        if(!mObjectTmp->mmBAFrameIdAndObjVertexID.count(pFrame->mnId))
        {
            cout<<"该object的顶点设置错误！"<<endl;
            assert(0);
        }
        //vector<MapObjectPoint*> vpMapPointTmp = cCuboidTmp->GetInFrameMapObjectPoints();
        vector<MapObjectPoint*> vpMapPointTmp = pFrame->mvpMapObjectPoints[nInFrameDetObjOrder];
        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> veMonoEdgesTmp; // 点投影临时容器
        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>veStereoEdgesTmp;
        vector<size_t> vnMonoPointVertexTmp;
        vector<size_t> vnStereoPointVertexTmp;
        veMonoEdgesTmp.reserve(vpMapPointTmp.size());
        veStereoEdgesTmp.reserve(vpMapPointTmp.size());
        vnMonoPointVertexTmp.reserve(vpMapPointTmp.size());
        vnStereoPointVertexTmp.reserve(vpMapPointTmp.size());
        for(size_t j=0; j<vpMapPointTmp.size(); j++)
        {
            MapObjectPoint *pMP = vpMapPointTmp[j];
            if(!pMP)
                continue;
            if(pMP->mnFirstFrame == int(pFrame->mnId))// 如果该3D点是当前帧建立 就跳过，因为这个优化没有意义
                assert(0);
            if(pFrame->mvuObjKeysRight[nInFrameDetObjOrder][j]<0)// Monocular observation
            {
                vnInitialCorrespondences[i]++;
                pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][j] = false;
                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvObjKeysUn[nInFrameDetObjOrder][j];
                obs << kpUn.pt.x, kpUn.pt.y;
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();//TODO 其中如果没有定义linearizeOplus()函数,则会直接调用数值求导,运算会比较慢
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mObjectTmp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);
                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetInObjFramePosition(); // Tco * Poj
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                optimizer.addEdge(e);
                nTotalEdges++;
                veMonoEdgesTmp.push_back(e);
                vnMonoPointVertexTmp.push_back(j);
            }
            else  // Stereo observation
            {
                vnInitialCorrespondences[i]++;
                pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][j] = false;
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvObjKeysUn[nInFrameDetObjOrder][j];
                const float &kp_ur = pFrame->mvuObjKeysRight[nInFrameDetObjOrder][j];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mObjectTmp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);
                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetInObjFramePosition();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                optimizer.addEdge(e);
                nTotalEdges++;
                veStereoEdgesTmp.push_back(e);
                vnStereoPointVertexTmp.push_back(j);
            }
        }
        veStereoEdges[i] = (veStereoEdgesTmp);
        veMonoEdges[i] = (veMonoEdgesTmp);
        vnMonoPointVertex[i] = (vnMonoPointVertexTmp);
        vnStereoPointVertex[i] = (vnStereoPointVertexTmp);
    }
    if(nTotalEdges<15)//15
        return false;
    cout<<"优化边数:"<<nTotalEdges<<endl;
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};
    vector<size_t> vnBads(pMObjects.size(), 0);
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.setVerbose(bVerbose);
        optimizer.optimize(its[it]);
        double chi2total = 0;
        vnBads = vector<size_t>(pMObjects.size(), 0);
        for(size_t i=0; i<pMObjects.size(); i++)
        {
            int nInFrameDetObjOrder = vnInFrameOrder[i];
            for(size_t j=0; j<veMonoEdges[i].size(); j++) // 单目投影
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = veMonoEdges[i][j];
                const size_t idx = vnMonoPointVertex[i][j];
                if (pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx]) // 为true, 计算该边的误差
                {
                    e->computeError();// NOTE g2o只会计算active edge的误差
                }
                else
                {
                    chi2total = e->chi2() + chi2total;
                }
                const float chi2 = e->chi2();
                if(1)
                {
                    if (chi2 > chi2Mono[it])
                    {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = true;
                        e->setLevel(1);// 设置为outlier
                        vnBads[i]++;
                    }
                    else {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = false;
                        e->setLevel(0);// 设置为inlier
                    }
                    if (it == 2)
                    {
                        e->setRobustKernel(0);// 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
                    }
                }
            }
            for(size_t j=0; j<veStereoEdges[i].size(); j++) // 双目投影
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = veStereoEdges[i][j];
                const size_t idx = vnStereoPointVertex[i][j];
                if (pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx])
                {
                    e->computeError();
                }
                else
                {
                    chi2total = e->chi2() + chi2total;
                }
                const float chi2 = e->chi2();
                if(1)
                {
                    if (chi2 > chi2Stereo[it])
                    {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = true;
                        e->setLevel(1);
                        vnBads[i]++;
                    }
                    else {
                        pFrame->mvbObjKeysOutlier[nInFrameDetObjOrder][idx] = false;
                        e->setLevel(0);
                    }
                    if (it == 2)
                    {
                        e->setRobustKernel(0);
                    }
                }
            }
        }
    }

    for(auto &m: vnBads)
    {
        cout<<"坏点数: "<<m<<endl;
    }

    if(1)
    {
        for(size_t i=0; i<pMObjects.size();i++)
        {
            MapObject *Object_temp = pMObjects[i];
            //DetectionObject* CuboidTmp = vcCuboids[i];
            //vector<MapObjectPoint* > vpMOPs = CuboidTmp->GetInFrameMapObjectPoints();
            vector<MapObjectPoint* > vpMOPs = pFrame->mvpMapObjectPoints[vnInFrameOrder[i]];
            g2o::VertexSE3Fix* vObject = dynamic_cast<g2o::VertexSE3Fix*>(optimizer.vertex(Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId]));
            if(vObject)
            {
                // 优化前误差: 比较在相机系下的结果
                //g2o::ObjectState PoseTruth = CuboidTmp->mTruthPosInCameraFrame;
                g2o::ObjectState PoseBeforeOptimize = Object_temp->GetCFInFrameObjState(pFrame->mnId); // 注意这是世界系
                //g2o::SE3Quat PoseErrBefore = PoseTruth.pose.inverse() * PoseBeforeOptimize.pose;
                //cout<<Object_temp->mnTruthID<<"优化前误差："<< PoseErrBefore.toMinimalVector()<<endl;
                cout<<Object_temp->mnTruthID<<"优化前pose："<< PoseBeforeOptimize.pose<<endl;
                // 优化后误差
                g2o::SE3Quat PoseAfterOptimize = vObject->estimate();
                //g2o::SE3Quat PoseErrAfter = PoseTruth.pose.inverse() * PoseAfterOptimize;
                //cout<<Object_temp->mnTruthID<<"优化后误差："<<PoseErrAfter.toMinimalVector()<<endl;
                cout<<Object_temp->mnTruthID<<"优化后pose："<<PoseAfterOptimize<<endl;
                g2o::ObjectState Swo(pFrame->mSETcw.inverse() * vObject->estimate(), PoseBeforeOptimize.scale);
                Object_temp->SetInFrameObjState(Swo, pFrame->mnId);
                Object_temp->SetCFInFrameObjState(g2o::ObjectState(vObject->estimate(), PoseBeforeOptimize.scale), pFrame->mnId);
                Object_temp->SetHaveBeenOptimizedInFrameFlag();
            }
        }
    }
    return true;
}

int Optimizer::ObjSta2DAndScaleOptimization(Frame *pFrame, const vector<std::size_t> &vnNeedToBeOptimized, const bool &bVerbose)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    typedef g2o::VertexCuboidFixScale g2o_object_vertex;
    typedef g2o::EdgeSE3OnlyCuboidFixScaleProj g2o_camera_obj_2d_edge;

    std::vector<MapObject *> pMObjects;
    std::vector<DetectionObject *> vcCuboids;
    for(size_t i=0; i<vnNeedToBeOptimized.size(); i++)
    {
        size_t n = vnNeedToBeOptimized[i];
        MapObject* mMapObjectTmp = pFrame->mvMapObjects[n];
        DetectionObject* cCuboidTmp = pFrame->mvDetectionObjects[n];
        if(!(cCuboidTmp->mbGoodDetFlag == true && mMapObjectTmp->mbScaleIsKnownAndPreciseFlag == true))
            continue;
        if(mMapObjectTmp==NULL) // 如果该目标不存在
            assert(0);
        vcCuboids.push_back(cCuboidTmp);
        pMObjects.push_back(mMapObjectTmp);
    }

    if(pMObjects.size()==0)
        return false;
    for(size_t i=0; i<pMObjects.size();i++)// Set Frame vertex
    {
        DetectionObject* dObj = vcCuboids[i];
        MapObject *Object_temp = pMObjects[i];

        Object_temp->mmBAFrameIdAndObjVertexID.clear();
        g2o::ObjectState cube_pose = Object_temp->GetCFInFrameObjState(pFrame->mnId);

        g2o_object_vertex *vObj = new g2o_object_vertex();
        vObj->setEstimate(cube_pose);
        vObj->whether_fixheight = false;
        vObj->whether_fixrollpitch = true;
        vObj->fixedscale = dObj->mScale;
        cout<<"obj scale:"<<vObj->fixedscale;
        vObj->setId(i);
        vObj->setFixed(false);
        optimizer.addVertex(vObj);
        Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId] = i;
    }

    int nTotalEdges = 0;
    vector<g2o_camera_obj_2d_edge *> veBBoxEdges;// 2D 投影, 可以改成双目2D投影
    //vector<vector<size_t>> vnIndexCuboidBBox;
    veBBoxEdges.resize(pMObjects.size());
    //vnIndexCuboidBBox.reserve(pMObjects.size());



    for(size_t i=0; i<pMObjects.size(); i++)
    {
        MapObject *Object_temp = pMObjects[i];
        DetectionObject *obs_cuboid = vcCuboids[i];

        if(!Object_temp->mmBAFrameIdAndObjVertexID.count(pFrame->mnId))
        {
            cout<<"该object的顶点设置错误！"<<endl;
            assert(0);
        }

        Eigen::Vector4d inv_sigma;
        inv_sigma.setOnes();
        inv_sigma = inv_sigma * EdBBoxBAWeight;
        Eigen::Matrix4d camera_object_sigma = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
        obs_cuboid->mbBBoxOutlier = false;
        g2o_camera_obj_2d_edge  *e = new g2o_camera_obj_2d_edge();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId])));
        e->setMeasurement(Eigen::Vector4d(obs_cuboid->mrectBBox.x + obs_cuboid->mrectBBox.width/2, obs_cuboid->mrectBBox.y + obs_cuboid->mrectBBox.height/2,  obs_cuboid->mrectBBox.width, obs_cuboid->mrectBBox.height));
        e->Tcw.fromMinimalVector(Eigen::Matrix<double, 6, 1>::Zero());
        e->Kalib = EdCamProjMatrix;
        e->setInformation(camera_object_sigma * obs_cuboid->mdMeasQuality * obs_cuboid->mdMeasQuality);
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(EdBBoxThHuber);
        if(0)
        {
            cv::Mat image = pFrame->mRawImg;
            cv::Rect bbox_2d = obs_cuboid->mrectBBox;
            cv::rectangle(image, bbox_2d, cv::Scalar(255, 255, 0), 2);
            cv::imshow("2D检测", image);
            cv::waitKey(0);
        }
        nTotalEdges++;
        optimizer.addEdge(e);
        veBBoxEdges[i] = e;
    }
    if(nTotalEdges<1)//15
        return false;


    vector<size_t> vnBads(pMObjects.size(), 0);

    optimizer.initializeOptimization(0);
    optimizer.setVerbose(bVerbose);
    optimizer.optimize(15);


    for(size_t i=0, iend = veBBoxEdges.size(); i <iend; i++)
    {
        g2o_camera_obj_2d_edge  *e = veBBoxEdges[i];
        if(e == NULL)
            continue;
        DetectionObject* cCuboidTmp = vcCuboids[i];
        const float chi2 = e->chi2();
        if(chi2 > 80)
        {
            cCuboidTmp->mbBBoxOutlier = true; // 怎么处理该object
        }
    }


    for(size_t i=0; i<pMObjects.size();i++)
    {
        MapObject *Object_temp = pMObjects[i];
        //DetectionObject* CuboidTmp = vcCuboids[i];
        g2o_object_vertex * vObject = dynamic_cast<g2o_object_vertex*>(optimizer.vertex(Object_temp->mmBAFrameIdAndObjVertexID[pFrame->mnId]));
        if(vObject)
        {
            // 优化前误差: 比较在相机系下的结果
            //g2o::ObjectState PoseTruth = CuboidTmp->mTruthPosInCameraFrame;
            g2o::ObjectState PoseBeforeOptimize = Object_temp->GetCFInFrameObjState(pFrame->mnId); // 注意这是世界系
            //g2o::SE3Quat PoseErrBefore = PoseTruth.pose.inverse() * PoseBeforeOptimize.pose;
            //cout<<Object_temp->mnTruthID<<"优化前误差："<< PoseErrBefore.toMinimalVector()<<endl;
            cout<<Object_temp->mnTruthID<<"优化前pose："<< PoseBeforeOptimize.pose<<endl;
            // 优化后误差
            g2o::ObjectState PoseAfterOptimize = vObject->estimate();
            //g2o::SE3Quat PoseErrAfter = PoseTruth.pose.inverse() * PoseAfterOptimize.pose;
            //cout<<Object_temp->mnTruthID<<"优化后误差："<<PoseErrAfter.toMinimalVector()<<endl;
            cout<<Object_temp->mnTruthID<<"优化后pose："<<PoseAfterOptimize.pose<<endl;
            g2o::ObjectState Swo(pFrame->mSETcw.inverse() * vObject->estimate().pose, PoseBeforeOptimize.scale);
            Object_temp->SetInFrameObjState(Swo, pFrame->mnId);
            Object_temp->SetCFInFrameObjState(g2o::ObjectState(vObject->estimate()), pFrame->mnId);
            Object_temp->SetHaveBeenOptimizedInFrameFlag();
        }
    }

    return true;
}

void Optimizer::SE3ObjectLocalBundleAdjustment(ObjectKeyFrame *pKF, const bool &bVerbose)
{

//    cout<<YELLOW<<"目标"<<pKF->mpMapObjects->mnTruthID<<" 关键帧:"<<pKF->mnId<<" "<<pKF->mnFrameId<<" 优化开始"<<endl;
//    cout<<WHITE<<endl;

    list<ObjectKeyFrame *> lLocalKeyFrames;  // 得到需要优化的共视关键帧
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    const vector<ObjectKeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        ObjectKeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    list<MapObjectPoint *> lLocalMapPoints; // 得到需要优化的共视关键帧的目标点们
    for (list<ObjectKeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapObjectPoint *> vpMPs = (*lit)->GetMapObjectPointMatches();
        for (vector<MapObjectPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapObjectPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;// 防止重复添加
                    }
        }
    }

    list<ObjectKeyFrame*> lFixedCameras; // 目标点的所有观测除了上述关键帧外的关键帧作为固定帧
    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<ObjectKeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<ObjectKeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            ObjectKeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    g2o::SparseOptimizer optimizer; // 优化器
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    unsigned long maxKFid = 0; // 需优化关键帧与固定关键帧顶点
    for(list<ObjectKeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        //g2o::VertexSE3Fix * vSE3 = new g2o::VertexSE3Fix();
        //vSE3->whether_fixheight = true;
        //vSE3->whether_fixrollpitch = true;
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }
    for(list<ObjectKeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size(); // 相关容器
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<ObjectKeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapObjectPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    vector<ObjectKeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapObjectPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815); //优化目标点顶点 与 边设置
    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++) {
        MapObjectPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetInObjFrameEigenPosition());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<ObjectKeyFrame *, size_t> observations = pMP->GetObservations();
        //Set edges
        for (map<ObjectKeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            ObjectKeyFrame *pKFi = mit->first;
            if (!pKFi->isBad()) {
                const cv::KeyPoint &kpUn = pKFi->mvObjKeysUn[mit->second];
                // Monocular observation
                if (pKFi->mvuObjKeysRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuObjKeysRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(bVerbose);
    optimizer.optimize(5);

    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapObjectPoint* pMP = vpMapPointEdgeMono[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        e->setRobustKernel(0);
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapObjectPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;
        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        e->setRobustKernel(0);
    }
    optimizer.initializeOptimization(0);
    optimizer.setVerbose(bVerbose);
    optimizer.optimize(10);

    vector<pair<ObjectKeyFrame*,MapObjectPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapObjectPoint* pMP = vpMapPointEdgeMono[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            ObjectKeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapObjectPoint* pMP = vpMapPointEdgeStereo[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            ObjectKeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            ObjectKeyFrame* pKFi = vToErase[i].first;
            MapObjectPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    MapObject* pMO = pKF->mpMapObjects;
    for(list<ObjectKeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFTmp = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFTmp->mnId));

        g2o::SE3Quat sE3quat = vSE3->estimate();
        pKFTmp->SetPose(sE3quat);

        /*
        cout<<"关键帧"<<pKFTmp->mnFrameId<<" 优化结果打印:"<<endl;
        if(1)// 关键帧优化误差打印
        {

            DetectionObject* pDet = pKFTmp->mpDetectionObject;
            g2o::ObjectState Tco_gt = pDet->mTruthPosInCameraFrame;// 优化前误差: 比较在相机系下的结果
            g2o::ObjectState Tco_est_before = pMO->GetCFObjectKeyFrameObjState(pKFTmp);
            g2o::SE3Quat err1 = Tco_gt.pose.inverse() * Tco_est_before.pose;
            cout<<" 优化前误差："<< err1.toMinimalVector()<<endl;
            g2o::SE3Quat Tco_est_after = sE3quat;// 优化后误差
            g2o::SE3Quat err2 = Tco_gt.pose.inverse() * Tco_est_after;
            cout<<" 优化后误差："<<err2.toMinimalVector()<<endl;
            //cout<<WHITE;
        }*/


        g2o::ObjectState x(sE3quat, pKFTmp->mScale);
        pMO->SetCFObjectKeyFrameObjState(pKFTmp, x);
    }
    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapObjectPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetInObjFramePosition(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::ObjectLocalBundleAdjustment(ObjectKeyFrame *pKF, const bool &bVerbose)
{

    cout<<YELLOW<<"\n\n目标"<<pKF->mpMapObjects->mnTruthID<<" 关键帧:"<<pKF->mnId<<" 优化开始\n\n\n"<<WHITE;

    list<ObjectKeyFrame *> lLocalKeyFrames;  // 得到需要优化的共视关键帧
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    const vector<ObjectKeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
//
//    cout<<"打印weight "<<endl;
//    for(int i=0; i<vNeighKFs.size(); i++)
//    {
//        int weight = pKF->GetWeight(vNeighKFs[i]);
//        cout<<weight<<" ";
//    }
//    cout<<endl<<endl;
    int CurrentId = pKF->mnObjId;
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        ObjectKeyFrame *pKFi = vNeighKFs[i];
        if (pKFi->mObjTrackId!=pKF->mObjTrackId)
            assert(0);
        if (CurrentId - pKFi->mnObjId > 11)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    list<MapObjectPoint *> lLocalMapPoints; // 得到需要优化的共视关键帧的目标点们
    for (list<ObjectKeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapObjectPoint *> vpMPs = (*lit)->GetMapObjectPointMatches();
        for (vector<MapObjectPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapObjectPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;// 防止重复添加
                    }
        }
    }

    list<ObjectKeyFrame*> lFixedCameras; // 目标点的所有观测除了上述关键帧外的关键帧作为固定帧
    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<ObjectKeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<ObjectKeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            ObjectKeyFrame* pKFi = mit->first;
            if (CurrentId - pKFi->mnObjId > window)
                continue;
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    g2o::SparseOptimizer optimizer; // 优化器
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    vector<pair<int,int>> vObjKFEdges;
    vector<vector<int>> vvPointEdges;


    unsigned long maxKFid = 0; // 需优化关键帧与固定关键帧顶点
    for(list<ObjectKeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFi = *lit;
        //g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        g2o::VertexSE3Fix * vSE3 = new g2o::VertexSE3Fix();
        vSE3->whether_fixheight = false;
        vSE3->whether_fixrollpitch = true;
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        vObjKFEdges.push_back(make_pair(pKFi->mnId,0));
    }
    for(list<ObjectKeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size(); // 相关容器
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<ObjectKeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapObjectPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    vector<ObjectKeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapObjectPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815); //优化目标点顶点 与 边设置


    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++) {
        MapObjectPoint *pMP = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetInObjFrameEigenPosition());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<ObjectKeyFrame *, size_t> observations = pMP->GetObservations();
        vector<int> vPointEdges;
        //Set edges
        for (map<ObjectKeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            ObjectKeyFrame *pKFi = mit->first;
            if (CurrentId - pKFi->mnObjId > window)
                continue;
            if (!pKFi->isBad()) {

                for(auto &i:vObjKFEdges){
                    if(i.first==pKFi->mnId) {
                        i.second = i.second + 1;
                        break;
                    }
                }
                vPointEdges.push_back(pKFi->mnFrameId);
                const cv::KeyPoint &kpUn = pKFi->mvObjKeysUn[mit->second];
                // Monocular observation
                if (pKFi->mvuObjKeysRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuObjKeysRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
        vvPointEdges.push_back(vPointEdges);
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(bVerbose);
    optimizer.optimize(5);

    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapObjectPoint* pMP = vpMapPointEdgeMono[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        e->setRobustKernel(0);
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapObjectPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;
        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        e->setRobustKernel(0);
    }
    optimizer.initializeOptimization(0);
    optimizer.setVerbose(bVerbose);
    optimizer.optimize(10);

    vector<pair<ObjectKeyFrame*,MapObjectPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapObjectPoint* pMP = vpMapPointEdgeMono[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            ObjectKeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapObjectPoint* pMP = vpMapPointEdgeStereo[i];
        if(pMP->isBad())
            continue;
        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            ObjectKeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            ObjectKeyFrame* pKFi = vToErase[i].first;
            MapObjectPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    MapObject* pMO = pKF->mpMapObjects;
//    cout<<RED<<"The edges of all the local map points print: "<<WHITE<<endl;
//    for(auto i : vvPointEdges){
//        for(int j : i)
//            cout<<j<<",";
//        cout<<endl;
//    }
//    cout<<endl;

    for(list<ObjectKeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        ObjectKeyFrame* pKFTmp = *lit;
        //g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFTmp->mnId));
        g2o::VertexSE3Fix* vSE3 = static_cast<g2o::VertexSE3Fix*>(optimizer.vertex(pKFTmp->mnId));
        g2o::SE3Quat sE3quat = vSE3->estimate();

        if(0)
        {
            int num = 0;
            for(auto &i:vObjKFEdges)
                if (i.first==pKFTmp->mnId)
                    num = i.second;
            g2o::ObjectState PoseTruth = pKFTmp->mpDetectionObject->mTruthPosInCameraFrame;
            g2o::SE3Quat PoseErrBefore = PoseTruth.pose.inverse() * Converter::toSE3Quat(pKFTmp->mTco);
            g2o::SE3Quat PoseErrAfter = PoseTruth.pose.inverse() * sE3quat;
            double E = PoseTruth.pose.toMinimalVector().norm();
            auto deltaE1 = PoseErrBefore.toMinimalVector();
            double Rel_error = deltaE1.norm()/E * 100;
            cout<<RED<<"目标关键帧"<<pKFTmp->mnFrameId<<"  优化边个数:"<<num
            <<"\n优化前误差："<<Rel_error<<"%, "<< PoseErrBefore.toMinimalVector().transpose()<<endl;
            auto deltaE2 = PoseErrAfter.toMinimalVector();
            Rel_error = deltaE2.norm()/E * 100;
            cout<<"优化后误差: "<<Rel_error<<"%, "<<PoseErrAfter.toMinimalVector().transpose()<<endl;
            cout<<WHITE;
        }

        //cout<<RED<<"目标关键帧"<<pKFTmp->mnFrameId<<" 优化前Pose: "<<pKFTmp->mTco<<endl;
        //cout<<"优化后Pose: "<<sE3quat<<endl;
        //cout<<WHITE;


        pKFTmp->SetPose(sE3quat);
        g2o::ObjectState x(sE3quat, pKFTmp->mScale);
        pMO->SetCFObjectKeyFrameObjState(pKFTmp, x);
        pMO->SetCFInFrameObjState(x,pKFTmp->mnFrameId);

    }
    for(list<MapObjectPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapObjectPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetInObjFramePosition(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    vector<pair<int,int>> vKFEdges;
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad()){
            lLocalKeyFrames.push_back(pKFi);
            vKFEdges.push_back(make_pair(pKFi->mnId,0));
        }

    }
//    cout<<YELLOW<<pKF->mnId<<": 打印相机weight "<<endl;
//    for(int i=0; i<vNeighKFs.size(); i++)
//    {
//        int weight = pKF->GetWeight(vNeighKFs[i]);
//        cout<<weight<<" ";
//    }
//    cout<<WHITE<<endl;

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                for (auto &i:vKFEdges) {
                    if (i.first==pKFi->mnId)
                        i.second = i.second +1;
                }

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    //cout<<"Localmapping 2 : 相机+静态点： 边数: "<<optimizer.edges().size()<<endl;
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers

    //optimizer.setVerbose(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
//        if (EnDataSetNameNum<=1) {
//            for (auto &i: vKFEdges) {
//                if (i.first == pKF->mnId) {
//                    cout << RED << "相机关键帧" << i.first << " 优化边个数: " << i.second << endl;
//                    break;
//                }
//            }
//            g2o::SE3Quat TruePose = Converter::toSE3Quat(OfflineFramePoses[pKF->mnFrameId]);
//            // Umeyama alignment and inverse transform
//            // GPS/IMU与 visual 坐标系不同
//            Eigen::Matrix<double,4,4> alignment = Eigen::Matrix<double,4,4>::Identity();
////            alignment<<-0.00622326, -0.99216644, 0.12476793, -0.13131138,
////             0.00805024, -0.12481601, -0.99214725, 0.28329853,
////             0.99994823, -0.00516998,  0.00876394, 2.63234767,
////             0,0,0,1;
//              alignment<<0,-1,0,0,
//              0,0,1,0,
//              -1,0,0,0,
//              0,0,0,1;
////            alignment<<-0.00622326, 0.00805024, 0.99994823, -2.63234767,
////                    -0.99216644, -0.12481601, -0.00516998, -0.08131337,
////                    0.12476793, -0.99214725,  0.00876394, 0.27438757,
////                    0,0,0,1;
//
//            TruePose = Converter::toSE3Quat(Converter::toCvMat(alignment))*TruePose;
//            //TruePose = TruePose.inverse();
//            g2o::SE3Quat PoseErrBefore = TruePose.inverse() * Converter::toSE3Quat(pKF->GetPose());
//            g2o::SE3Quat PoseErrAfter = TruePose.inverse() * SE3quat;
//            double E = TruePose.toMinimalVector().head(3).norm() + 1e-8;
//            double Rel_error1 = PoseErrBefore.toMinimalVector().head(3).norm() / E * 100;
//            double Rel_error2 = PoseErrAfter.toMinimalVector().head(3).norm() / E * 100;
//            cout << "优化前误差: " << Rel_error1 << "% " << TruePose.toMinimalVector().transpose() << endl;
//            cout << "优化后误差: " << Rel_error2 << "% " << SE3quat.toMinimalVector().transpose() << endl;
//        }
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    //cout<<*pbStopFlag<<endl;
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
