#ifndef TRACKER_H
#define TRACKER_H


#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "model.h"

using std::vector;
namespace DS {
    class NearNeighborDisMetric;

    class tracker {
    public:
        NearNeighborDisMetric *metric;
        float max_iou_distance;
        int max_age;
        int n_init;

        KalmanFilter *kf;

        int _next_idx;


    public:


        vector<Track> tracks;

        tracker(/*NearNeighborDisMetric* metric,*/
                float max_cosine_distance, int nn_budget,
                float max_iou_distance = 0.7, int max_age = 2, int n_init = 0);// 0.7 200 20, 是什么意思
        /// max_age 相当于保存tracks最大长度， 如果超过这个长度，该track还是没有被匹配上， 则删除该track
        /// n_init 表示该目标的连续检测次数？ 只有达到这个该tracks才会被建立
        /// trackdid什么时候会被更新？
        ///
        void predict();

        void update(const DETECTIONS &detections);


        void update(const DETECTIONSV2 &detectionsv2);

        typedef DYNAMICM (tracker::* GATED_METRIC_FUNC)(
                vector<Track> &tracks,
                const DETECTIONS &dets,
                const vector<int> &track_indices,
                const vector<int> &detection_indices);

    private:
        void _match(const DETECTIONS &detections, TRACHER_MATCHD &res);

        void _initiate_track(const DETECTION_ROW &detection);

        void _initiate_track(const DETECTION_ROW &detection, CLSCONF clsConf);

    public:
        DYNAMICM gated_matric(
                vector<Track> &tracks,
                const DETECTIONS &dets,
                const vector<int> &track_indices,
                const vector<int> &detection_indices);

        DYNAMICM iou_cost(
                vector<Track> &tracks,
                const DETECTIONS &dets,
                const vector<int> &track_indices,
                const vector<int> &detection_indices);

        Eigen::VectorXf iou(DETECTBOX &bbox,
                            DETECTBOXSS &candidates);
    };
}
#endif // TRACKER_H
