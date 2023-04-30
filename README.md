# PointSLOT
This project is the source code of simultaneous localization and object tracking (SLOT) named PointSLOT.

PointSLOT is an online, real-time and general system without any artificial priors. The system makes use of ORB features for static scene representation, ego-localization and object motion estimation. In particular, we explicitly identify moving and stationary objects by computing the object motion probabilities, such that features of static objects are utilized to improve the camera pose estimation instead of treating all object regions as outliers. Based on the camera-centric parameterization for the object pose, the trajectories and dynamic features of multi-keyframe objects are efficiently solved through an object-based bundle adjustment approach. The link of the releted paper is [here](https://ieeexplore.ieee.org/abstract/document/10068732).

![image](https://user-images.githubusercontent.com/73513416/211151506-7ed54900-f0c4-40d5-b5e5-b742d0fdc313.png)

# Install
This project is based on the SLAM framework of [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2). You should also include the libtorch (test in 1.11.0 and cuda11.3) package and TensorRT8.4.0.6 package for real-time YOLO and deepsort network inference. If you want to quickly test this system without these annoying packages, you can comment out the codes related to YOLO and deepsort and select the running mode (shown as SLOT.MODE in yaml file) as 4. This mode reads the object deection and association gt results into the system and thus the SLOT system runs normally without network inferences.
# Run
This project is tested on the public KITTI tracking datasets using the running mode of 3 (online) or 4 (read the offline information).
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.bin Examples/Stereo/0000-0013.yaml your/sequence/path Kitti_Tracking 0.1
```
You can select the running mode as 2 to track any kinds of objects in datasets of your own making. This mode allows you to manually select an object region in the first frame, and then the system tracks the selected object region.
