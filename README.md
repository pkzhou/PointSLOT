# PointSLOT
This project is the source code of simultaneous localization and object tracking (SLOT) named PointSLOT.

PointSLOT is an online, real-time and general system without any artificial priors. The system makes use of ORB features for static scene representation, ego-localization and object motion estimation. In particular, we explicitly identify moving and stationary objects by computing the object motion probabilities, such that features of static objects are utilized to improve the camera pose estimation instead of treating all object regions as outliers. Based on the camera-centric parameterization for the object pose, the trajectories and dynamic features of multi-keyframe objects are efficiently solved through an object-based bundle adjustment. approach.
![image](https://user-images.githubusercontent.com/73513416/211150625-5b13fb0a-fcb4-4a5f-9eed-ab0aa2126070.png)

# Install
This project is based on the SLAM framework of [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2). You should also include the libtorch (test in 1.11.0 and cuda11.3) package and TensorRT8.4.0.6 package for real-time YOLO and deepsort network inference. If you want to quickly test this system without these annoying packages, you can comment out the codes related to YOLO and deepsort and select the running mode (shown as SLOT.MODE in yaml file) as 4. This mode read the object deection and association gt results into the system and thus the SLOT system runs normally without network inferences.
# Run

