#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

#define USE_CV_CUDA     0
#if USE_CV_CUDA
extern cv::Ptr<cv::cuda::CornersDetector> g_corner_detector;
extern cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> g_lk_tracker;
void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec);
void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec);
void initCudaHandler();
#endif //USE_CV_CUDA

class FeatureTracker
{
  public:
    FeatureTracker();

#if USE_CV_CUDA
    void readImage(const cv::cuda::GpuMat &img);
#else
    void readImage(const cv::Mat &img);
#endif

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    vector<cv::Point2f> undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
#if USE_CV_CUDA
    cv::cuda::GpuMat d_cur_img, d_forw_img;
    cv::cuda::GpuMat d_cur_pts, d_forw_pts;
#else
    cv::Mat cur_img, forw_img;
#endif
    vector<cv::Point2f> new_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<int> ids;
    vector<int> track_cnt;
    camodocal::CameraPtr m_camera;

    static int n_id;
};
