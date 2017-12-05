#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

#ifdef GLB_DEFINITION_CPP
#define GLB_EXT
#else
#define GLB_EXT extern
#endif

//#define DEPTH_PRIOR
//#define GT
#define UNIT_SPHERE_ERROR

const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;

GLB_EXT std::string FISHEYE_MASK;
GLB_EXT int MAX_CNT;
GLB_EXT int MIN_DIST;
GLB_EXT int FREQ;
GLB_EXT double F_THRESHOLD;
GLB_EXT int SHOW_TRACK;
GLB_EXT int STEREO_TRACK;
GLB_EXT int EQUALIZE;
GLB_EXT int FISHEYE;
GLB_EXT bool PUB_THIS_FRAME;

GLB_EXT int IMAGE_ROW,IMAGE_COL;

GLB_EXT double FOCAL_LENGTH;
GLB_EXT double LOOP_INFO_VALUE;

GLB_EXT double INIT_DEPTH;
GLB_EXT double MIN_PARALLAX;
GLB_EXT int ESTIMATE_EXTRINSIC;

GLB_EXT double ACC_N, ACC_W;
GLB_EXT double GYR_N, GYR_W;

GLB_EXT std::vector<Eigen::Matrix3d> RIC;
GLB_EXT std::vector<Eigen::Vector3d> TIC;
GLB_EXT Eigen::Vector3d G;

GLB_EXT double BIAS_ACC_THRESHOLD;
GLB_EXT double BIAS_GYR_THRESHOLD;
GLB_EXT double SOLVER_TIME;
GLB_EXT int NUM_ITERATIONS;
GLB_EXT std::string EX_CALIB_RESULT_PATH;
GLB_EXT std::string VINS_RESULT_PATH;
GLB_EXT std::string VINS_FOLDER_PATH;

GLB_EXT int LOOP_CLOSURE;
GLB_EXT int MIN_LOOP_NUM;
GLB_EXT int MAX_KEYFRAME_NUM;
GLB_EXT std::string PATTERN_FILE;
GLB_EXT std::string VOC_FILE;
GLB_EXT std::vector<std::string> CAM_NAMES;
GLB_EXT std::string IMAGE_TOPIC;
GLB_EXT std::string IMU_TOPIC;

void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
