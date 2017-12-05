#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "feature_tracker.h"
#include "estimator.h"
#include "parameters.h"
#include "vs_viz3d.h"
#include "utility/visualization.h"
#include "utility/tic_toc.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

struct RawImu
{
    RawImu(double _ts=0,double _ax=0,double _ay=0,double _az=0,double _wx=0,double _wy=0, double _wz=0)
    :ts(_ts),ax(_ax),ay(_ay),az(_az),wx(_wx),wy(_wy),wz(_wz){}
    double ts;
    double ax,ay,az;
    double wx,wy,wz;
};

struct RawFeatures
{
    RawFeatures():ts(0){}
    RawFeatures(double _ts, const std::map<int, std::vector<std::pair<int, Vector3d>>>& _features)
    :ts(_ts), features(_features){}
    double ts;
    std::map<int, std::vector<std::pair<int, Vector3d>>> features;
};

FeatureTracker trackerData[NUM_OF_CAM];
Estimator estimator;
std::queue<std::pair<std::vector<RawImu>, RawFeatures>> g_measurements;
std::mutex g_mtx_measurements;

void send_imu(const RawImu &imu)
{
    static double current_time = -1;
    double t = imu.ts;
    if (current_time < 0)
        current_time = t;
    double dt = t - current_time;
    current_time = t;

    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};

    double dx = imu.ax - ba[0];
    double dy = imu.ay - ba[1];
    double dz = imu.az - ba[2];

    double rx = imu.wx - bg[0];
    double ry = imu.wy - bg[1];
    double rz = imu.wz - bg[2];
    //ROS_DEBUG("IMU %f, dt: %f, acc: %f %f %f, gyr: %f %f %f", t, dt, dx, dy, dz, rx, ry, rz);

    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
}

// thread: visual-inertial odometry
void backend_loop()
{
    printf("Start back-end...\n");
    std::vector<cv::Affine3f> traj;
    Viz3dThread viz;
    while (true)
    {
        g_mtx_measurements.lock();
        if(g_measurements.empty())
        {
            g_mtx_measurements.unlock();
            continue;
        }
        auto measurement = g_measurements.front();
        g_measurements.pop();
        g_mtx_measurements.unlock();

    #if 0
        printf("imu: %d", measurement.first.size());
        for(auto i: measurement.first)
        {
            printf("(%f, %.3f %.3f %.3f, %.3f %.3f %.3f) ", i.ts, i.ax, i.ay, i.az, i.wx, i.wy, i.wz);
        }
        printf("\n");

        printf("vis: %f, %d\n", measurement.second.ts, measurement.second.features.size());

        getchar();
    #endif

        TicToc t_s;

        for (auto &imu_msg : measurement.first)
            send_imu(imu_msg);

        auto features = measurement.second;
        // printf("processing frame %f %d %d\n", features.ts, features.features.size(), measurement.first.size());
        estimator.processImage(features.features, features.ts);

        double whole_t = t_s.toc();
        printStatistics(estimator, whole_t);
        std_msgs::Header header;
        header.stamp = ros::Time(features.ts);
        header.frame_id = "world";
        Eigen::Vector3d relocalize_t{Eigen::Vector3d(0, 0, 0)};
        Eigen::Matrix3d relocalize_r{Eigen::Matrix3d::Identity()};
        pubOdometry(estimator, header, relocalize_t, relocalize_r);
        pubKeyPoses(estimator, header, relocalize_t, relocalize_r);
        pubCameraPose(estimator, header, relocalize_t, relocalize_r);
        pubPointCloud(estimator, header, relocalize_t, relocalize_r);
        pubTF(estimator, header, relocalize_t, relocalize_r);
    }
}

RawFeatures feature_track(const cv::Mat& img, ros::Time ts)
{
    static double first_image_time = 0;
    static int pub_count = 1;
    static bool first_image_flag = true;
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = ts.toSec();
    }

    // frequency control
    if (round(1.0 * pub_count / (ts.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (ts.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = ts.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        cv::Mat timg(img.rowRange(IMAGE_ROW * i, IMAGE_ROW * (i + 1)));
        if (EQUALIZE)
        {
            static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(timg, timg);
        }
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(timg);
        else
            trackerData[i].cur_img = timg;
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    vector<uchar> r_status;

    if (PUB_THIS_FRAME)
    {
        std::map<int, std::vector<std::pair<int, Vector3d>>> features;
        pub_count++;
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            if (i != 1 || !STEREO_TRACK)
            {
                auto un_pts = trackerData[i].undistortedPoints();
                auto &cur_pts = trackerData[i].cur_pts;
                auto &ids = trackerData[i].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    features[ids[j]].emplace_back(i, Vector3d(un_pts[j].x, un_pts[j].y, 1));
                }
            }
            else if (STEREO_TRACK)
            {
                auto r_un_pts = trackerData[1].undistortedPoints();
                auto &ids = trackerData[0].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (r_status[j])
                    {
                        features[ids[j]].emplace_back(i, Vector3d(r_un_pts[j].x, r_un_pts[j].y, 1));
                    }
                }
            }
        }
        return RawFeatures(ts.toSec(), features);
    }
    return RawFeatures();
}

int frontend_loop()
{
    // const char* datadir = "/home/symao/data/mynteye/20171107vins_outside/1";
    const char* datadir = "/home/symao/data/euroc/MH_04_difficult";
    std::string video_file = std::string(datadir)+"/img.avi";
    std::string video_ts_file = std::string(datadir)+"/imgts.txt";
    std::string imu_file = std::string(datadir)+"/imu.txt";

    cv::VideoCapture cap(video_file);
    if(!cap.isOpened())
    {
        printf("[ERROR] stereo_video_play: cannot open video %s\n", video_file.c_str());
        return -1;
    }
    std::ifstream fin_imgts(video_ts_file);
    if(!fin_imgts.is_open())
    {
        printf("[ERROR] stereo_video_play: cannot open file %s\n", video_ts_file.c_str());
        return -1;
    }
    std::ifstream fin_imu(imu_file);
    if(!fin_imu.is_open())
    {
        printf("[ERROR] stereo_video_play: cannot open file %s\n", imu_file.c_str());
        return -1;
    }

    cv::Mat img,imgl,imgr;
    ros::Time prev_ts(0);
    ros::Time prev_pubts(0);
    ros::Time img_ts(0);
    ros::Time imu_ts(0);
    double gyro[3] = {0};
    double acc[3] = {0};

    auto f_read_imu = [&](void){
        if(fin_imu.eof()) {return false;}
        double temp;
        fin_imu >> temp >> acc[0] >> acc[1] >> acc[2] >> gyro[0] >> gyro[1] >> gyro[2];
        for(int i=0; i<3; i++)
        {
            // acc[i] *= 9.8;
            // gyro[i] /= 57.2956;
        }
        imu_ts = ros::Time(temp);
        return true;
    };

    auto f_read_image = [&](void){
        if(fin_imgts.eof()) {return false;}
        double temp;
        fin_imgts >> temp;
        img_ts = ros::Time(temp);
        if(!cap.read(img)){return false;}
        cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
        return true;
    };

    f_read_image();
    f_read_image();
    while(imu_ts<=img_ts)
        f_read_imu();
    f_read_image();

    std::vector<RawImu> imu_list;
    printf("Start front-end...\n");
    while(ros::ok())
    {
        if(imu_ts<=img_ts)
        {
            imu_list.push_back(RawImu(imu_ts.toSec(), acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]));
            if(!f_read_imu()){break;}
        }
        else
        {
            auto fea = feature_track(img, img_ts);
            if(!fea.features.empty())
            {
                g_mtx_measurements.lock();
                g_measurements.push(std::make_pair(imu_list, fea));
                imu_list.clear();
                g_mtx_measurements.unlock();
                cv::imshow("image", img);
                cv::waitKey(30);
            }
            if(!f_read_image()){break;}
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    std::thread thread_frontend{frontend_loop};
    std::thread thread_backend{backend_loop};

    ros::spin();

    return 0;
}
