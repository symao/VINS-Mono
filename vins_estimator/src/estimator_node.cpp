#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "loop-closure/loop_closure.h"
#include "loop-closure/keyframe.h"
#include "loop-closure/keyframe_database.h"
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

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
std::mutex m_buf;

//camera param
std_msgs::Header cur_header;
Eigen::Vector3d relocalize_t{Eigen::Vector3d(0, 0, 0)};
Eigen::Matrix3d relocalize_r{Eigen::Matrix3d::Identity()};

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // the lastest imu is earlier than the oldest feature
        if (imu_buf.back()->header.stamp <= feature_buf.front()->header.stamp)
        {
            // ROS_WARN("wait for imu, only should happen at the beginning");
            return measurements;
        }

        // the oldest imu is later than the oldest feature
        if (imu_buf.front()->header.stamp >= feature_buf.front()->header.stamp)
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp <= img_msg->header.stamp)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

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
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        // this thread will keep blocking here until m_buf is notified and measurements is not empty
        lk.unlock();

        for (auto &measurement : measurements)
        {
            for (auto &imu_msg : measurement.first)
                send_imu(RawImu(
                    imu_msg->header.stamp.toSec(),
                    imu_msg->linear_acceleration.x,
                    imu_msg->linear_acceleration.y,
                    imu_msg->linear_acceleration.z,
                    imu_msg->angular_velocity.x,
                    imu_msg->angular_velocity.y,
                    imu_msg->angular_velocity.z));

            auto img_msg = measurement.second;
            printf("ts:%f cnt:%d %d\n", img_msg->header.stamp.toSec(), img_msg->points.size(), measurement.first.size());

            TicToc t_s;
            //<feature_id, vector<camera_id, (x,y,z)>>
            map<int, vector<pair<int, Vector3d>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                ROS_ASSERT(z == 1);
                image[feature_id].emplace_back(camera_id, Vector3d(x, y, z));
            }
            estimator.processImage(image, img_msg->header.stamp.toSec());

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            cur_header = header;
            if (estimator.relocalize)
            {
                relocalize_t = estimator.relocalize_t;
                relocalize_r = estimator.relocalize_r;
            }
            pubOdometry(estimator, header, relocalize_t, relocalize_r);
            pubKeyPoses(estimator, header, relocalize_t, relocalize_r);
            pubCameraPose(estimator, header, relocalize_t, relocalize_r);
            pubPointCloud(estimator, header, relocalize_t, relocalize_r);
            pubTF(estimator, header, relocalize_t, relocalize_r);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
