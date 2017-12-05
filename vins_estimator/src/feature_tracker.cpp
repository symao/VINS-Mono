#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

#if USE_CV_CUDA
cv::Ptr<cv::cuda::CornersDetector> g_corner_detector;
cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> g_lk_tracker;

void initCudaHandler()
{
    g_corner_detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, MAX_CNT, 0.1, MIN_DIST);
    g_lk_tracker = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3);
}

void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}
#endif

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < IMAGE_COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < IMAGE_ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{

}

void FeatureTracker::setMask()
{
#if 1
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(IMAGE_ROW, IMAGE_COL, CV_8UC1, cv::Scalar(255));
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
#else //acceleration
    std::vector<uchar> status(track_cnt.size(), 1);
    int new_cols = IMAGE_COL/MIN_DIST + 1;
    int new_rows = IMAGE_ROW/MIN_DIST + 1;
    std::vector<int> track_cnt_vec(new_rows*new_cols, -1);
    for(size_t i = 0; i < forw_pts.size(); i++)
    {
        const auto& p = forw_pts[i];
        int idx = int(p.x/MIN_DIST) + int(p.y/MIN_DIST)*new_cols;
        if(track_cnt_vec[idx] < 0)
        {
            track_cnt_vec[idx] = i;
        }
        else if(track_cnt[track_cnt_vec[idx]]<track_cnt[i])
        {
            status[track_cnt_vec[idx]] = 0;
            track_cnt_vec[idx] = i;
        }
        else
        {
            status[i] = 0;
        }
    }
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(IMAGE_ROW, IMAGE_COL, CV_8UC1, cv::Scalar(255));
    for (const auto& p : forw_pts)
        cv::circle(mask, p, MIN_DIST, 0, -1);
#endif
}

void FeatureTracker::addPoints()
{
    for (auto &p : new_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

#if USE_CV_CUDA
void FeatureTracker::readImage(const cv::cuda::GpuMat &img)
{
    if (d_forw_img.empty())
    {
        img.copyTo(d_cur_img);
    }
    d_forw_img = img;
#else
void FeatureTracker::readImage(const cv::Mat &img)
{
    if (forw_img.empty())
    {
        cur_img = img;
    }
    forw_img = img;
#endif
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
#if USE_CV_CUDA
        cv::cuda::GpuMat d_status;
        g_lk_tracker->calc(d_cur_img, d_forw_img, d_cur_pts, d_forw_pts, d_status);
        download(d_status, status);
        download(d_forw_pts, forw_pts);
#else
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
#endif
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    if (PUB_THIS_FRAME)
    {
        rejectWithF();

        for (auto &n : track_cnt)
            n++;

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != img.size())
                cout << "wrong size " << endl;
#if USE_CV_CUDA
            cv::cuda::GpuMat d_mask(mask);
            g_corner_detector->detect(d_forw_img, d_forw_pts, d_mask);
            download(d_forw_pts, new_pts);
#else
            cv::goodFeaturesToTrack(forw_img, new_pts, MAX_CNT - forw_pts.size(), 0.1, MIN_DIST, mask);
#endif
        }
        else
            new_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());

        prev_pts = forw_pts;
    }
#if USE_CV_CUDA
    d_forw_img.copyTo(d_cur_img);
    d_cur_pts = cv::cuda::GpuMat(forw_pts);
    cur_pts = forw_pts;
#else
    cur_img = forw_img;
    cur_pts = forw_pts;
#endif
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_prev_pts(prev_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < prev_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + IMAGE_COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + IMAGE_ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + IMAGE_COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + IMAGE_ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_prev_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = prev_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
#if USE_CV_CUDA
#else
    cv::Mat undistortedImg(IMAGE_ROW + 600, IMAGE_COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < IMAGE_COL; i++)
        for (int j = 0; j < IMAGE_ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + IMAGE_COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + IMAGE_ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < IMAGE_ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < IMAGE_COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
#endif
}

vector<cv::Point2f> FeatureTracker::undistortedPoints()
{
    vector<cv::Point2f> un_pts;
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }

    return un_pts;
}
