#include "salsa/salsa.h"

using namespace cv;

namespace salsa
{

void Salsa::initImg(const std::string& filename)//, int _radius, cv::Size _size)
{
    get_yaml_eigen("focal_len", filename, cam_.focal_len_);
    get_yaml_eigen("cam_center", filename, cam_.cam_center_);
    get_yaml_eigen("image_size", filename, cam_.image_size_);
    get_yaml_eigen("distortion", filename, cam_.distortion_);
    get_yaml_node("cam_skew", filename, cam_.s_);
    get_yaml_node("kf_parallax_thresh", filename, kf_parallax_thresh_);
    get_yaml_node("kf_feature_thresh", filename, kf_feature_thresh_);
    get_yaml_node("simulate_klt", filename, sim_KLT_);
    get_yaml_node("show_matches", filename, show_matches_);
    get_yaml_node("get_feature_radius", filename, get_feature_radius_);
    get_yaml_node("track_feature_min_distance", filename, track_feature_min_distance_);
    get_yaml_node("tracker_freq", filename, tracker_freq_);
    get_yaml_node("disable_vision", filename, disable_vision_);
    get_yaml_node("show_skip", filename, show_skip_);
    get_yaml_node("klt_quality", filename, klt_quality_);
    get_yaml_node("klt_block_size", filename, klt_block_size_);
    get_yaml_node("klt_winsize", filename, klt_winsize_.width);
    get_yaml_node("klt_winsize", filename, klt_winsize_.height);
    get_yaml_node("klt_max_levels", filename, klt_max_levels_);
    get_yaml_node("ransac_thresh", filename, ransac_thresh_);
    get_yaml_node("ransac_prob", filename, ransac_prob_);
    bool use_distort_mask;
    get_yaml_node("use_distort_mask", filename, use_distort_mask);
    get_yaml_node("make_video", filename, make_video_);


    got_first_img_ = false;
    next_feature_id_ = 0;
    prev_features_.clear();
    current_feat_.clear();
    kf_feat_.clear();

    colors_.clear();
    for (int i = 0; i < nf_; i++)
    {
        colors_.push_back(Scalar(std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255)));
    }

    if (use_distort_mask)
    {
        createDistortionMask();
    }
    else
    {
        std::string mask_filename;
        get_yaml_node("mask_filename", filename, mask_filename);
        mask_ = cv::imread(mask_filename, IMREAD_GRAYSCALE);
        if (mask_.empty())
        {
            mask_.create(cv::Size(cam_.image_size_(0), cam_.image_size_(1)), CV_8UC1);
            mask_ = 255;
        }
    }
    cvtColor(cv::Scalar::all(255) - mask_, mask_overlay_, COLOR_GRAY2BGR);
    t_next_klt_output_ = NAN;

    if (show_matches_)
      cv::namedWindow("tracked points");

    if (make_video_)
    {
      video_ = new cv::VideoWriter();
      video_->open(log_prefix_ + "/tracked.avi", cv::VideoWriter::fourcc('M','P','E','G'), 10,
                   cv::Size(cam_.image_size_(0), cam_.image_size_(1)));
    }
}

bool Salsa::dropFeature(int idx)
{
    kf_feat_.rm(idx);
    current_feat_.rm(idx);
    prev_features_.erase(prev_features_.begin() + idx);
}

void Salsa::createDistortionMask()
{
    using namespace Eigen;
    Camera<double> calibrated_cam = cam_;
    // Define undistorted image boundary
    const int num_ppe = 10; // number of points per edge
    double width = cam_.image_size_.x();
    double height = cam_.image_size_.y();
    Matrix<double, num_ppe*4, 2, RowMajor> boundary;
    int row = 0;
    for (uint32_t i = 0; i < num_ppe; i++)
        boundary.row(row++) = cam_.proj(calibrated_cam.invProj(Vector2d(i*(width/num_ppe), 0.0), 1.0)); // bottom
    for (uint32_t i = 0; i < num_ppe; i++)
        boundary.row(row++) = cam_.proj(calibrated_cam.invProj(Vector2d(width, i*(height/num_ppe)), 1.0)); // right
    for (uint32_t i = 0; i < num_ppe; i++)
        boundary.row(row++) = cam_.proj(calibrated_cam.invProj(Vector2d(width - i*(width/num_ppe), height), 1.0)); // top
    for (uint32_t i = 0; i < num_ppe; i++)
        boundary.row(row++) = cam_.proj(calibrated_cam.invProj(Vector2d(0, height - i*(height/num_ppe)), 1.0)); // left

    // Convert boundary to mat and create the mask by filling in a polygon defined by the boundary
    cv::Mat boundary_mat(cv::Size(2, num_ppe*4), CV_64F, boundary.data());
    boundary_mat.convertTo(boundary_mat, CV_32SC1);
    mask_ = cv::Mat(cv::Size(cam_.image_size_.x(), cam_.image_size_.y()), CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(mask_, boundary_mat, cv::Scalar(255));
}

void Salsa::setFeatureMask(const std::string& filename)
{
    cv::threshold(cv::imread(filename, IMREAD_GRAYSCALE), mask_, 1, 255, CV_8UC1);
}

void Salsa::imageCallback(const double& tc, const Mat& img, const Eigen::Matrix2d& R_pix)
{
    if (enable_static_start_ && (lt(tc, static_start_end_ + camera_start_delay_)))
    {
        SD(2, "Waiting for Camera delay after static start");
        return;
    }

    if (img.channels() > 1)
        cvtColor(img, current_img_, COLOR_BGR2GRAY);
    else
        img.copyTo(current_img_);

    if (got_first_img_)
    {
        trackFeatures();
        filterFeaturesOutOfBounds();
        filterFeaturesTooClose(track_feature_min_distance_);
        filterFeaturesRANSAC();
        calcCurrentZetas();
    }
    else
    {
        got_first_img_ = true;
    }

    bool new_keyframe = false;
    if (calcNewKeyframeConditionKLT() != NOT_NEW_KEYFRAME)
    {
        new_keyframe = true;
        createNewKeyframe();
    }

    if (std::isnan(t_next_klt_output_))
    {
        t_next_klt_output_ = tc - 0.01;
    }
    if (tc > t_next_klt_output_)
    {
        if (!disable_vision_)
            addMeas(meas::Img(tc, current_feat_, R_pix, new_keyframe));
        t_next_klt_output_ += 1.0/tracker_freq_;
    }

    current_img_.copyTo(prev_img_);
    prev_features_ = current_feat_.pix;

    if (show_matches_)
        showImage();
}

void Salsa::trackFeatures()
{
    std::vector<float> err;
    calcOpticalFlowPyrLK(prev_img_, current_img_, prev_features_, current_feat_.pix, match_status_,
                         err, klt_winsize_, klt_max_levels_);
}

void Salsa::calcCurrentZetas()
{
    for (int i = 0; i < current_feat_.size(); i++)
    {
        Eigen::Vector2d pix(current_feat_.pix[i].x, current_feat_.pix[i].y);
        current_feat_.zetas[i] = cam_.invProj(pix, 1.0);
    }
}

void Salsa::filterFeaturesOutOfBounds()
{
    for (int i = current_feat_.size()-1; i >= 0; i--)
    {
        // Drop Feature if the match is either not in the image or we didn't find a match
        double new_x = current_feat_.pix[i].x;
        double new_y = current_feat_.pix[i].y;
        if (match_status_[i] == 0
                || new_x <= 1.0 || new_y <= 1.0
                || new_x >= current_img_.cols-1.0 || new_y >= current_img_.rows-1.0
                || mask_.at<uint8_t>(cv::Point(round(new_x), round(new_y))) != 255)
        {
            dropFeature(i);
            continue;
        }
    }
}

int Salsa::calcNewKeyframeConditionKLT()
{
    if (current_feat_.size() == 0)
    {
        kf_condition_ = FIRST_KEYFRAME;
        return FIRST_KEYFRAME;
    }

    if (current_feat_.size() < std::round(kf_feature_thresh_ * nf_))
    {
        SD(2, "KLT new keyframe, not enough matches: = %d/%f",
           current_feat_.size(), std::round(kf_feature_thresh_ * nf_));
        kf_condition_ = INSUFFICIENT_MATCHES;
        return INSUFFICIENT_MATCHES;
    }

    if (current_kf_ < 0)
        return NOT_NEW_KEYFRAME;

    kf_parallax_ = 0;
    quat::Quatd q_I2i(lastKfState().x.q());
    quat::Quatd q_I2j(current_state_.x.q());
    quat::Quatd q_b2c(x_b2c_.q());
    quat::Quatd q_cj2ci = q_b2c.inverse() * q_I2j.inverse() * q_I2i * q_b2c;
    Eigen::Matrix3d R_cj2ci = q_cj2ci.R();

    assert(kf_feat_.size() == current_feat_.size());
    Camera<double> calibrated_cam = cam_; // Don't use distortion model here
    for (int i = 0; i < kf_feat_.size(); i++)
    {
        Eigen::Vector3d zihat = R_cj2ci * current_feat_.zetas[i];
        Eigen::Vector2d lihat = calibrated_cam.proj(zihat);
        Eigen::Vector2d li(kf_feat_.pix[i].x, kf_feat_.pix[i].y);
        double err = (lihat - li).norm();
        kf_parallax_ += err;
    }
    kf_parallax_ /= current_feat_.size();

    if (kf_parallax_ > kf_parallax_thresh_)
    {
        SD(2, "KLT new keyframe, too much parallax: = %f", kf_parallax_);
        kf_condition_ = TOO_MUCH_PARALLAX;
        return TOO_MUCH_PARALLAX;
    }

    return NOT_NEW_KEYFRAME;
}

void Salsa::filterFeaturesTooClose(double dist)
{
    for (int i = 0; i < current_feat_.size(); i++)
    {
        for (int j = current_feat_.size()-1; j > i; j--)
        {
            double dx = current_feat_.pix[i].x - current_feat_.pix[j].x;
            double dy = current_feat_.pix[i].y - current_feat_.pix[j].y;
            if (std::sqrt(dx*dx + dy*dy) < dist)
            {
                SD(1, "feature %d is too close to feature %d, dropping %d",
                   current_feat_.feat_ids[i], current_feat_.feat_ids[j], current_feat_.feat_ids[j]);
                dropFeature(j);
            }
        }
    }
}

void Salsa::filterFeaturesRANSAC()
{
    if (kf_feat_.size() < 8 || current_feat_.size() < 8)
        return;

    Mat mask;
    cv::findFundamentalMat(kf_feat_.pix, current_feat_.pix, mask, cv::RANSAC, ransac_thresh_, ransac_prob_);
    //    cv::findEssentialMat(prev_features_, new_features_, cam_.focal_len_(0),
    //                         cv::Point2d(cam_.cam_center_(0), cam_.cam_center_(1)),
    //                         RANSAC, 0.999, 1.0, mask);
    for (int i = mask.rows-1; i >= mask.rows; --i)
    {
        if (mask.at<float>(i,0) == 0)
        {
            dropFeature(i);
            SD(1, "feature %d filtered by RANSAC", current_feat_.feat_ids[i]);
        }
    }
}

void Salsa::collectNewfeatures()
{
    if (current_feat_.size() < nf_)
    {
        // create a mask around current points
        mask_.copyTo(point_mask_);
        for (int i = 0; i < current_feat_.size(); i++)
        {
            circle(point_mask_, current_feat_.pix[i], get_feature_radius_, 0, -1, 0);
        }

        // Now find a bunch of points, not in the mask
        int num_new_features = nf_ - current_feat_.size();
        std::vector<Point2f> new_corners;
        goodFeaturesToTrack(current_img_, new_corners, num_new_features, klt_quality_,
                            get_feature_radius_, point_mask_, klt_block_size_);

        for (int i = 0; i < new_corners.size(); i++)
        {
            Eigen::Vector2d pix(new_corners[i].x, new_corners[i].y);
            current_feat_.zetas.push_back(cam_.invProj(pix, 1.0));
            current_feat_.pix.emplace_back(std::move(new_corners[i]));
            current_feat_.feat_ids.push_back(next_feature_id_++);
            current_feat_.depths.push_back(NAN);
        }
    }
}

void Salsa::showImage()
{
    if (show_skip_count_ == 0 || make_video_)
    {
        cvtColor(prev_img_, color_img_, COLOR_GRAY2BGR);
        color_img_ = color_img_ - 0.2*mask_overlay_;
        // draw features and ids
        for (int i = 0; i < current_feat_.size(); i++)
        {
            const Scalar& color(colors_[current_feat_.feat_ids[i] % nf_]);
            auto ftit = xfeat_.find(current_feat_.feat_ids[i]);
            if (ftit != xfeat_.end() && ftit->second.funcs.size() > 0)
            {
                cv::Point2f center = current_feat_.pix[i];
                double rho = ftit->second.rho;
                double width = 0.2*cam_.focal_len_[0] * rho;
                cv::Point2f offset(width/2.0, width/2.0);
                cv::Rect box(center + offset, center - offset);
                circle(color_img_, current_feat_.pix[i], 3, color, -1);
                rectangle(color_img_, box, color, 1);
            }
            else
            {
                drawX(color_img_, prev_features_[i], 5, Scalar(255, 0, 0));
            }
        }

        cv::imshow("tracked points", color_img_);
        cv::waitKey(1);
        if (video_)
            video_->write(color_img_);
    }
    show_skip_count_ = (show_skip_count_+1)%show_skip_;
}

void Salsa::drawX(Mat &img, Point2f &center, int size, const Scalar& color)
{
    line(img, cv::Point2f(center.x+size/2, center.y+size/2), cv::Point2f(center.x-size/2, center.y-size/2), color);
    line(img, cv::Point2f(center.x-size/2, center.y+size/2), cv::Point2f(center.x+size/2, center.y-size/2), color);
}



}
