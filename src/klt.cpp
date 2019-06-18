#include "salsa/salsa.h"

using namespace cv;

namespace salsa
{

void Salsa::initImg(const std::string& filename)//, int _radius, cv::Size _size)
{
    get_yaml_eigen("focal_len", filename, cam_.focal_len_);
    get_yaml_eigen("cam_center", filename, cam_.cam_center_);
    get_yaml_eigen("image_size", filename, cam_.image_size_);
    get_yaml_node("cam_skew", filename, cam_.s_);
    get_yaml_node("kf_parallax_thresh", filename, kf_parallax_thresh_);
    get_yaml_node("kf_feature_thresh", filename, kf_feature_thresh_);
    get_yaml_node("simulate_klt", filename, sim_KLT_);
    get_yaml_node("show_matches", filename, show_matches_);
    get_yaml_node("get_feature_radius", filename, get_feature_radius_);
    get_yaml_node("track_feature_min_distance", filename, track_feature_min_distance_);
    get_yaml_node("tracker_freq", filename, tracker_freq_);
    get_yaml_node("disable_vision", filename, disable_vision_);

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

    std::string mask_filename;
    get_yaml_node("feature_mask", filename, mask_filename, false);
    if (!mask_filename.empty())
    {
        setFeatureMask(mask_filename);
    }
    else
    {
        mask_.create(cv::Size(cam_.image_size_(0), cam_.image_size_(1)), CV_8UC1);
        mask_ = 255;
    }
    t_next_klt_output_ = NAN;
}

bool Salsa::dropFeature(int idx)
{
    kf_feat_.rm(idx);
    current_feat_.rm(idx);
    prev_features_.erase(prev_features_.begin() + idx);
}

void Salsa::setFeatureMask(const std::string& filename)
{
    cv::threshold(cv::imread(filename, IMREAD_GRAYSCALE), mask_, 1, 255, CV_8UC1);
}

void Salsa::imageCallback(const double& tc, const Mat& img, const Eigen::Matrix2d& R_pix)
{
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

    current_img_.copyTo(prev_img_);
    std::swap(current_feat_.pix, prev_features_);

    if (show_matches_)
        showImage();
    if (std::isnan(t_next_klt_output_))
    {
        t_next_klt_output_ = tc - 0.01;
    }
    if (tc > t_next_klt_output_)
    {
        addMeas(meas::Img(tc, current_feat_, R_pix, new_keyframe));
        t_next_klt_output_ += 1.0/tracker_freq_;
    }
}

void Salsa::trackFeatures()
{
    std::vector<float> err;
    calcOpticalFlowPyrLK(prev_img_, current_img_, prev_features_, current_feat_.pix, match_status_, err);
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

    if (current_feat_.size() < std::round(kf_feature_thresh_ * kf_num_feat_))
    {
        SD(2, "KLT new keyframe, not enough matches: = %d/%f",
           kf_Nmatch_feat_, std::round(kf_feature_thresh_ * kf_num_feat_));
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
    for (int i = 0; i < kf_feat_.size(); i++)
    {
        Eigen::Vector3d zihat = R_cj2ci * current_feat_.zetas[i];
        Eigen::Vector2d lihat = cam_.proj(zihat);
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
    cv::findFundamentalMat(kf_feat_.pix, current_feat_.pix, mask, cv::RANSAC, 0.1, 0.999);
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
        goodFeaturesToTrack(current_img_, new_corners, num_new_features, 0.3,
                            get_feature_radius_, point_mask_, 7);

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
    cvtColor(prev_img_, color_img_, COLOR_GRAY2BGR);
    // draw features and ids
    for (int i = 0; i < current_feat_.size(); i++)
    {
        const Scalar& color(colors_[current_feat_.feat_ids[i] % nf_]);
        circle(color_img_, prev_features_[i], 5, color, -1);
        putText(color_img_, std::to_string(current_feat_.feat_ids[i]), prev_features_[i],
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    }

    cv::imshow("tracked points", color_img_);
    cv::waitKey(1);
}



}
