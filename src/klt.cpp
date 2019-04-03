#include "salsa/salsa.h"

using namespace cv;

namespace salsa
{

void Salsa::loadKLT(const std::string& filename)//, int _radius, cv::Size _size)
{
    got_first_img_ = false;
    nf_ = nf_;

    get_yaml_node("show_matches", filename, show_matches_);
    get_yaml_node("feature_nearby_radius", filename, feature_nearby_radius_);

    next_feature_id_ = 0;
    prev_features_.clear();
    new_features_.clear();
    ids_.clear();

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
}

bool Salsa::dropFeatureKLT(int feature_id)
{
    // get the local index of this feature_id
    int local_id = std::distance(ids_.begin(), std::find(ids_.begin(), ids_.end(), feature_id));
    if (local_id < ids_.size())
    {
        ids_.erase(ids_.begin() + local_id);
        new_features_.erase(new_features_.begin() + local_id);
        return true;
    }
    else
    {
        return false;
    }
}

void Salsa::setFeatureMask(const std::string& filename)
{
    cv::threshold(cv::imread(filename, IMREAD_GRAYSCALE), mask_, 1, 255, CV_8UC1);
}

//void KLT_Tracker::load_image(const Mat& img, double t, std::vector<Point2f> &features, std::vector<int> &ids, OutputArray& output)
void Salsa::imageCallback(const Mat& img)//, double t, std::vector<Point2f> &features, std::vector<int> &ids, OutputArray& output)
{
    if (img.channels() > 1)
        cvtColor(img, gray_img_, COLOR_BGR2GRAY);
    else
        gray_img_ = img;

    if (got_first_img_)
    {
        trackFeatures();
        filterFeaturesRANSAC();
    }
    else
        got_first_img_ = true;

    collectNewfeatures();
    gray_img_.copyTo(prev_img_);
    std::swap(new_features_, prev_features_);
    new_features_.resize(prev_features_.size());\

    if (show_matches_)
        showImage();
}

void Salsa::trackFeatures()
{
    vector<uchar> status;
    vector<float> err;
    if (prev_features_.size() < 3)
    {
        prev_features_.clear();
        std::cerr << "KLT Fatal Failure!  Re-starting tracker" <<std::endl;
        return;
    }

    calcOpticalFlowPyrLK(prev_img_, gray_img_, prev_features_, new_features_, status, err);

    vector<int> good_ids;
    for (int i = prev_features_.size()-1; i >= 0; i--)
    {
        // If we found a match and the match is in the image
        double new_x = new_features_[i].x;
        double new_y = new_features_[i].y;
        if (status[i] == 0
            || new_x <= 1.0 || new_y <= 1.0
            || new_x >= gray_img_.cols-1.0 || new_y >= gray_img_.rows-1.0
            || mask_.at<uint8_t>(cv::Point(round(new_x), round(new_y))) != 255)
        {
            new_features_.erase(new_features_.begin() + i);
            prev_features_.erase(prev_features_.begin()+i);
            ids_.erase(ids_.begin() + i);
            continue;
        }

        // Make sure that it's not too close to other points
        bool good_id = true;
        for (auto it = good_ids.begin(); it != good_ids.end(); it++)
        {
            double dx = new_features_[*it].x - new_x;
            double dy = new_features_[*it].y - new_y;
            if (std::sqrt(dx*dx + dy*dy) < feature_nearby_radius_)
            {
                new_features_.erase(new_features_.begin() + i);
                prev_features_.erase(prev_features_.begin()+i);
                ids_.erase(ids_.begin() + i);
                good_id = false;
                break;
            }
        }
        if (good_id)
            good_ids.push_back(i);
    }
}

void Salsa::filterFeaturesRANSAC()
{
    if (prev_features_.size() < 8 || new_features_.size() < 8)
        return;

    Mat mask;
    cv::findFundamentalMat(prev_features_, new_features_, mask, cv::RANSAC, 0.1, 0.999);
//    cv::findEssentialMat(prev_features_, new_features_, cam_.focal_len_(0),
//                         cv::Point2d(cam_.cam_center_(0), cam_.cam_center_(1)),
//                         RANSAC, 0.999, 1.0, mask);
    for (int i = mask.rows-1; i >= mask.rows; --i)
    {
        if (mask.at<float>(i,0) == 0)
        {
            new_features_.erase(new_features_.begin()+i);
            prev_features_.erase(prev_features_.begin()+i);
            ids_.erase(ids_.begin()+i);
        }
    }
}

void Salsa::collectNewfeatures()
{
    if (new_features_.size() < nf_)
    {
        // create a mask around current points
        mask_.copyTo(point_mask_);
        for (int i = 0; i < new_features_.size(); i++)
        {
            circle(point_mask_, new_features_[i], feature_nearby_radius_, 0, -1, 0);
        }

        // Now find a bunch of points, not in the mask
        int num_new_features = nf_ - new_features_.size();
        vector<Point2f> new_corners;
        goodFeaturesToTrack(gray_img_, new_corners, num_new_features, 0.3,
                            feature_nearby_radius_, point_mask_, 7);

        for (int i = 0; i < new_corners.size(); i++)
        {
            new_features_.push_back(new_corners[i]);
            ids_.push_back(next_feature_id_++);
        }
    }
}

void Salsa::showImage()
{
    cvtColor(prev_img_, color_img_, COLOR_GRAY2BGR);
    // draw features and ids
    for (int i = 0; i < new_features_.size(); i++)
    {
        const Scalar& color(colors_[ids_[i] % nf_]);
        circle(color_img_, prev_features_[i], 5, color, -1);
        putText(color_img_, to_string(ids_[i]), prev_features_[i],
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    }

    cv::imshow("tracked points", color_img_);
    cv::waitKey(1);
}



}
