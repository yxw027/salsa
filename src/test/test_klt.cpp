#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "salsa/salsa.h"

using namespace Eigen;
using namespace xform;


TEST (KLT, TrackVideo)
{
//    std::string video_file=SALSA_DIR"/vid/away.avi";
    std::string video_file=SALSA_DIR"/vid/towards.avi";
//    std::string video_file=SALSA_DIR"/vid/walk_across.avi";

    cv::VideoCapture cap(video_file);
    ASSERT_TRUE(cap.isOpened());
    double dt =  1.0 / cap.get(cv::CAP_PROP_FPS);

    salsa::Salsa salsa;
    salsa.init(default_params("/tmp/Salsa/KLT_TrackVideo"));
    salsa.disable_solver_ = true;

    Vector6d z;
    z << 0, 0, -9.80665, 0, 0, 0;
    Matrix6d R = Matrix6d::Identity();
    double t;
    while(1)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        salsa.imageCallback(t, frame, Matrix2d::Identity());
        for (int i = 0; i < 3; i++)
        {
            t += dt/3.0;
            salsa.imuCallback(t, z, R);
        }
        cv::imshow("kf", salsa.kf_img_);
        cv::waitKey(1);

    }

    cap.release();
    cv::destroyAllWindows();
}

TEST (DISABLED_KLT, TrackCamera)
{
    cv::VideoCapture cap(0);
    ASSERT_TRUE(cap.isOpened());

    salsa::Salsa salsa;
    salsa.load(default_params("KLT.TrackVideo"));

    while(1)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        salsa.imageCallback(0, frame, Matrix2d::Identity());
        //        cv::imshow("vid", frame);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}
