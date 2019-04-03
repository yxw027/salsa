#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "salsa/salsa.h"


TEST (KLT, TrackVideo)
{
//    std::string video_file=SALSA_DIR"/vid/away.avi";
//    std::string video_file=SALSA_DIR"/vid/towards.avi";
    std::string video_file=SALSA_DIR"/vid/walk_across.avi";

    cv::VideoCapture cap(video_file);
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
        cv::waitKey(500/cap.get(cv::CAP_PROP_FPS));
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
