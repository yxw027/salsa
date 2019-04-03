#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "salsa/salsa.h"


TEST (KLT, TrackVideo)
{
    std::string video_file=SALSA_DIR"/vid/output.avi";

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
        salsa.imageCallback(frame);
//        cv::imshow("vid", frame);
        cv::waitKey(1);
    }

    cap.release();
    cv::destroyAllWindows();
}
