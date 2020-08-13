#include "net.h"
#include "platform.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

static void draw_pose(const cv::Mat& image, const std::vector<KeyPoint>& keypoints)
{
    // draw bone
    static const int joint_pairs[16][2] = {
        {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };
    for (int i = 0; i < 16; i++)
    {
        const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
        const KeyPoint& p2 = keypoints[joint_pairs[i][1]];
        if (p1.prob < 0.2f || p2.prob < 0.2f)
            continue;
        cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }
    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];
        fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);
        if (keypoint.prob < 0.2f)
            continue;
        cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }
}

int runpose(cv::Mat& roi, ncnn::Net posenet, int pose_size_width, int pose_size_height, std::vector<KeyPoint>& keypoints,float x1, float y1)
{
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 roi.cols, roi.rows, pose_size_width, pose_size_height);
    //数据预处理
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = posenet.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    keypoints.clear();
    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for (int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float)out.w+x1, max_y * h / (float)out.h+y1);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }
    return 0;
}


int demo(cv::Mat& image, ncnn::Net detectornet, int detector_size_width, int detector_size_height, \
         ncnn::Net posenet, int pose_size_width, int pose_size_height)
{
    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detectornet.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

    for (int i = 0; i < out.h; i++)
    {
        printf("==================================\n");
        float x1, y1, x2, y2, score, label;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;
        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;
        //截取人体ROI
        cv::Mat roi;
        roi = bgr(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
        std::vector<KeyPoint> keypoints;
        runpose(roi, posenet, pose_size_width, pose_size_height,keypoints, x1, y1);
        draw_pose(image, keypoints);
        cv::rectangle (image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255), 2, 8, 0);
        break;
    }
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}


int main()
{
    cv::Mat img;
    img = cv::imread("test.jpg");

    //定义检测器
    ncnn::Net detectornet;  
    detectornet.load_param("ncnnmodel/person_detector.param");
    detectornet.load_model("ncnnmodel/person_detector.bin");
    int detector_size_width  =  320;
    int detector_size_height = 320;

    //定义人体姿态关键点预测器
    ncnn::Net posenet;  
    posenet.load_param("ncnnmodel/Ultralight-Nano-SimplePose.param");
    posenet.load_model("ncnnmodel/Ultralight-Nano-SimplePose.bin");
    int pose_size_width  =  192;
    int pose_size_height =  256;
    
    demo(img, detectornet, pose_size_width, pose_size_height, posenet, pose_size_width,pose_size_height);
    // cv::imshow("demo", img);
    // cv::waitKey(0);
    return 0;
}
