#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/resource.h>

// Peak resident memory (MB)
double get_memory_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./motion_traditional <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    CV_Assert(cap.isOpened());

    cv::Mat frame, prev_frame;
    cap.read(prev_frame);

    std::cout << "Start Memory: " << get_memory_mb() << " MB\n";

    cv::namedWindow("Traditional Motion Detection", cv::WINDOW_NORMAL);

    int frame_count = 0;
    while (cap.read(frame)) {
        cv::Mat curr_gray, prev_gray;
        cv::Mat curr_blur, prev_blur;
        cv::Mat diff, motion;

        cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(curr_gray, curr_blur, cv::Size(5,5), 0);
        cv::GaussianBlur(prev_gray, prev_blur, cv::Size(5,5), 0);

        cv::absdiff(curr_blur, prev_blur, diff);
        cv::threshold(diff, motion, 25, 255, cv::THRESH_BINARY);

        // Display motion mask
        cv::Mat display;
        cv::resize(motion, display, cv::Size(), 0.3, 0.3);
        cv::imshow("Traditional Motion Detection", display); // <-- show window

        prev_frame = frame.clone();

        if (cv::waitKey(30) >= 0) break;
    }


    std::cout << "End Memory: " << get_memory_mb() << " MB\n";
    return 0;
}
