#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/resource.h>

// // Get peak memory usage (MB)
double get_memory_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./motion_gapi <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    CV_Assert(cap.isOpened());

    // // G-API graph: two-frame motion detection
    cv::GMat curr, prev;

     // Convert both frames to grayscale
    cv::GMat curr_gray = cv::gapi::BGR2Gray(curr);
    cv::GMat prev_gray = cv::gapi::BGR2Gray(prev);

    // Blur to reduce noise
    cv::GMat curr_blur = cv::gapi::blur(curr_gray, cv::Size(5,5));
    cv::GMat prev_blur = cv::gapi::blur(prev_gray, cv::Size(5,5));

    // Compute absolute difference between current and previous frame
    cv::GMat diff   = cv::gapi::absDiff(curr_blur, prev_blur);
    cv::GMat motion = cv::gapi::threshold(diff, 25, 255, cv::THRESH_BINARY);

    // Threshold to get binary motion mask
    cv::GComputation graph(cv::GIn(curr, prev), cv::GOut(motion));

    // Main loop: process video
    cv::Mat frame, prev_frame, output;
    cap.read(prev_frame);

    std::cout << "Start Memory: " << get_memory_mb() << " MB\n";

    int frame_count = 0;
    while (cap.read(frame)) {
        graph.apply(cv::gin(frame, prev_frame),
                    cv::gout(output));

        // Display motion mask
        cv::Mat display;
        cv::resize(output, display, cv::Size(), 0.3, 0.3);
        cv::imshow("G-API Motion Detection", display);

        // Update previous frame
        prev_frame = frame.clone();

        if (cv::waitKey(30) >= 0) break;
    }

    std::cout << "End Memory: " << get_memory_mb() << " MB\n";
    return 0;
}
