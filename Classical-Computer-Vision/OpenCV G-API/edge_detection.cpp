#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/resource.h> // For getrusage

double get_memory_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // KB -> MB
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./graph_video <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    CV_Assert(cap.isOpened());

    std::cout << "Start Memory: " << get_memory_mb() << " MB\n";

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat vga, gray, blurred, edges, bgr[3], output;

        // Traditional OpenCV pipeline
        cv::resize(frame, vga, cv::Size(frame.cols / 2, frame.rows / 2));
        cv::cvtColor(vga, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);
        cv::Canny(blurred, edges, 32, 128, 3);
        cv::split(vga, bgr);
        bgr[1] |= edges;
        cv::merge(bgr, 3, output);

        // Display
        cv::Mat display;
        cv::resize(output, display, cv::Size(), 0.8, 0.8);
        cv::imshow("Traditional OpenCV Output", display);

        std::cout << "Current Memory: " << get_memory_mb() << " MB\n";

        if (cv::waitKey(30) >= 0) break;
    }

    std::cout << "End Memory: " << get_memory_mb() << " MB\n";
    return 0;
}
