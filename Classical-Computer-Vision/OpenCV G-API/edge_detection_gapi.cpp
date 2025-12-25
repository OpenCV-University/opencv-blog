#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <iostream>
#include <sys/resource.h> // For getrusage

// Get memory usage in MB
double get_memory_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // KB -> MB
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./gapi_video <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    CV_Assert(cap.isOpened());

    // Declare G-API graph : resize, grayscale, blr, edge detect, merge channels
    cv::GMat in;
    cv::GMat vga     = cv::gapi::resize(in, cv::Size(), 0.5, 0.5);
    cv::GMat gray    = cv::gapi::BGR2Gray(vga);
    cv::GMat blurred = cv::gapi::blur(gray, cv::Size(5,5));
    cv::GMat edges   = cv::gapi::Canny(blurred, 32, 128, 3);

    //Split the original resized frame into B, G, R channels
    cv::GMat b, g, r;
    std::tie(b, g, r) = cv::gapi::split3(vga);
    cv::GMat out = cv::gapi::merge3(b, g | edges, r);

    // Compile the G-API computation graph
    cv::GComputation graph(in, out);

    cv::Mat frame, output;
    std::cout << "Start Memory: " << get_memory_mb() << " MB\n";

    //main loop: read video frames and process
    while (cap.read(frame)) {
        graph.apply(frame, output);

        cv::Mat display;
        cv::resize(output, display, cv::Size(), 0.8, 0.8);
        cv::imshow("G-API Video Output", display);

        std::cout << "Current Memory: " << get_memory_mb() << " MB\n";

        if (cv::waitKey(30) >= 0) break;
    }

    std::cout << "End Memory: " << get_memory_mb() << " MB\n";
    return 0;
}
