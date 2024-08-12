
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;


int main()
{
    // Initialize the video capture device (default camera)
    cv::VideoCapture cap("Cars.mp4");

    // Check if the video capture device is opened successfully
    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open video capture device." << std::endl;
        return 1;
    }

    // Create a background subtractor
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor = cv::createBackgroundSubtractorMOG2();

    // Create a vector to store the detected contours
    std::vector<std::vector<cv::Point>> contours;

    while (true)
    {
        // Capture a new frame
        cv::Mat frame;
        cap >> frame;

        // Apply background subtraction
        cv::Mat foregroundMask;
        backgroundSubtractor->apply(frame, foregroundMask);

        // Apply thresholding to the foreground mask
        cv::threshold(foregroundMask, foregroundMask, 25, 255, cv::THRESH_BINARY);

        // Find contours in the thresholded image
        cv::findContours(foregroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Iterate through the detected contours
        for (const auto &contour : contours)
        {
            // Calculate the area of the contour
            double area = cv::contourArea(contour);

            // Filter out small noise (area < 100)
            if (area > 100)
            {
                // Draw a bounding rectangle around the contour
                cv::Rect boundingRect = cv::boundingRect(contour);
                cv::rectangle(frame, boundingRect, cv::Scalar(0, 255, 0), 2);

                // Calculate the centroid of the contour
                cv::Moments moments = cv::moments(contour);
                cv::Point centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);
                cv::circle(frame, centroid, 2, cv::Scalar(0, 0, 255), -1);

                // Track the movement of the car (simple implementation)
                static cv::Point prevCentroid;
                cv::line(frame, prevCentroid, centroid, cv::Scalar(0, 0, 255), 2);
                prevCentroid = centroid;
            }
        }

        // Display the output
        cv::imshow("Car Detection and Tracking", frame);

        // Exit on key press
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}

/*

This program captures video from the default camera, applies background subtraction, thresholding, and contour detection to identify moving objects (cars). It then draws a bounding rectangle around each detected car and tracks the movement of the car by drawing a line between the current and previous centroids.

Note that this is a basic implementation and may not work accurately in all scenarios. You may need to adjust the parameters of the background subtractor, thresholding, and contour detection to improve the detection and tracking performance.

Also, this implementation assumes a relatively simple scenario where the cars are moving in a straight line. In a real-world scenario, you may need to consider more advanced tracking techniques, such as the Kalman filter or particle filter, to handle more complex motions.

Make sure to have OpenCV installed and linked to your project to run this code.

*/