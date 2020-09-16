// The "Paper Detector and Camera Pose Estimation" program.
// It loads several images sequentially and tries to find paper in
// each image

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

void help() {
    cout <<
         "\nA program using thresholding, eroding, contours, approximates a polygonal curve and\n"
         "to find paper in a list of images\n"
         "Returns sequence of squares detected on the image.\n"
         "the sequence is stored in the specified memory storage\n"
         "Call:\n"
         "./squares\n"
         "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


RNG rng(12345);

// comparison function object
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
    double i = fabs(contourArea(Mat(contour1)));
    double j = fabs(contourArea(Mat(contour2)));
    return (i < j);
}

void detectPaperContours(const Mat &image, vector<vector<Point> > &contours) {
    contours.clear();

    // Convert to hsv-space, then split the channels
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    Mat hsv[3];
    split(hsvImage, hsv);

    Mat threshed;
    // We threshold Saturation channel divided by Brightness channel to highlight paper
    threshold((hsv[1] / hsv[2] * 255), threshed, 50, 255, THRESH_BINARY_INV);

    Mat eroded;
    int morph_type = MORPH_RECT;
    int kernel_size = 5;
    Mat element = getStructuringElement(morph_type,
                                        Size(2 * kernel_size + 1, 2 * kernel_size + 1),
                                        Point(kernel_size, kernel_size));
    // We erode edges of paper to separate contours of paper
    erode(threshed, eroded, element);

    // Find all the external contours on the eroded image
    findContours(eroded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find biggest contour
    sort(contours.begin(), contours.end(), compareContourAreas);
    vector<Point> biggestContour = contours[contours.size() - 1];

    double epsilon = 0.01 * arcLength(biggestContour, true);
    approxPolyDP(Mat(biggestContour), biggestContour, epsilon, true);
    vector<vector<Point> > temp(1, vector<Point>(biggestContour.size()));
    temp[0] = biggestContour;
    contours = temp;

//    imshow("threshed", threshed);
//    imshow("eroded", eroded);
}


// the function draws all the squares in the image
void drawPaperContours(Mat &image, const vector<vector<Point> > &contours) {
    for (int i = 0; i < contours.size(); i++) {
        drawContours(image, contours, i, CV_RGB(255, 0, 0), 5, 8, noArray(), 0, Point());
    }
}

double distanceFromOrigin(Point point) {
    return point.x * point.x + point.y * point.y;
}

bool comparePaperPoints(Point point1, Point point2) {
    return distanceFromOrigin(point1) < distanceFromOrigin(point2);
}

double rad2deg(const double rad) {
    return rad * 180.0 / CV_PI;
}

// Calculates rotation matrix to euler angles
Vec3f rotationMatrixToEulerAngles(Mat &R) {

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return Vec3f(rad2deg(x), rad2deg(y), rad2deg(z));
}

void calculatePoints(vector<Point2d> &image_points, vector<Point3d> &model_points,
                     const vector<vector<Point> > &contours) {
    vector<Point> paperEdges = contours[0];
    if (paperEdges.size() != 4) return;
    sort(paperEdges.begin(), paperEdges.end(), comparePaperPoints);

    // 2D image points
    image_points.push_back(paperEdges[0]);
    image_points.push_back(paperEdges[1]);
    image_points.push_back(paperEdges[2]);
    image_points.push_back(paperEdges[3]);
    // 3D model points
    model_points.emplace_back(0.0f, 0.0f, 0.0f);
    model_points.emplace_back(210.0f, 0.0f, 0.0f);
    model_points.emplace_back(0.0f, 297.0f, 0.0f);
    model_points.emplace_back(210.0f, 297.0f, 0.0f);
}

double distanceToCamera(double knownWidth, double focalLength, int perWidth) {
    //compute and return the distance from the paper to the camera
    return (knownWidth * focalLength) / perWidth;
}

double getFocalLength() {
    vector<vector<Point> > contours;
    Mat image = imread("../data/calibration_photo.JPG");
    detectPaperContours(image, contours);

    vector<Point2d> image_points;
    vector<Point3d> model_points;
    calculatePoints(image_points, model_points, contours);

    // initialize the known distance from the camera to the object, which
    // in this case is 47 centimeters
    double KNOWN_DISTANCE = 470.0;

    double knownWidth = model_points[3].y - model_points[0].y;
    double perWidth = image_points[3].y - image_points[0].y;

    double focalLength = (perWidth * KNOWN_DISTANCE) / knownWidth;

    return focalLength;
}

void poseEstimation(Mat &image, const vector<vector<Point> > &contours) {
    vector<Point2d> image_points;
    vector<Point3d> model_points;
    calculatePoints(image_points, model_points, contours);

    // Camera internals
    double focalLength = getFocalLength();
    Point2d center = cv::Point2d(image.cols / 2, image.rows / 2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focalLength, 0, center.x, 0, focalLength, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

    cout << "Camera Matrix " << endl << camera_matrix << endl;
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    Vec3f eulerAngles = rotationMatrixToEulerAngles(rotation_vector);

    double knownWidth = model_points[3].y - model_points[0].y;
    double perWidth = image_points[3].y - image_points[0].y;
    double distance = distanceToCamera(knownWidth, focalLength, perWidth);

    putText(image,
            format("Distance: %.1f cm, Euler angles: x: %.1f, y: %.1f, z: %.1f degrees", distance/10, eulerAngles[0],
                   eulerAngles[1], eulerAngles[2]),
            cv::Point(image.cols / 15, image.rows / 25),
            cv::FONT_HERSHEY_PLAIN,
            5.0,
            CV_RGB(255, 0, 0), //font color
            7);
}


int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: ./program <file>" << endl;
        return -1;
    }

    static const char *names[] = {argv[1], 0};

    help();
    vector<vector<Point> > contours;

    for (int i = 0; names[i] != 0; i++) {
        Mat image = imread(names[i]);
        if (image.empty()) {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        detectPaperContours(image, contours);
        drawPaperContours(image, contours);
        poseEstimation(image, contours);
        imwrite("../out.jpg", image);
    }

    return 0;
}