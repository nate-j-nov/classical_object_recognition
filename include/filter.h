/*
 * Nate Novak
 * CS5330 Project 2 // Spring 2022
 * Header file for filter.cpp
 */

#include <string>
#include <iostream>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

/* Function to apply a greyscale filter to the image
param: Mat &src passing source image by reference
param: Mat &dst passing dst by reference
return: 0 to indicate success
*/
int greyscale(Mat &src, Mat &dst);

/* Function to apply Gaussian 5X5 filter
param: Mat &src passing source image by reference
param: Mat &dst passing dst by reference
return: 0 to indicate success
*/
int blur5x5(const Mat &src, Mat &dist);

/* Function to apply the Sobel X 3X3 filter
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int sobelX3x3(const Mat &src, cv::Mat &dst);

/* Function to apply the Sobel Y 3X3 filter
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int sobelY3x3(const Mat &src, Mat &dst);

/* Function to get the absolute value of each color channel of each pixel
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int absValOfPixels(const Mat &src, Mat &dst);

/* Function to calculate the magnitude of a given pixel
param: Mat &sx result from sobel x passed by reference
param Mat &sy result from sobel y passed by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int magnitude_nate(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst);

/* Function to blur and quantize an image
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: const int levels is the number of buckets the various colors should be separated into
return: 0 to indicate success
*/
int blurQuantize(const Mat &src, Mat &dst, const int levels);

/* Function to cartoonize an image
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: int levels indicates the number of buckets the various colors should be separated into
param: int magThreshold indicates the magnitude at which the borders should be made totally black
return: 0 to indicate success
*/
int cartoon(const Mat &src, Mat &dst, int levels, int magThreshold);

/* Function to cartoonize and clean an image
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: int levels indicates the number of buckets the various colors should be separated into
param: int magThreshold indicates the magnitude at which the borders should be made totally black
return: 0 to indicate success
*/
int cartoonWithCleaning(const Mat &src, Mat &dst, int levels, int magThreshold);

/*Compare function for qsort*/
int compare(const void * a, const void * b);

/* Function to change constrast and brightness
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: float contrast is the contrast coefficient
param: int brightness indicates the brightness coefficient
return: 0 to indicate success
*/
int contrastBrightness(const Mat &src, Mat &dst, float contrast, int brightness);

/* Function to turn the image upside down
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int upsideDown(const Mat &src, Mat &dst);

/* Function to an image into a news screen
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: const string ticker is the text to be written at the top of the stream
param: const uint tickerStart indicates where the text should begin being displayed
param: Mat small image is the image to be shown in the graphic
return: 0 to indicate success
*/
int news(const Mat &src, Mat &dst, const string message, const string ticker, const uint tickerStart, Mat smallImage);
