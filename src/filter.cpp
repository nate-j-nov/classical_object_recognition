/*
 * Nate Novak
 * CS5330 Project 2 // Spring 2022
 * Collection of filters 
 */
#include "../include/filter.h"

using namespace std; 
using namespace cv; 


/* Function to apply a greyscale filter to the image
param: Mat &src passing source image by reference
param: Mat &dst passing dst by reference
return: 0 to indicate success
*/
int greyscale(Mat &src, Mat &dst) 
{ 
    for(int r = 0; r < src.rows; ++r) {
        for(int c = 0; c < src.cols; ++c) {
            Vec3b color = src.at<Vec3b>(r, c); // Get color here
            uchar green = color[1]; 
            uchar red = color[2]; 
            uchar newColVal = (green + red) / 2; 
            Vec3b newCol = {newColVal, newColVal, newColVal}; 
            dst.at<Vec3b>(r, c) = newCol;
        }
    }

    return 0; 
}

/* Function to apply Gaussian 5X5 filter 
param: Mat &src passing source image by reference
param: Mat &dst passing dst by reference
return: 0 to indicate success
*/
int blur5x5(const Mat &src, Mat &dst) {
    Mat tmp; 
    src.copyTo(tmp); 
    
    // Get the average value for color channel in first row
    int color[3] = {0, 0, 0}; 
    for(int col = 0; col < src.cols; ++col) {
        for(int c = 0; c < 3; ++c) {
            color[c] += src.at<Vec3b>(0, col)[c]; 
        }
    }

    for(int c = 0; c < 3; ++c) {
        color[c] /= src.cols; 
    }
    

    Vec3b firstRowAvg = {0, 0, 0}; 
    firstRowAvg[0] = (uchar) color[0]; 
    firstRowAvg[1] = (uchar) color[1]; 
    firstRowAvg[2] = (uchar) color[2];
    
    // Set first two rows
    for(int r = 0; r < 2; ++r) {
        for(int c = 0; c < src.cols; ++c) {
            tmp.at<Vec3b>(r, c) = firstRowAvg; 
            dst.at<Vec3b>(r, c) = firstRowAvg; 
        }
    } 

    // Set last two rows
    for(int r = src.rows - 2; r < src.rows; ++r) {
        for(int c = 0; c < src.cols; ++c) {
            tmp.at<Vec3b>(r, c) = firstRowAvg; 
            dst.at<Vec3b>(r, c) = firstRowAvg; 
        }
    }

    // set first two cols
    for(int c = 0; c < 2; ++c) {
        for(int r = 0; r < src.rows; ++r){
            tmp.at<Vec3b>(r, c) = firstRowAvg; 
            dst.at<Vec3b>(r, c) = firstRowAvg; 
        }
    }

    // set last two cols
    for(int c = src.cols - 2; c < src.cols; ++c) {
        for(int r = 0; r < src.rows; ++r){
            tmp.at<Vec3b>(r, c) = firstRowAvg; 
            dst.at<Vec3b>(r, c) = firstRowAvg; 
        }
    }
    
    // Conduct horizontal Gaussian filter
    for(int r = 2; r < src.rows - 2; ++r) {
        for(int c = 2; c < src.cols - 2; ++c) {
            Vec3b resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; co++) {
                int hgauss = src.at<Vec3b>(r, c - 2)[co] + src.at<Vec3b>(r, c - 1)[co] * 2 + 
                             src.at<Vec3b>(r, c)[co] * 4 +
                             src.at<Vec3b>(r, c + 1)[co] * 2 + src.at<Vec3b>(r, c + 2)[co]; 

                hgauss /= 10; 

                resultpix[co] = (uchar) hgauss; 
            }
            tmp.at<Vec3b>(r, c) = resultpix;  
        }
    }

    // Conduct vertical Gaussian filter
    for(int r = 2; r < tmp.rows - 2; ++r) {
        for(int c = 2; c < tmp.cols - 2; ++c) {
            Vec3b resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; co++) {
                int vgauss = tmp.at<Vec3b>(r - 2, c)[co] + tmp.at<Vec3b>(r - 1, c)[co] * 2 +
                             tmp.at<Vec3b>(r, c)[co] * 4 + 
                             tmp.at<Vec3b>(r + 1, c)[co] * 2 + tmp.at<Vec3b>(r + 2, c)[co]; 

                vgauss /= 10; 

                resultpix[co] = (uchar) vgauss;
            }
            dst.at<Vec3b>(r, c) = resultpix; 
        }
    }

    return 0; 
}

/* Function to apply the Sobel X 3X3 filter
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/ 
int sobelX3x3(const Mat &src, cv::Mat &dst ) {
    Mat tmp(src.rows, src.cols, CV_16S);   
    src.convertTo(tmp, CV_16S);
    src.convertTo(dst, CV_16S);

    // Conduct vertical 1X3 Sobel X filter
    for(int r = 0; r < src.rows; ++r) {
        for(int c = 0; c < src.cols; ++c) {
            Vec3s resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; ++co) {
                short vsobel = 0; 
                if(r == 0) {
                    continue; 
                } else if(r == src.rows - 1) {
                    continue; 
                } else {
                    vsobel = src.at<Vec3b>(r - 1, c)[co] + src.at<Vec3b>(r, c)[co] * 2 + src.at<Vec3b>(r + 1, c)[co];
                }
                
                vsobel /= 4; 
                resultpix[co] = (short) vsobel; 
            }
            tmp.at<Vec3s>(r, c) = resultpix;  
        }
    }
    
    // Conduct horizontal 1X3 Sobel X filter
    for(int r = 0; r < tmp.rows; ++r) {
        for(int c = 0; c < tmp.cols; ++c) {
            Vec3s resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; co++) {
                short hsobel = 0; 
                if(c == 0) {
                    continue;
                } else if(c == src.cols - 1) {
                    continue; 
                } else {
                    hsobel = tmp.at<Vec3s>(r, c - 1)[co] * -1 + tmp.at<Vec3s>(r, c)[co] * 0 + tmp.at<Vec3s>(r, c + 1)[co];
                }
                
                resultpix[co] = (short) hsobel; 
            }
            dst.at<Vec3s>(r, c) = resultpix;  
        }
    }

    return 0;
}

/* Function to apply the Sobel Y 3X3 filter
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int sobelY3x3(const Mat &src, cv::Mat &dst ) {
    Mat tmp(src.rows, src.cols, CV_16S); 
    src.convertTo(tmp, CV_16S);
    src.convertTo(dst, CV_16S); 

    // Conduct vertical 1X3 Sobel filter
    for(int r = 0; r < src.rows; r++) {
        for(int c = 0; c < src.cols; c++) {
            Vec3s resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; co++) {
                short vsobel = 0; 
                if(r == 0) {
                    //vsobel = src.at<Vec3b>(r - 1, c)[co] + src.at<Vec3b>(r + 1, c)[co] * -1;
                    continue;
                } else if(r == src.rows - 1) {
                    //vsobel = src.at<Vec3b>(r - 1, c)[co] + src.at<Vec3b>(r + 1, c)[co] * -1;
                    continue;
                } else {
                    vsobel = src.at<Vec3b>(r - 1, c)[co] + src.at<Vec3b>(r, c)[co] * 0 + src.at<Vec3b>(r + 1, c)[co] * -1;
                }
                
                resultpix[co] = (short) vsobel; 
            }
            tmp.at<Vec3s>(r, c) = resultpix;  
        }
    }

    // Conduct horizontal 1X3 Sobel filter
    for(int r = 0; r < tmp.rows; r++) {
        for(int c = 0; c < tmp.cols; c++) {
            Vec3s resultpix = {0, 0, 0}; 
            for(int co = 0; co < 3; co++) {
                short hsobel = 0; 
                if(c == 0) {
                    //hsobel = tmp.at<Vec3s>(r, c - 1)[co] + tmp.at<Vec3s>(r, c + 1)[co]; 
                    continue;
                } else if(c == src.cols - 1) {
                    //hsobel = tmp.at<Vec3s>(r, c - 1)[co] + tmp.at<Vec3s>(r, c + 1)[co];
                    continue; 
                } else {
                    hsobel = tmp.at<Vec3s>(r, c - 1)[co] + tmp.at<Vec3s>(r, c)[co] * 2 + tmp.at<Vec3s>(r, c + 1)[co];
                }
                hsobel /= 4; 
                resultpix[co] = (short) hsobel; 
            }
            dst.at<Vec3s>(r, c) = resultpix;  
        }
    }

    return 0;
}

/* Function to get the absolute value of each color channel of each pixel 
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int absValOfPixels(const Mat &src, Mat &dst) {
    for(int r = 0; r < src.rows; ++r) {
        for(int c = 0; c < src.cols; ++c) {
            for(int co = 0; co < 3; ++co) {
                dst.at<Vec3b>(r,c)[co] = abs(src.at<Vec3s>(r,c)[co]); 
            }
        }
    }

    return 0; 
}

/* Function to calculate the magnitude of a given pixel
param: Mat &sx result from sobel x passed by reference
param Mat &sy result from sobel y passed by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int magnitude_nate(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst) {
    for(int r = 0; r < sx.rows; ++r) {
        for(int c = 0; c < sx.cols; ++c) {
            for(int co = 0; co < 3; ++co) {
                short sxval = sx.at<Vec3s>(r, c)[co]; 
                short syval = sy.at<Vec3s>(r, c)[co]; 
                uchar magnitude = sqrt(sxval * sxval + syval * syval); 
                dst.at<Vec3b>(r,c)[co] = magnitude; 
            }
        }
    }

    return 0; 
}

/* Function to blur and quantize an image
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: const int levels is the number of buckets the various colors should be separated into
return: 0 to indicate success
*/
int blurQuantize(const Mat &src, Mat &dst, const int levels) {
    Mat blurred; 
    src.copyTo(blurred);
    
    int success = blur5x5(src, blurred);

    uchar bucketSize = 255 / levels; 
    
    for(int r = 0; r < blurred.rows; r++) {
        for(int c = 0; c < blurred.cols; c++) {
            for(int co = 0; co < 3; co++) {
                uchar color = blurred.at<Vec3b>(r, c)[co]; 
                uchar tempColor = color / bucketSize; 
                uchar finalColor = tempColor * bucketSize; 
                dst.at<Vec3b>(r, c)[co] = finalColor;
            }
        }
    }

    return 0; 
}


/* Function to cartoonize an image
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: int levels indicates the number of buckets the various colors should be separated into
param: int magThreshold indicates the magnitude at which the borders should be made totally black
return: 0 to indicate success
*/
int cartoon(const Mat &src, Mat &dst, int levels, int magThreshold) {
    Mat xsob; 
    Mat ysob; 
    src.copyTo(xsob); 
    src.copyTo(ysob); 

    // Calc X/Y Sobel
    int success = sobelX3x3(src, xsob); 
    success = sobelY3x3(src, ysob); 

    // Calculate magnitude
    Mat gradMag; 
    src.copyTo(gradMag); 
    success = magnitude_nate(xsob, ysob, gradMag); 

    // Blur and quantize source
    Mat blurQuant; 
    src.copyTo(blurQuant); 
    success = blurQuantize(src, blurQuant, levels); 

    // Assign new values
    for(int r = 0; r < blurQuant.rows; r++) {
        for(int c = 0; c < blurQuant.cols; c++) {
            for(int co = 0; co < 3; co++) {
                uchar color = 0; 
                if(gradMag.at<Vec3b>(r,c)[co] > magThreshold) {
                    color = 0; 
                } else  {
                    color = blurQuant.at<Vec3b>(r,c)[co]; 
                }
                dst.at<Vec3b>(r,c)[co] = color; 
            }
        }
    }
    
    return 0; 
}

/* Function to cartoonize and clean an image with median filter
NOTE: I wrote this function bc my original cartoonization was grainy and I wanted to clean it. 
I realized I was doing it incorrectly, and fixed it, but I kept this filter. It no longer really serves a purpose.
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: int levels indicates the number of buckets the various colors should be separated into
param: int magThreshold indicates the magnitude at which the borders should be made totally black
return: 0 to indicate success
*/
int cartoonWithCleaning(const Mat &src, Mat &dst, int levels, int magThreshold) {
    Mat cartoonized; 
    src.copyTo(cartoonized); 
    int success = cartoon(src, cartoonized, levels, magThreshold); 
    
    // Aply median filter
    for(int r = 2; r < cartoonized.rows; r++) {
        for(int c = 2; c < cartoonized.cols; c++) {
            for(int co = 0; co < 3; co++) {
                uchar vals[9] = { cartoonized.at<Vec3b>(r - 1, c - 1)[co], cartoonized.at<Vec3b>(r - 1, c)[co], cartoonized.at<Vec3b>(r - 1, c + 1)[co],
                                  cartoonized.at<Vec3b>(r, c - 1)[co], cartoonized.at<Vec3b>(r, c)[co], cartoonized.at<Vec3b>(r, c + 1)[co],
                                  cartoonized.at<Vec3b>(r + 1, c - 1)[co], cartoonized.at<Vec3b>(r + 1, c)[co], cartoonized.at<Vec3b>(r + 1, c + 1)[co] }; 

                qsort(vals, 9, sizeof(uchar), compare); 

                dst.at<Vec3b>(r, c)[co] = vals[4]; 
            }
        }
    }

    return 0;
}

/*Compare function for qsort*/
int compare(const void * a, const void * b) {
    return (*(int*)a - *(int*)b); 
}

/* Function to change constrast and brightness
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: float contrast is the contrast coefficient
param: int brightness indicates the brightness coefficient
return: 0 to indicate success
*/
int contrastBrightness(const Mat &src, Mat &dst, float contrast, int brightness) {
    for(int r = 0; r < src.rows; r++) {
        for(int c = 0; c < src.cols; c++) {
            for(int co = 0; co < 3; co++) {
                // Adjust contrast and brightness depending on coefficients passed
                dst.at<Vec3b>(r, c)[co] = saturate_cast<uchar>(src.at<Vec3b>(r,c)[co] * contrast + brightness); 
            }
        }
    }

    return 0; 
}

/* Function to turn the image upside down
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
return: 0 to indicate success
*/
int upsideDown(const Mat &src, Mat &dst) {
    for(int r = 0; r < src.rows; r++) {
        for(int c = 0; c < src.cols; c++) {
            int drow = dst.rows - 1 - r; 
            int dcol = dst.cols - 1 - c; 
            dst.at<Vec3b>(drow, dcol) = src.at<Vec3b>(r, c); 
        }
    }
    return 0; 
}

/* Function to an image into a news screen
param: Mat &src passing source image by reference
param: Mat &dst passing destination image by reference
param: const string ticker is the text to be written at the top of the stream
param: const uint tickerStart indicates where the text should begin being displayed
param: Mat small image is the image to be shown in the graphic
return: 0 to indicate success
*/
int news(const Mat &src, Mat &dst, const string message, const string ticker, const uint tickerStart, Mat smallImage) {
    // Draws blue Rectangle
    rectangle(dst, Point(0, src.rows - 1), Point(90, src.rows - 50), Scalar(200, 0, 0), FILLED);
    
    // Draw the message rectangle
    rectangle(dst, Point(91, src.rows - 1), Point(src.cols - 1, src.rows - 50), Scalar(160, 160, 160), FILLED); 

    putText(dst, "NEWS", Point(0, src.rows - 15), FONT_HERSHEY_SIMPLEX, 1, Scalar::all(255), 5);

    putText(dst, message, Point(94, src.rows - 18), FONT_HERSHEY_SIMPLEX, .75, Scalar::all(255), 2); 

    // Next draw moving text box. 
    rectangle(dst, Point(0, 0), Point(src.cols - 1, 30), Scalar(0, 0, 0), FILLED);
    
    putText(dst, ticker, Point(tickerStart, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 1); 

    int labelw = getTextSize(ticker, FONT_HERSHEY_PLAIN, 1, 1, 0).width;

    if(tickerStart + labelw > src.cols) {
        int abosolutew = tickerStart + labelw; 
        int overflow = abosolutew - src.cols; 
        int start = overflow - labelw; 
        putText(dst, ticker, Point(start, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 1); 
    }

    // Add the smaller image to the large image. 
    smallImage.copyTo(dst(Rect(50, 50, smallImage.cols, smallImage.rows))); 
    return 0; 
}