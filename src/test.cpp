/**
 * @file test.cpp
 * @author Nate Novak (novak.n@northeastern.edu@domain.com)
 * @brief Driver for Project 3
 * @note CS5330
 * @date 2022-02-18
 * 
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include "../include/recognition.h"

#define BUFSIZE 256
#define BINS 16

using namespace std;

int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  DIR *dirp;
  struct dirent *dp;

  if( argc < 2) {
    printf("usage: %s <directory path>", argv[0]);
    exit(-1);
  }
  
  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    char curdir[256]; 
    getcwd(curdir, 256); 
    printf("CURR DIR: %s\n", curdir); 

    exit(-1);
  }
  
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	    strstr(dp->d_name, ".png") ||
	    strstr(dp->d_name, ".ppm") ||
	    strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      // MY PROCESSING
      cv::Mat src = cv::imread(buffer, cv::IMREAD_COLOR); 
      cv::Mat thresh; 
      src.copyTo(thresh); 
      nate_threshold(src, thresh, 127); 

      cv::Mat my_morph; 
      src.copyTo(my_morph); 
      cleanup(thresh, my_morph, 4, 4); 

      cv::Mat labels(thresh.size(), CV_32S);
      cv::Mat stats; 
      cv::Mat centroids; 
      find_components(my_morph, labels, stats, centroids);
      
      int region_id = -1; 
    
      determine_desired_region(region_id, src, stats);
      
      for(int i = 0; i < my_morph.rows; i++) {
        for(int j = 0; j < my_morph.cols; j++) {
          if(labels.at<int>(i, j) == region_id) {
            my_morph.at<uchar>(i, j) = 50; 
          }
        }
      }

      //printf("cent xbar: %.4f | cent ybar: %.4f\n"); 
      printf("centroids type: %d; centroids size: (%d,%d)\n", centroids.depth(), centroids.rows, centroids.cols);
      printf("stats type: %d; stats size: (%d,%d)\n", stats.depth(), stats.rows, stats.cols);
      printf("region id = %d\n", region_id);
      for(int i = 0; i < centroids.rows; i++) {
        printf("desired region %d centroid: (%0.2f, %0.2f)\n", i, centroids.at<double>(i,0), centroids.at<double>(i,1) );    
      }
      //printf("desired region centroid: (%0.2f, %0.2f)\n", centroids.at<float>(region_id,0), centroids.at<float>(region_id,1) );


      cv::Mat features; 
      src.copyTo(features); 
      std::vector<float> fvec(4); 
      calculate_features(labels, region_id, features, fvec);
      
      printf("features calculated\n"); 
      
      // OPENCV FUNCTIONS
      /*
      cv::Mat temp;
      src.copyTo(temp); 
      cv::threshold(src, temp, 127, 255, cv::THRESH_BINARY); 

      cv::Mat ero;
      src.copyTo(ero); 
      cv::Mat kernel = cv::Mat::ones(1, 1, CV_8U); 
      cv::erode(temp, ero, kernel); 
      cv::Mat dil; 
      src.copyTo(dil); 
      cv::dilate(ero, dil, kernel);*/
      

      for(;;) {

        char key = waitKey(10); 
        cv::imshow("Test", features);
        //cv::imshow("OpenCV methods", temp); 

        if(key == 'q') {
          break; 
        } 
      }
    }
  }

  printf("Terminating\n");

  return(0);
}