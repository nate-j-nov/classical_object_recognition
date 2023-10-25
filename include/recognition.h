/**
 * @file recognition.h
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief H file for recognition.cpp, a library of functions to do real-time 2-D recognition
 * @note CS5330
 * @date 2022-02-18
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/filter.h"

#define ID 0
#define LABEL 1
#define W2H 2
#define DENSITY 3
#define HU0 4

struct knn_point {
  int label; 
  float w2h, density, hu0, distance; 
}; 

/**
 * @brief Function to conduct thresholding on the image. Uses one-sided threshold
 * @note I basically looked at the behavior of the the opencv version and built this based on that
 *      perhaps go to office hours about this.
 * @param src source image
 * @param dst image to be written to 
 * @param thresh is the threshold to separate colors
 * @return int returns non-zero val on failure
 */ 
int nate_threshold(const cv::Mat &src, cv::Mat &dst, const int thresh); 

/**
 * @brief Function to clean up the image using morphological filtering
 * 
 * @param src source image
 * @param dst image to be written to 
 * @param ero_iters Number of iterations of the erosion
 * @param dil_iters Number of iterations of dilation
 * @return int returns non-zero val on failure
 */
int cleanup(const cv::Mat &src, cv::Mat &dst, int dil_iters = 1, int ero_iters = 1); 

/**
 * @brief Function to find the connected components of an image
 * 
 * @param src source images
 * @param labels Mat to store labels in
 * @param stats Mat to store the stats for each connected component
 * @param centroids Mat to store the centroids
 * @return int returns non-zero value on failure. 
 */
int find_components(const cv::Mat &src, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids); 

/**
 * @brief Function to run connectivity algorithm and display connected components
 * 
 * @note I followed the following tutorial from OpenCV closely: https://docs.opencv.org/3.4/de/d01/samples_2cpp_2connected_components_8cpp-example.html
 * @param src source image
 * @param dst Image that displays the regions 
 * @return int returns non-zero value on failure
 */
int show_components(const cv::Mat &src, cv::Mat &dst); 

/**
 * @brief Function to determine which region is the one that contains the object
 * we are trying to recognize.
 * 
 * @param region_id region id that we will write to. This is the region id of the object we are recognizing
 * @param src source image being analyzed
 * @param stats stats is the stats output from the connected component 
 * @return int returns non-zero value on failure
 */
int determine_desired_region(int &region_id, const cv::Mat &src, const cv::Mat stats); 

/**
 * @brief Function to calculate translation, scale, and rotation invariant feautres. 
 * Right now, includes: central moment and oriented bounding box. 
 *
 * @param region_labels Mat that represents the labels for each pixel
 * @param region_id region Id for which to caculate these features. 
 * @param f_vec is a feature vector of type vector<double> that holds each image's features. 
 * @return int returns non-zero value on failure
 */
int calculate_features(const cv::Mat &region_labels, const int region_id, cv::Mat &dst, std::vector<float> &f_vec); 

/**
 * @brief Function to unrotate an x value
 * 
 * @param x_p integer that holds the x' value
 * @param y_p integer that holds the y' value
 * @param alpha value that holds the alpha angle, or axis of least central moment
 * @param x_bar center x of the region
 * @return int return the "image rotated" version of this x value. 
 */
int unrotate_x(int x_p, int y_p, double alpha, int x_bar); 

/**
 * @brief Function to unrotate a y value
 * 
 * @param x_p integer that holds the x' value
 * @param y_p integer that holds the y' value
 * @param alpha value that holds the alpha angle, or axis of least central moment
 * @param y_bar center x of the region
 * @return int return the "image rotated" version of this y value. 
 */
int unrotate_y(int x_p, int y_p, double alpha, int y_bar); 

/**
 * @brief Function to get the distance between two points
 * 
 * @param p1 First Point to compare 
 * @param p2 Second point to compare
 * @return double distance between the two points
 */
double get_distance(const cv::Point p1, const cv::Point p2); 

/**
 * @brief Function to find the standard deviation of a vector of floats
 * 
 * @param raw_data vector of floats to find the standard deviation 
 * @return float value of the standard deviation
 */
float std_dev(std::vector<float> raw_data); 

/**
 * @brief Get the standard deviations for each label
 * 
 * @param std_devs map<int, float> to write to 
 * @param db_data data that is extracted from the csv file. 
 * @return int returns non-zero value on failure
 */
int get_db_stdev(std::vector<float> &std_devs, std::vector<std::vector<float>> db_data); 

/**
 * @brief Function to find the scaled euclidean distance between two values
 * 
 * @param dist float distance to be written to. 
 * @param x1 first value to be compared
 * @param x2 second vlaue to be compared 
 * @param stdev_x standard deviation of feature X from the DB. 
 * @return int return non-zero value on failure. 
 */
int scaled_euc_dist(float &dist, float x1, float x2, float stdev_x); 

/**
 * @brief calculate the distances for each image in the database
 * 
 * @param db_data vector of floats that contains the images data 
 * @param target_fv vector of floats that holds the current target's feature vector. 
 * @param std_devs vector of floats that that are the standard deviations for each of the features
 * @param distances output map that holds the ID of the image and its distance from the currrent target
 * @return int return non-zero value on failure
 */
int get_distances(std::vector<std::vector<float>> db_data, std::vector<float> target_fv, std::vector<float> std_devs, std::map<int, float> &distances); 

/**
 * @brief Function to conduct the k-nearest neighbors algorithm 
 * 
 * @param test_points vector of test points
 * @param k the number of neighbors specified for the algorithm 
 * @param target point we are trying to classify 
 * @param w2h_stdev width to height ratio standard deviation
 * @param den_stdev density standard deviation
 * @param cent_loc_stdev centroid location standard deviation'
 * @param conf_mat bool to indicate making the confustion matrix
 * @param cm_vec the output vector of the cm row
 * @return int returns the label of the point 
 */
int knn(std::vector<knn_point> test_points, int k, knn_point target, float w2h_stdev, float den_stdev, float cent_loc_stdev, bool conf_mat, std::vector<int> &cm_vec); 

/**
 * @brief Function to detemine the top match based on the distance metric
 * 
 * @param distances map thta contains the ID of the image and its distance from the current feature vector. 
 * @param id_label_pairs map where key = id and value = label
 * @return int integer representation of the label
 */
int get_top_match(std::map<int, float> distances, std::map<int, int> id_label_pairs); 

/**
 * @brief Function to convert the DB data to the id_label pairs
 * 
 * @param db_data Data extracted from the db
 * @param id_label_pair map where key = id, value = label
 * @return int returns non-zero value on success 
 */
int vector_to_map(std::vector<std::vector<float>> db_data, std::map<int, int> &id_label_pair); 

/**
 * @brief Function to convert a feature vector to a knn_point struct
 * 
 * @param db_data feature vector from the db
 * @param point Output point that data is written to. 
 * @return int 
 */
int fv_2_knn_point(std::vector<float> db_data, knn_point &point); 

/**
 * @brief comparator function for sort function
 * 
 * @param p1 first point to compare
 * @param p2 second point to compare
 * @return true if p1 < p2
 * @return false if p1 >= p2 
 */
bool compare_dist(knn_point p1, knn_point p2); 

/**
 * @brief Get the name of the object that was identified
 * 
 * @param label integer label that is determined from the modelling
 * @param name name of the label to be copied to. 
 * @return int returns non-zero value on failure. 
 */
int get_obj_name(int label, string &name); 