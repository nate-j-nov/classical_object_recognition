/**
 * @file recognition.cpp
 * @author Nate Novak (novak.n@northeastern.edu) 
 * @brief Library of functions to conduct Real-Time Object Recognition
 * @note CS5330
 * @date 2022-02-18
 */

#include "../include/recognition.h"

/**
 * @brief Function to conduct thresholding on the image. Uses one-sided threshold
 * @note I basically looked at the behavior of the the opencv version and built this based on that
 *      perhaps go to office hours about this.
 * @param src source image
 * @param dst image to be written to 
 * @param thresh is the threshold to separate colors
 * @return int returns non-zero val on failure
 */ 
int nate_threshold(const cv::Mat &src, cv::Mat &dst, const int thresh) {
  // First, blur the image
  cv::Mat blurred; 
  src.copyTo(blurred); 
  int success = blur5x5(src, blurred); 

  if(success != 0) {
    printf("error blurring image\n"); 
    return -1; 
  }
  
  // Caluculate saturation 
  //source: http://help.cognex.com/Content/KB_Topics/In-Sight/ToolsFunctions/696.htm
  for(int i = 0; i < blurred.rows; i++) {
    for(int j = 0; j < blurred.cols; j++) {
      float saturation = 0.0; 
      Vec3b pix_v = blurred.at<Vec3b>(i, j); // get current pixel
      std::vector<uchar> pix = { pix_v[0], pix_v[1], pix_v[2] }; 
      uchar max = *max_element(pix.begin(), pix.end());
      uchar min = *min_element(pix.begin(), pix.end()); 
      
      if(min == 0) min = 1; // Avoiding divide by zero errors
      
      short temp_den = max + min; // temporary denominator
      
      uchar sat_den = 0; 
      if(temp_den > 255) { // denominator has max of 255
        sat_den = 255; 
      } else {
        sat_den = temp_den; 
      }
      
      uchar sat_num = max - min; 

      saturation = ( (double) sat_num / (double) (sat_den) ); 

      if(isgreater(saturation, 0.5)) {
        for(int k = 0; k < 3; k++) {
          int newval = blurred.at<Vec3b>(i, j)[k] /= 4; 
        }
      }
    }
  }

  cv::Mat gray; 
  cv::cvtColor(blurred, gray, COLOR_BGR2GRAY);
  cv::cvtColor(blurred, dst, COLOR_BGR2GRAY); 

  // Thresholding
  for(int i = 0;  i < gray.rows; i++) {
    for(int j = 0; j < gray.cols; j++) {
      for(int c = 0; c < 3; c++) {
        uchar val = gray.at<uchar>(i, j); 
        if(val < thresh) {
          dst.at<uchar>(i, j) = 255; 
        } else {
          dst.at<uchar>(i, j) = 0; 
        }
      }
    }
  }

  return 0; 
}

/**
 * @brief Function to clean up the image using morphological filtering
 * 
 * @param src source image
 * @param dst image to be written to 
 * @param ero_iters Number of iterations of the erosion
 * @param dil_iters Number of iterations of dilation
 * @return int returns non-zero val on failure
 */
int cleanup(const cv::Mat &src, cv::Mat &dst, int dil_iters, int ero_iters) {
  cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U); 
  cv::Mat temp; 
  src.copyTo(temp); 
  cv::dilate(src, temp, kernel, Point(-1, -1), dil_iters);
  cv::erode(temp, dst, kernel, Point(-1, -1), ero_iters);
  return 0; 
}

/**
 * @brief Function to find the connected components of an image
 * 
 * @param src source images
 * @param labels Mat to store labels in
 * @param stats Mat to store the stats for each connected component
 * @param centroids Mat to store the centroids
 * @return int returns non-zero value on failure. 
 */
int find_components(const cv::Mat &src, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids) {
  int n_labels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);  
  return 0; 
}

/**
 * @brief Function to run connectivity algorithm and display connected components
 * 
 * @note I followed the following tutorial from OpenCV closely: https://docs.opencv.org/3.4/de/d01/samples_2cpp_2connected_components_8cpp-example.html
 * @param src source image
 * @param dst Image that displays the regions 
 * @return int returns non-zero value on failure
 */
int show_components(const cv::Mat &src, cv::Mat &dst) {
  cv::Mat label_image(src.size(), CV_32S);  

  int n_labels = cv::connectedComponents(src, label_image, 8, CV_32S); 

  std::vector<Vec3b> color_vec;

  color_vec.push_back( Vec3b(0, 0, 0) ); // set the background

  for(int n = 1; n < n_labels; n++) {
    color_vec.push_back( Vec3b((rand()&255), (rand()&255), (rand()&255)) ); 
  }  

  for(int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      int region_num = label_image.at<int>(i, j);
      Vec3b color = color_vec[region_num]; 
      dst.at<Vec3b>(i, j) = color; 
    }
  }

  return 0; 
}

/**
 * @brief Function to determine which region is the one that contains the object
 * we are trying to recognize.
 * 
 * @param region_id region id that we will write to. This is the region id of the object we are recognizing
 * @param src source image being analyzed
 * @param stats stats is the stats output from the connected component 
 * @return int returns non-zero value on failure
 */
int determine_desired_region(int &region_id, const cv::Mat &src, const cv::Mat stats) {
  // The method I utilize is finding the object with the most central bounding box. 
  std::map<int, std::vector<int>> regions;

  // First, let's just filter out tiny regions in case there's any noise near the center.
  // Also, we skip the background, which is always the first item.  
  for(int i = 1; i < stats.rows; i++) {
    int area = stats.at<int>(i, CC_STAT_AREA); // get the area
    if(area > 10) { // If it's greater than 10 pixels, we add it to the map of regions. 
      std::vector<int> stat_vec(stats.cols);
      for(int j = 0; j < stats.cols; j++) {
        stat_vec[j] = stats.at<int>(i, j);  // Create vector to add to the map.  
      }
      regions[i] = stat_vec; 
    }
  }
  
  // Now it's time to determine which ones are most central 
  int center_x = src.cols / 2; 
  int center_y = src.rows / 2;  
  int cur_min_reg_id = -1; 
  double cur_min_reg_dist = std::numeric_limits<double>::max(); 
  
  for(auto const& item : regions) {
    std::vector<int> this_stat_vec = item.second;

    // get various numbers we need 
    int this_left = this_stat_vec[CC_STAT_LEFT]; 
    int this_top = this_stat_vec[CC_STAT_TOP]; 
    int this_width = this_stat_vec[CC_STAT_WIDTH]; 
    int this_height = this_stat_vec[CC_STAT_HEIGHT]; 

    // calc distance from center of image
    int this_cen_x = this_left + (this_width / 2); 
    int this_cen_y = this_top + (this_height / 2); 
    int diff_x = (this_cen_x - center_x) * (this_cen_x - center_x); 
    int diff_y = (this_cen_y - center_y) * (this_cen_y - center_y); 
    int sum_dif = diff_x + diff_y; 
    double this_dist_from_center = sqrt(sum_dif); 
    // assign current min values if this is less than current. 
    if(isless(this_dist_from_center, cur_min_reg_dist)) {
      cur_min_reg_id = item.first; 
      cur_min_reg_dist = this_dist_from_center; 
    }
  }
  region_id = cur_min_reg_id; 

  return 0; 
}

/**
 * @brief Function to calculate translation, scale, and rotation invariant feautres. 
 * Right now, includes: central moment and oriented bounding box. 
 *
 * @param region_labels Mat that represents the labels for each pixel
 * @param region_id region Id for which to caculate these features. 
 * @param f_vec is a feature vector of type vector<double> that holds each image's features. 
 * @return int returns non-zero value on failure
 */
int calculate_features(const cv::Mat &region_labels, const int region_id, cv::Mat &dst, std::vector<float> &f_vec)
{ 
  // Create mask
  cv::Mat mask(region_labels.rows, region_labels.cols, CV_8U);  
  for(int i = 0; i < region_labels.rows; i++) {
    for(int j = 0; j < region_labels.cols; j++) {
      if(region_labels.at<int>(i, j) == region_id) {
        mask.at<uchar>(i, j) = 255; 
      } else {
        mask.at<uchar>(i, j) = 0; 
      }
    }
  }

  cv::Moments moments = cv::moments(mask, true);
  double huMoments[7]; 
  cv::HuMoments(moments, huMoments); 

  // calculate axis of least central moment
  double alpha_num = 2.0 * moments.mu11; 
  double alpha_dem = moments.mu20 - moments.mu02; 
  double alpha = 0.5 * atan2(alpha_num, alpha_dem);
  alpha *= -1; 

  // get xbar ybar
  double xbar = moments.m10 / moments.m00;   
  double ybar = (region_labels.rows - 1) - (moments.m01 / moments.m00);
  cv::Point centroid(xbar, (region_labels.rows - 1 ) - ybar); 

  //printf("xbar: %.4f | ybar: %.4f\n", xbar, ybar); 
  
  // declare and init min and max x', y'
  int min_x_p = INT_MAX;   
  int max_x_p = INT_MIN; 
  int min_y_p = INT_MAX;
  int max_y_p = INT_MIN;

  int activated_pixels = 0;
    
  // Loop through image to find the min/max x', y'
  for(int i = 0; i < region_labels.rows; i++) {
    for(int j = 0; j < region_labels.cols; j++) {
      if(region_labels.at<int>(i, j) == region_id) {
        activated_pixels++; 
        int x = j; 
        int y = (region_labels.rows - 1) - i; 
        float x_p_f = ( (x - xbar) * cos(alpha) ) + ( (y - ybar) * sin(alpha) ); 
        float y_p_f = ( (x - xbar) * -1.0 * sin(alpha) ) + ( (y - ybar) * cos(alpha) ); 
        
        int x_p = (int) x_p_f; 
        int y_p = (int) y_p_f; 
        
        if(x_p < min_x_p) {
          min_x_p = x_p; 
        }
        if(x_p > max_x_p) {
          max_x_p = x_p; 
        }
        if(y_p < min_y_p) {
          min_y_p = y_p; 
        }
        if(y_p > max_y_p) {
          max_y_p = y_p; 
        }
      }
    }
  }

  //printf("min_x_p: %d max_x_p: %d min_y_p: %d max_y_p: %d\n", min_x_p, max_x_p, min_y_p, max_y_p); 

  // 2nd point to display the major axis. 
  int x2 = xbar + 200 * cos(alpha); 
  int y2 = ybar + 200 * sin(alpha);
  
  cv::Point p2(x2, (region_labels.rows - 1) - y2); 

  // 2nd point to display the minor axis. 
  int x3 = xbar + 200 * -1 * sin(alpha); 
  int y3 = ybar + 200 * cos(alpha); 
  cv::Point p3(x3, (region_labels.rows - 1) - y3); 

  // Now, let's color in dst
  for(int i = 0; i < region_labels.rows; i++) {
    for(int j = 0; j < region_labels.cols; j++) {
      if(region_labels.at<int>(i, j) == region_id) {
        cv::Vec3b newcolor = { 255, 0, 0 }; 
        dst.at<Vec3b>(i, j) = newcolor; 
      } else {
        dst.at<Vec3b>(i, j) = {0, 0, 0}; 
      }
    }
  }

  // now, let's unrotate the points we have.
  // Oriented bounding box

  // "bottom left" 
  int min_x_min_y_x_u = unrotate_x(min_x_p, min_y_p, alpha, xbar); 
  int min_x_min_y_c = min_x_min_y_x_u; 
  int min_x_min_y_y_u = unrotate_y(min_x_p, min_y_p, alpha, ybar);
  int min_x_min_y_r = (region_labels.rows - 1) - min_x_min_y_y_u;

  cv::Point bottom_left(min_x_min_y_c, min_x_min_y_r);  

  // "top left"
  int min_x_max_y_x_u = unrotate_x(min_x_p, max_y_p, alpha, xbar); 
  int min_x_max_y_c = min_x_max_y_x_u;
  int min_x_max_y_y_u = unrotate_y(min_x_p, max_y_p, alpha, ybar); 
  int min_x_max_y_r = (region_labels.rows - 1) - min_x_max_y_y_u; 

  cv::Point top_left(min_x_max_y_c, min_x_max_y_r); 

  // "bottom right" 
  int max_x_min_y_x_u = unrotate_x(max_x_p, min_y_p, alpha, xbar);
  int max_x_min_y_c = max_x_min_y_x_u;
  int max_x_min_y_y_u = unrotate_y(max_x_p, min_y_p, alpha, ybar);
  int max_x_min_y_r = (region_labels.rows - 1) - max_x_min_y_y_u;

  cv::Point bottom_right(max_x_min_y_c, max_x_min_y_r);

  // "top right"  
  int max_x_max_y_x_u = unrotate_x(max_x_p, max_y_p, alpha, xbar); 
  int max_x_max_y_c = max_x_max_y_x_u;
  int max_x_max_y_y_u = unrotate_y(max_x_p, max_y_p, alpha, ybar); 
  int max_x_max_y_r = (region_labels.rows - 1) - max_x_max_y_y_u; 

  cv::Point top_right(max_x_max_y_c, max_x_max_y_r); 

  // Draw lines onto output 
  cv::line(dst, bottom_left, top_left, {0, 255, 255 }); // left
  cv::line(dst, top_left, top_right, {0, 255, 255}); // top 
  cv::line(dst, top_right, bottom_right, {0, 255, 255}); // right
  cv::line(dst, bottom_right, bottom_left, {0, 255, 255}); // bottom

  cv::line(dst, centroid, p2, {255, 255, 0}, 3); // major axis
  cv::line(dst, centroid, p3, {0, 255, 0}, 3); // minor axis

  cv::circle(dst, centroid, 4, {255, 255, 255}, 4);

  // Now, let's get the features we're going to calculate
  float width = get_distance(bottom_left, bottom_right); 
  float height = get_distance(bottom_left, top_left); 

  float w_2_h_rat = width / height;

  f_vec[W2H]= w_2_h_rat; 

  float density = ((float) activated_pixels) / (width * height);
  f_vec[DENSITY] = density;

  f_vec[HU0] = (float) huMoments[0];

  return 0; 
}

/**
 * @brief Function to unrotate an x value
 * 
 * @param x_p integer that holds the x' value
 * @param y_p integer that holds the y' value
 * @param alpha value that holds the alpha angle, or axis of least central moment
 * @param x_bar center x of the region
 * @return int return the "image rotated" version of this x value. 
 */
int unrotate_x(int x_p, int y_p, double alpha, int x_bar) {
  return ((x_p * cos(alpha)) - (y_p * sin(alpha))) + x_bar; 
}

/**
 * @brief Function to unrotate a y value
 * 
 * @param x_p integer that holds the x' value
 * @param y_p integer that holds the y' value
 * @param alpha value that holds the alpha angle, or axis of least central moment
 * @param y_bar center x of the region
 * @return int return the "image rotated" version of this y value. 
 */
int unrotate_y(int x_p, int y_p, double alpha, int y_bar) {
  return ((x_p * sin(alpha)) + (y_p * cos(alpha))) + y_bar;  
}

/**
 * @brief Function to get the distance between two points
 * 
 * @param p1 First Point to compare 
 * @param p2 Second point to compare
 * @return double distance between the two points
 */
double get_distance(const cv::Point p1, const cv::Point p2) {
  int diff_x = (p1.x - p2.x) * (p1.x - p2.x); 
  int diff_y = (p1.y - p2.y) * (p1.y - p2.y); 
  int sum_dif = diff_x + diff_y; 
  return sqrt(sum_dif);
}

/**
 * @brief Function to find the standard deviation of a vector of floats
 * 
 * @param raw_data vector of floats to find the standard deviation 
 * @return float value of the standard deviation
 */
float std_dev(std::vector<float> raw_data) {
  float sum = 0.0; 
  int n = raw_data.size(); 

  // get mean
  for(int i = 0; i < n; i++) {
    sum += raw_data[i]; 
  }

  float mean = sum / ((float) n);

  // get std dev
  float variance = 0.0; 
  for(int k = 0; k < n; k++) {
    float value = raw_data[k] - mean;  
    variance += (value * value); 
  }
  
  variance /= ((float) n); 
  float stdev = sqrt(variance);

  return stdev;  
}

/**
 * @brief Get the standard deviations for each label
 * @note MUST BE UPDATED EACH TIME I ADD A FEATURE
 * @param std_devs vector of standard deviations to write to.   
 * @param db_data data that is extracted from the csv file. 
 * @return int returns non-zero value on failure
 */
int get_db_stdev(std::vector<float> &std_devs, std::vector<std::vector<float>> db_data) {
  std::vector<float> w2h_data(db_data.size());  
  std::vector<float> density_data(db_data.size()); 
  std::vector<float> cent_loc_data(db_data.size()); 

  // Loop through the data in the database. 
  for(int i = 0; i < db_data.size(); i++) {
    w2h_data[i] = db_data[i][W2H]; 
    density_data[i] = db_data[i][DENSITY];
    cent_loc_data[i] = db_data[i][HU0];  
  }

  // get stnadard deviation for each feature
  float stdev_w2h = std_dev(w2h_data);  
  float stdev_density = std_dev(density_data);
  float stdev_cent_loc = std_dev(cent_loc_data);  

  // add to the std_devs vector
  if(std_devs.size() != db_data[0].size()) {
    printf("error: std_devs array and number features do not match\n"); 
    printf("std_dev.size==%d, db_data[0].size() ==%d\n", std_devs.size(), db_data[0].size()); 
    return -1; 
  }

  std_devs[ID] = 0.1; // dummy values so all vectors are the same length.  
  std_devs[LABEL] = 0.1; // Dummy value. 
  std_devs[W2H] = stdev_w2h; 
  std_devs[DENSITY] = stdev_density;
  std_devs[HU0] = stdev_cent_loc;

  return 0; 
}

/**
 * @brief Function to find the scaled euclidean distance between two values
 * 
 * @param dist float distance to be written to. 
 * @param x1 first value to be compared
 * @param x2 second vlaue to be compared 
 * @param stdev_x standard deviation of feature X from the DB. 
 * @return int return non-zero value on failure. 
 */
int scaled_euc_dist(float &dist, float x1, float x2, float stdev_x) {
  dist = (x1 - x2) * (x1 - x2);
  dist /= (stdev_x * stdev_x ); 
  return 0; 
}

/**
 * @brief calculate the distances for each image in the database
 * 
 * @param db_data vector of floats that contains the images data 
 * @param target_fv vector of floats that holds the current target's feature vector. 
 * @param std_devs vector of floats that that are the standard deviations for each of the features
 * @param distances output map that holds the ID of the image and its distance from the currrent target
 * @return int return non-zoer value on failure
 */
int get_distances(std::vector<std::vector<float>> db_data, std::vector<float> target_fv, std::vector<float> std_devs, std::map<int, float> &distances) {
  // Loop through all of the images in my db.  
  for(int i = 0; i < db_data.size(); i++) {
    std::vector<float> cur_fv = db_data[i]; 
    float dist = 0.0; 

    // calculate the distance
    float w2h_dist = 0.0; 
    scaled_euc_dist(w2h_dist, target_fv[W2H], cur_fv[W2H], std_devs[W2H]); 
    dist += w2h_dist; 

    float density_dist = 0.0; 
    scaled_euc_dist(density_dist, target_fv[DENSITY], cur_fv[DENSITY], std_devs[DENSITY]); 
    dist += density_dist; 

    float cent_loc_dist = 0.0; 
    scaled_euc_dist(cent_loc_dist, target_fv[HU0], cur_fv[HU0], std_devs[HU0]); 
    dist += cent_loc_dist; 

    int img_id = (int) cur_fv[0]; 

    distances[img_id] = dist; // add to distances map 
  }

  return 0; 
}

/**
 * @brief Function to detemine the top match based on the distance metric
 * 
 * @param distances map thta contains the ID of the image and its distance from the current feature vector. 
 * @param id_label_pairs map where key = id and value = label
 * @return int integer representation of the label
 */
int get_top_match(std::map<int, float> distances, std::map<int, int> id_label_pairs) {
  float min_dist = std::numeric_limits<float>::max();
  int min_id = -1;   
  for(auto const& item : distances) {
      if(isless(item.second, min_dist)) {
        min_dist = item.second; 
        min_id = (int) item.first; 
      }
  }

  return id_label_pairs[min_id]; 
}

/**
 * @brief Function to convert the DB data to the id_label pairs
 * 
 * @param db_data Data extracted from the db
 * @param id_label_pair map where key = id, value = label
 * @return int returns non-zero value on success 
 */
int vector_to_map(std::vector<std::vector<float>> db_data, std::map<int, int> &id_label_pair) {
  for(int i = 0; i < db_data.size(); i++) {
    std::vector<float> cur_fv = db_data[i]; 
    int id = (int) cur_fv[0]; 
    int label = (int) cur_fv[1]; 
    id_label_pair[id] = label; 
  }
  
  return 0; 
}

/**
 * @brief Function to convert a feature vector to a knn_point struct
 * 
 * @param db_data feature vector from the db
 * @param point Output point that data is written to. 
 * @return int 
 */
int fv_2_knn_point(std::vector<float> db_data, knn_point &point) {
  int label = (int) db_data[1]; 
  float w2h = db_data[2]; 
  float density = db_data[3];

  point.label = label; 
  point.w2h = w2h; 
  point.density = density; 
  
  return 0; 
}

/**
 * @brief Function to conduct the k-nearest neighbors algorithm 
 * Followed geeks for geeks quite closely
 * Link: https://www.geeksforgeeks.org/k-nearest-neighbours/
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
int knn(std::vector<knn_point> test_points, int k, knn_point target, float w2h_stdev, float den_stdev, float cent_loc_stdev, bool conf_mat, std::vector<int> &cm_vec) {
  int size = test_points.size(); 

  // Get all of the distances of the points 
  for(int i = 0; i < size; i++) {
    float dist_w2h = 0.0; 
    scaled_euc_dist(dist_w2h, test_points[i].w2h, target.w2h, w2h_stdev);  
    float dist_den = 0.0; 
    scaled_euc_dist(dist_den, test_points[i].density, target.density, den_stdev);
    float dist_hu0 = 0.0; 
    scaled_euc_dist(dist_hu0, test_points[i].hu0, target.hu0, cent_loc_stdev);  

    float dist_final = dist_w2h + dist_den + dist_hu0; 

    test_points[i].distance = dist_final; 
  }

  // sort the points 
  std::sort(test_points.begin(), test_points.end(), compare_dist); 

  std::map<int, int> freq; // map of the frequencies. 

  for(int i = 1; i <= 10; i++)  {
    freq[i] = 0; // init all to zero
  }

  // Get the frequencies of of each label within the closest k neighbors
  for(int i = 0; i < k; i++) {
    if(test_points[i].label == 1) {
      freq[1]++; 
    } else if(test_points[i].label == 2) {
      freq[2]++; 
    } else if(test_points[i].label == 3) {
      freq[3]++;  
    }else if(test_points[i].label == 4) {
      freq[4]++;  
    }else if(test_points[i].label == 5) {
      freq[5]++;  
    }else if(test_points[i].label == 6) {
      freq[6]++;  
    }else if(test_points[i].label == 7) {
      freq[7]++;  
    }else if(test_points[i].label == 8) {
      freq[8]++;  
    }else if(test_points[i].label == 9) {
      freq[9]++;  
    }else if(test_points[i].label == 10) {
      freq[10]++;
    }
  }

  if(conf_mat) {
    if(cm_vec.size() != 12)  {
      printf("error: confusion matrix vector not the right size\n"); 
      exit(-1); 
    }

    int true_lbl = 0; 
    std::cout << "What's the true label for this image?" << endl; 
    std::cin >> true_lbl;  
    cm_vec[0] = true_lbl;
    int sum = 0;  
    for(int i = 1; i <= 10; i++) {
      cm_vec[i] = freq[i]; 
      sum += freq[i]; 
    } 
    cm_vec[11] = sum;  
  }

  int max_freq = -1; 
  int max_freq_id = -1; 

  // Get the max frequency Id. 
  for(auto const& item : freq) {
    if(item.second > max_freq) {
      max_freq = item.second; 
      max_freq_id = item.first; 
    }
  }

  return max_freq_id; 
}

/**
 * @brief comparator function for sort function
 * 
 * @param p1 first point to compare
 * @param p2 second point to compare
 * @return true if p1 < p2
 * @return false if p1 >= p2 
 */
bool compare_dist(knn_point p1, knn_point p2) {
  return isless(p1.distance, p2.distance); 
}

/**
 * @brief Get the name of the object that was identified
 * 
 * @param label integer label that is determined from the modelling
 * @param name name of the label to be copied to. 
 * @return int returns non-zero value on failure. 
 */
int get_obj_name(int label, string &name) {
  switch(label) {
    case 1:
      name = "Black Pen"; 
      return 0; 

    case 2:
      name = "Brush"; 
      return 0;

    case 3:
      name = "Fob"; 
      return 0; 

    case 4:
      name = "Stone"; 
      return 0;

    case 5:
      name = "Folded Mask";  
      return 0;

    case 6:
      name = "Glasses Case"; 
      return 0; 

    case 7:
      name = "Knife Case"; 
      return 0; 

    case 8:
      name = "Remote"; 
      return 0; 
    
    case 9:
      name = "Switch Controller";  
      return 0; 

    case 10:
      name = "Wood knife";  
      return 0; 

    default:
      name = "None";
      return 0; 
    }
}
