/**
 * @file collect_data.cpp
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Program to collect sample image data
 * @date 2022-03-02
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/recognition.h"
#include "../include/csv_util.h"

enum Label { None, BlackPen, Brush, Calc, CrushedBeerCan, FoldedMask, GlassesCase, KnifeCase, Remote, SwitchController, WoodKnife }; 

std::string none = "None"; 
std::string blackpen = "blackpen"; 
std::string brush = "brush"; 
std::string fob = "fob"; 
std::string stone = "stone"; 
std::string foldedmask = "foldedmask"; 
std::string glassescase = "glassescase"; 
std::string knifecase = "knifecase"; 
std::string remote = "remote"; 
std::string switchcontroller = "switchcontroller"; 
std::string woodknife = "woodknife"; 

char fv_file_name[20] = "image_db.csv";   

int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev;

  bool test_data = false;

  // open the video device
  capdev = new cv::VideoCapture(2);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);
  
  cv::namedWindow("2D Obj Rec", 1); 
  cv::Mat frame;
  
  
  int img_num = 0;

  if(argc > 1) {
    img_num = atoi(argv[1]); 
  }

  bool conf_mat = false;

  if(argc > 2) {
    conf_mat = atoi(argv[2]);
  }

  int gather_image_data = 0; 
  if(argc > 3) {
    gather_image_data = atoi(argv[3]); 
  }

  std::vector<std::vector<float>> db_data;

  int success = read_image_data_csv(fv_file_name, db_data); // read the database
  if(success != 0) {
    printf("Error reading database\n"); 
    exit(-1); 
  }
  
  std::vector<float> db_stdevs(5); 
  get_db_stdev(db_stdevs, db_data); // calculate the standard deviations.  

  bool euc_dist = true;  

  string name = "./rptimgs/"; 

    
  for(;;) {
    int label = -1;
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }                 

    // see if there is a waiting keystroke
    
    int keyEx = cv::waitKeyEx(10);

    if(keyEx == 'q')
    {
      break;
    }

    else if(keyEx == 'k') {
      euc_dist = !euc_dist; 
    }

    else if(keyEx == 'n')
    {
      // Handles screenshot

      test_data = true;

      bool get_name = true;

      std::string lbl_name = "";

      while(get_name) {

        std::cout << "Please enter the label for this image" << std::endl;

        printf("0 - None\n");
        printf("1 - Black Pen\n");
        printf("2 - Brush\n");
        printf("3 - Fob\n");
        printf("4 - Stone\n");
        printf("5 - Folded Mask\n");
        printf("6 - Glasses Case\n");
        printf("7 - Knife Case\n");
        printf("8 - Remote\n");
        printf("9 - Switch Controller\n");
        printf("10 - Woodknife\n");

        std::cin >> label; 

        switch(label) {
          case 1:
            lbl_name = blackpen;
            get_name = false;
            break;

          case 2:
            lbl_name = brush;
            get_name = false;
            break;

          case 3:
            lbl_name = fob;
            get_name = false;
            break;

          case 4:
            lbl_name = stone;
            get_name = false;
            break;

          case 5:
            lbl_name = foldedmask;
            get_name = false;
            break;

          case 6:
            lbl_name = glassescase;
            get_name = false;
            break;

          case 7:
            lbl_name = knifecase;
            get_name = false;
            break;

          case 8:
            lbl_name = remote;
            get_name = false;
            break;

          case 9:
            lbl_name = switchcontroller;
            get_name = false;
            break;

          case 10:
            lbl_name = woodknife;
            get_name = false;
            break;

          default:
            break;
        }
      }

      std::string sPath = "./sample/" + lbl_name + "_" + std::to_string(img_num) + ".png";
      img_num++; 
      cv::imwrite(sPath, frame);
    }

    // threshold
    cv::Mat thresh;
    frame.copyTo(thresh);
    nate_threshold(frame, thresh, 127);
    if(gather_image_data) {
      name += "thresh" + std::to_string(gather_image_data) + ".png"; 
      cv::imwrite(name, thresh); 
      name = "./rptimgs/"; 
    }

    // morphological operations
    cv::Mat morph;
    frame.copyTo(morph);
    cleanup(thresh, morph, 4, 4);
    if(gather_image_data) {
      name += "morph" + std::to_string(gather_image_data) + ".png"; 
      cv::imwrite(name, morph); 
      name = "./rptimgs/"; 
    }

    // Find connected components
    cv::Mat labels(thresh.size(), CV_32S);
    cv::Mat stats;
    cv::Mat centroids;
    find_components(morph, labels, stats, centroids);
    if(gather_image_data) {
      cv::Mat dst; 
      frame.copyTo(dst); 
      show_components(morph, dst); 
      name += "comps" + std::to_string(gather_image_data) + ".png"; 
      cv::imwrite(name, dst); 
      name = "./rptimgs/"; 
    }

    int region_id = -1;

    determine_desired_region(region_id, frame, stats);

    // As of now, the feature vector holds 3 things:
    // The label, w to h ratio, and density
    // featvec = {id, label, feat 1, feat 2}
    std::vector<float> feat_vec(5);
    cv::Mat show_feats;
    frame.copyTo(show_feats);
    
    if(test_data) {
      feat_vec[ID] = (float) (rand() % 1000); // Id is somewhere between 0 and 500. Stored as float
      feat_vec[LABEL] = (float) label;
      calculate_features(labels, region_id, show_feats, feat_vec);
      append_image_data_csv(fv_file_name, feat_vec, 0);
      test_data = false; 
    } else {
      feat_vec[0] = -1;
      feat_vec[1] = -1;
      calculate_features(labels, region_id, show_feats, feat_vec);
      if(gather_image_data) {
        name += "feats" + std::to_string(gather_image_data) + ".png"; 
        cv::imwrite(name, show_feats); 
        name = "./rptimgs/"; 
      }
      
      int top_match = 0;
      // Then, here, I'll call get_distances()
      if(euc_dist) {
      
        std::map<int, float> id_distances;

        get_distances(db_data, feat_vec, db_stdevs, id_distances); // calculates the distance from each image 
        
        std::map<int, int> id_labels;
        vector_to_map(db_data, id_labels); // essentially creates a map with key = id and value = the label  
        top_match = get_top_match(id_distances, id_labels);    
      } else {
        std::vector<knn_point> db_points(db_data.size()); 

        for(int i = 0; i < db_data.size(); i++) {
          knn_point newPoint; 
          fv_2_knn_point(db_data[i], newPoint); 
          db_points[i] = newPoint; 
        } 

        knn_point targ_pnt; // init target knn point
        targ_pnt.w2h = feat_vec[W2H]; 
        targ_pnt.density = feat_vec[DENSITY]; 
        targ_pnt.hu0 = feat_vec[HU0]; 
        
        std::vector<int> cm_row_vec(12);  
        top_match = knn(db_points, 10, targ_pnt, db_stdevs[W2H], db_stdevs[DENSITY], db_stdevs[HU0], conf_mat, cm_row_vec); // get the top match

        char* cm_fname;
        //strcpy(cm_fname, "cm2.csv"); 
        if(conf_mat) {
          veci_append_image_data_csv(cm_fname, cm_row_vec);  
          conf_mat = false;
        }
      } 

    string lbl_nme = "";
    get_obj_name(top_match, lbl_nme); // Get text name of top match
    cv::putText(frame, lbl_nme, Point(5, 30), FONT_HERSHEY_PLAIN, 2, {0, 0, 0}, 2); // print it to the frame image.  
    
    }

    if(keyEx == 's') {
      int imnum = 0; 
      cout << "Enter image number" << endl; 
      cin >> imnum; 
      string name = "./rptimgs/recog" + std::to_string(imnum) + ".png";
      
      cv::imwrite(name, frame);
    }
    
    cv::imshow("2D Obj Rec", frame);
    gather_image_data = 0;
  }

  printf("Bye!\n"); 

  delete capdev;
  return(0);
}