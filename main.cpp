//
//  main.cpp
//  VehicleTracker
//
//  Created by Gonzalo Vera on 21/5/17.
//  Copyright © 2017 Gonzalo Vera. All rights reserved.
//

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
Mat preprocessImage(Mat frame);

/** Global variables */
String projectDir = "/Volumes/Data_Hard_Disk/Coding/My_OpenCV_examples/VehicleTracker/";
String dataDir = "/Volumes/Data_Hard_Disk/Coding/My_OpenCV_examples/TestData/";
String vehicle_cascade_name = projectDir + "cars_cascade.xml";
String video_filename = dataDir + "carsVideo3.mp4";
CascadeClassifier vehicle_cascade;



int main( void ) {
    
    VideoCapture capture(video_filename);
    
    Mat frame;
    
    // 1. Load the cascade
    if(!vehicle_cascade.load(vehicle_cascade_name)){
        printf("--(!)Error loading vehicle cascade\n");
        return -1;
    };

    
    // 2. Read the video stream
    if (!capture.isOpened()) {
        printf("--(!)Error opening video capture\n");
        return -1;
    };
    
    while (capture.read(frame)) {

        if(frame.empty()) {
            printf(" --(!) No captured frame -- Break!");
            break;
        };
        
        // 3. Apply classifier over the current frame
        detectAndDisplay(frame);
        capture.set(CV_CAP_PROP_FPS, 15); // Set video fps to reproduce

        waitKey(1);
    }
    
    return 0;
}




/*  ///  DETECT MULTI SCALE REVIEW  ///
scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm. 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.
minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

minSize – Minimum possible object size. Objects smaller than that are ignored.
This parameter determine how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detection.

maxSize – Maximum possible object size. Objects bigger than this are ignored.
This parameter determine how big size you want to detect. Again, you decide it! Usually, you don't need to set it manually, the default value assumes you want to detect without an upper limit on the size of the face.
*/

/*
    Perform car detection over the current frame
 */
void detectAndDisplay( Mat frame ) {
    
    vector<Rect> cars;
    
    // Detect cars
    vehicle_cascade.detectMultiScale(
                                  preprocessImage(frame),            // image
                                  cars,                  // output vector
                                  1.1,                   // scale factor
                                  7,                     // minimum neighbors (detection threshold)
                                  0|CASCADE_SCALE_IMAGE, // flags
                                  Size(70, 70)           // minimum size
                                  );
    
    for( size_t i = 0; i < cars.size(); i++ ) {
        
        // Track cars with square
        Rect rect(cars[i].x, cars[i].y, cars[i].width, cars[i].height);
        rectangle(frame, rect, Scalar(255,0,255), 2, 8, 0);
    }
    
    imshow("Face detection viewer", frame);
    
}


/*
    Perform a simple preprocessing over the frame before providing it to the car detection function.
    This enhances the edges on the image by equalizing its gray histogram.
 */
Mat preprocessImage(Mat frame){
    
    Mat frame_gray;
    
    cvtColor(
             frame,          // input Mat image
             frame_gray,     // output Mat image
             COLOR_BGR2GRAY  // colorspace conversion code
             );
    
    // Normalize the brightness of a grey image and increase it contrast
    equalizeHist(
                 frame_gray, // input frame
                 frame_gray  // output frame
                 );
    
    return frame_gray;
}

