---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: output_images/car_notcar.png
[image2]: output_images/hog_extraction.png
[image3]: output_images/sliding_window.png
[image4]: output_images/improved_scale.png
[image5]:  output_images/heatMap.png
[image6]: output_images/car_boxed.png
---
###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images.

HOG features are extracted from all the images found in the labeled data set [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).
 Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
 
![alt text][image1]

---
I tried different color spaces (YCrCb,LUV,YUV) combined with different skimage.feature.hog parameters (orientations, pixels_per_cell, and cells_per_block) to extract hog feartures of images from each of the two classes (car/notcar). Setting the visualise=True when calling hog function will give us the image representation of the features extracted.

 ```
 features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
 ```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. HOG parameters selection.

By trying different variation of hog parametes, I looked for the image representation of features extracted that can give more value in identifying certain class. Before combining Hog, bin spatial, and histogram features, I experimented with Hog parameters independenlty and found the following parameters to be good in detecting object edges/line orientation (CarND_VehicleDetection.py -> line 87):  
```
        color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32) # Spatial binning dimensions
  ```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I did the following steps in training the SVM classifier:
1. read and shuffled all labeled data set (vehicle/non-vehicle).(CarND_VehicleDetection.py -> line 72)
2. extracted HOG(all 3 channels), spatial, and histogram features of all
the images.(CarND_VehicleDetection.py -> line 106, functon_utils.py->extract_features function)
3. Split up data training(80%) and test(20%) sets. Made sure each set is balance between number of cars and not-cars (CarND_VehicleDetection.py -> line 138)
4. Train SVC classifier (CarND_VehicleDetection.py svc.fit(X_train, y_train)
 
 ###Sliding Window Search
####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the Hog  Sub-sampling window search (function_utils.py->find_cars function). The find_cars function accept road image (single or from a video frame) , hog parameters, spatial_size, histogram bin size, svc(classifier) and will do sliding window search. For each window that the code visit it extract the features and run through the svc classifier to predict if the region is car or not. All windows (bounding boxes) that were predicted as cars are returned. These bounding boxes some of which overlaps are eventually will be use to mark a real car(s)  after some thresholding, averaging.  I tried scales [0,5, 0.75,1,1.5,2] but found out that the lower the scale the more the false positive appears in the detection. I end it up using [1,1.5,2].


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

By combining YCrCb 3-channel HOG features, spatially binned color and histograms of color in the feature vector and experementing with the scale values in relation to sliding/overlap window,  I'm able to minimize the false positive scenario. See example images:

![alt text][image4]
---
### Video Implementation
####1. Final video output.  
Here's a [link to my video result](https://youtu.be/AClAYr3oGsU)

####2. Implementation of filter for false positives and method for combining overlapping bounding boxes.

All bounding boxes for positive detections were saved and a corresponding heatmap is created for all the boxes (CarND_VehicleDetection.py -> pipeline). Then a threshold has been applied to the map for vehicle position identification.
`scipy.ndimage.measurements.label()`is then applied to the thresholded heatmap  to identify individual blobs. Each blob identified can be assumed to a vehicle. The code attempted to create bounding boxes to cover the area of each blob detected (funtion_utils.py->draw_labeled_bboxes function)
### Here are four frames and their corresponding heatmaps:
![alt text][image5]

### Here is a sample of resulting bounding boxes drawn onto the image after filtering and thresholding.
![alt text][image6]


---

###Discussion
Looking at the annotated video, there are some few false positive detections. Also the bounding boxes sometimes are not fully covering the entire car. Bounding boxes also flickers. I also noticed that false positive sometimes just happen in an area where color has resemblance to a car(s)  that might have been part of the training set.Getting more data set (car and non-car) will probably help a bit in minimizing bad detection. Also experementing with other Classifier other that SVM will probably improve the detection accuracy.   

