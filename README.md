
# Vehicle Detection Project

[//]: # (Image References)

[training_data]: ./training_data.PNG
[hog]: ./hog.PNG
[slide_window]: ./slide_window.PNG
[heat_map_label]: ./heat_map_label.PNG
[hog_subsampling_heatmap]: ./hog_subsampling_heatmap.PNG



## Goals

The goals/steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.


### Histogram of Oriented Gradients (HOG)

#### Extracted HOG features from the training images


**Data Class Distribution**
- Vehicle train images count: 8792
- Non-vehicle train image count: 8968

![training_data]

I used the block of code provided by Udacity to extract the features from an image.
The function `extract_features` combine the other function and use the class `FeatureParameters` to hold all the parameters in a single place.

The following is example for HOG for a vehicle and non-vehicle:

![hog]

#### HOG parameters
I have spent some time experimenting different parameters values, my goal is to basically maximise the accuracy.
I train my model with images in **training_images** folder and later test my model performance on the images in **test_images** folder.
After several experiments of parameter tuning, I came up the following:

|Parameter|Value|
|:--------|----:|
|Color Space|YCrCb|
|HOG Orient|8|
|HOG Pixels per cell|8|
|HOG Cell per block|2|
|HOG Channels|All|
|Spatial bin size| (16,16)|
|Histogram bins|32|
|Histogram range|(0,256)|
|Classifier|LinearSVC|
|Scaler|StandardScaler|

The classifier accuracy is about **0.9921**.

### Sliding Window Search

#### Scales to search and overlap windows

I first implemented **sliding windows** to calculate all the windows, then I applied **feature extraction** to each window as to find the one with vehicle.

The best suited parameters for **scale to search** and **overlap windows** were simply found via multiple experiments until a successful result on the test images found.

The following image are the results on the test images:

![slide_window]

To further eliminate the false positive(window that does NOT contains vehicle) and combine the overlapping windows(blue boxes) found above, a heat map was implemented with a certain threshold and function `label()` of `scipy.ndimage.measurements` was implemented to locate the vehicles and resolving overlapping bounding boxes. 

The following image is the results on some test images:

![heat_map_label]

#### Optimize the performance on SVM classifier

**HOG sub-sampling** was attempted(**with false positives eliminated**) and It did improve quite huge amount of performance and spped up the HOG process.

The following image are the results applied to the test images and we can see that the window is **more accurately fit to the whole vehicle body** from the 1st and last images for instance...

![hog_subsampling_heatmap]

### Video Pipeline

#### Final video output

project_video_output.mp4

#### Filter for false positives and way to combining overlapping bounding boxes

As mention above, a heatmap with threshold has implemented to reduce false positives.

Furthermore, The overlapping bounding boxes were resolved using `label()` of `scipy.ndimage.measurements` to locate vehicles as mentioned. 

The heatmap image also been **averaged over 3 consecutive frames** as to filter false positives.

### Some Reflection

I found the main problem is the performance of the pipeline itself although it does improve a lot with hog subsampling. And just for future reference, I would try decreaseing the amount of space for window searching to further speed up the process.

Furthermore, although it is not neceesary, I may attempt decision tree in the future reduce redundancy on the feature vector and I could have used more than one scale to find the windows and  use them on the heatmap.

In classifier aspect, I may improve the classifier by additional data augmentation for instance random lightness and shadow tom increase robustness of the classifier, or even implement a deep nerual network??
