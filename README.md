##Advanced Vehicle Detection


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction combined with binned color features on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


###Feature Extraction

The feature extraction process is based on three methods: Histogram-Of-Gradients(HOG), Color binning, and spatial feature extraction. The functions `get_hog_features()`, `color_hist()` and `bin_spatial()` contain implementations respectively. First I started with preparing file lists of all vehicle and non-vehicle images. I then pass these lists to function `train_classifier()` which reads-in each image and performs feature extraction. For a given image; hog, spatial and color features are concatenated together to make a feature vector for that image. The following example shows hog images (each channel) for a vehicle as well as non-vehicle image from the training set. It is generated using `YCrCb` color-space and HOG parameters of `pix_per_cell=8`, `orient=9` and `cells_per_block=2`.

Vehicle: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/test_images/Vehicle.png" width="30%"> 

Hog CH1: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Vehicle_ch1.png" width="30%">

Hog CH2: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Vehicle_ch2.png" width="30%">

Hog CH3: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Vehicle_ch3.png" width="30%">

Non-Vehicle: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/test_images/Non-Vehicle.png" width="30%"> 

Hog CH1: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Non-Vehicle_ch1.png" width="30%">

Hog CH2: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Non-Vehicle_ch2.png" width="30%">

Hog CH3: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/Non-Vehicle_ch3.png" width="30%">


I explored differenct color spaces, hog features, color bins and spatial size, and settled on the following:

`
Color space = YCrCb
pix_per_cell = 8
orient = 9
cells_per_block = 2
color histogram bins = 32
spatial size  = (8,8)
`
I played around with hog features and noticed that the gradients were more chaotic (less sharpness in image) when I decreased `orient`. But for values more than 9, it deteriorated as well.

###Classication

I used `sklearn.preprocessing.StandardScaler()` to normalize the feature set that was extracted. I then used a linear classifier to classify car/non-car images. `sklearn.svm.LinearSVC` is used for this purporse. The function `train_classifier` is where this functionality is implemented.


###Sliding Window Search

I used a sliding window approach to sample the image and then look for a positive detection for each window. The region of interest that I picked was the lower 50% of the image since this is where the cars will always be. The region of interest is then sampled with a window size of 96x96, resized to 64x64 (to match the actual training image size) and then used to extract the same features as in training. The trained classifier is then used to predict the class of the sample. 

When HOG calculation is done for every window, the pipeline becomes painfuly slow. It takes an hour on my Macbook Pro to process a 50 second video. To avoid this, the HOG features are calculated on the whole image before sampling with the windows. This cuts downs the time to 15 minutes for the same video.

The following example is a representation of the sliding windows on an image. The windows correspond to positive detection.

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_sliding_windows.jpg" width="30%">

I then calculated a heatmap of the positive detections and thresholded it to remove false positives. `scipy.ndimage.measurements.label()` is then used to determine individual blobs in the heatmap and to construct bounding boxes on the image. The following images show an example of the heatmap and the final bounding boxes of the same image.

Heatmap: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_heatmap.jpg" width="30%">


Final Output <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_output_boxes.jpg" width="30%">


### Video Implementation

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

