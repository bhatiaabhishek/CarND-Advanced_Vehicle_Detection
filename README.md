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

Non-Vehicle: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/test_images/Non-vehicle.png" width="30%"> 

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

When HOG calculation is done for every window, the pipeline becomes painfuly slow. It takes an hour on my Macbook Pro to process a 50 second video. To avoid this, the HOG features are calculated on the whole image before sampling with the windows (function `find_cars1()`). This cuts downs the time to 15 minutes for the same video. To use the window size of 96x96 (found through trial), instead of increasing the window size by 1.5 (64x1.5 = 96), the image is scaled down by 1. Also, I used `cells_per_step = 2` for search, to obtain a 75% overlap between adjacent windows.

The following example is a representation of the sliding windows on an image. The windows correspond to positive detection.

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_sliding_windows.jpg" width="30%">

I then calculated a heatmap of the positive detections (`add_heat()`) and thresholded it to remove **false positives**. `scipy.ndimage.measurements.label()` is then used to determine individual blobs in the heatmap and to construct bounding boxes on the image. The following images show an example of the heatmap and the final bounding boxes of the same image.

Heatmap: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_heatmap.jpg" width="30%">


Final Output: <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Vehicle_Detection/blob/master/output_images/test1_output_boxes.jpg" width="30%">


### Video Implementation

For the video, I have also added my Advanced Lane Detection pipeline to the vehicle detection pipeline.

Here's a [link to my video result](./project_video_veh_detect.mp4)





###Discussion

Some trial and error went into finding the best color space and the window size that works. Choosing a large window size caused the cars near the edge of the frames to be un-dectected. It also led to smaller cars being un-detected. On the other hand, a very small window created the issue of false positives as well as a long runtime. If we have too many windows, the runtime adds up, especially for a 30fps video. Another gotcha, is that the features needed to be stacked the same way as were during the training phase. Another parameter to tweak was the overlap between the windows during search. Having a smaller overlap definitely helped in my case, at the cost of longer runtime.
The output boxes are wobbly but given more time, the boxes could be averaged over few frames to have a smoother output.
