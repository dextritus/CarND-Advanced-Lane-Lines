# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_both.png "Undistorted"
[image2]: ./output_images/undistorted_perspective.png "Warped"
[image3]: ./output_images/original_together_color.png "Color and gradient thresholds"
[image4]: ./output_images/binary_lane.png "Binary lanes"
[image5]: ./output_images/detected_lane.png "Projected resulting lanes"

---

### Camera Calibration

I calibrated the camera using the chessboard images provided. First, knowing the size of the chessboard (9x5), the object points were created, which are all the same. Since the object lays on a flat surface, the z coordinate is 0. I used the `findChessboardCorners` function to find the location of the chessboard points on each chessboard image. After obtaining complete object and image points vectors, I used the `calibrateCamera` method to obtain the mtx, dist(ances) variables require for the `undistortCamera` function. These values were saved using Pickle a file called `calib_data`, so that they can be reused later without the need to recompute the calibration values. An example of an undistorted image is shown below.

![alt text][image1]

### Pipeline (single images)

The pipeline consisted of the following steps, and is found in the method `pipeline`.

* Crop the image
* Convert the image to grayscale (for the edge detection) and HSL space
* Use gradient magnitude and direction thersholds, together with color threhsold. 
* Undistort, and warp the image to obtain a bird's eye view
* Identify the region of the lanes from a thresholded binary image
* Obtain second order polynomials for the lanes, calculate curvature and position on the lane
* Create a polygon from the lanes, warp back the image onto the original image and display curvature and lane position information.
* Hope for the best

First the top 300 pixels of the image are removed since they do not contain any useful information. This happens in the `crop_img` method. 

Image is converted to grayscale, and to HSL. The sobel operator is applied in both directions with a kernel size of 11. The threshold for the magnitude of the gradient is set between 60 and 150, and for the direction between pi/3 and pi/2 (between 60 and 90 degrees in absolute value). For the color thresholding, only the S channel was considered and the threshold values were between 140 and 190 (otherwise there are too many shadows visible). A binary image was created from the combined edge and color thresholds. This all happens in the `color_edges` method. An example is shown below.

![alt text][image3]

In order to obtain a bird's eye view of the road, sample points were obtained on a test image (sorce points), and mapped onto a destination image of size (1200,720). The method An example of the warped road image is given below.

![alt text][image2]

After the binary image was warped, the location of the lanes was found using convolution. Histograms of strips of the image of a certain height (`window_height`) are convoluted with a window of size `window_width`. The window for the image strips is then shifted by `dy`, in order to look for lane locations in the next layer. Histogram is used to identify the lane locations initially (from the bottom quarter of the image), and then those identified locations are used as starting points for the lane locations in the next layers. The margin for looking left and right is given by the variable `margin`. The lane locations are computed in the `window_centroids` method, and only the locations for which the convolution value is higher than `min_pix` are registered, otherwise the value is set to NaN. However, in order to keep a reference for the lane location in the next level, the value from the previous layer is kept.

For interpolation, only the location of the lanes that are not NaN are used for fitting a second order polynomial, resulting in a, b, and c coefficients. These coefficients are then averaged with the values in the last `n_prev` frames, in order to obtain a smoother estimate in case the interpolation does not yield a good result. The method is given in `interpolate_lanes`. An example of the interpolate lanes from a binary image is given below. Instead of using the mean, I tried to use a histogram of the interpolation coefficients and pick the one that appears the most in the 'n_prev' frames, however this method doesn't work as nicely due to the difficulty in choosing the number of histogram bins, which is sensitive to outliers. I tried this in the method `best_coeff`.

![alt text][image4]

Finally the radius of curvature is calculate using the a and b coefficients from the interpolation, and the location on the lane (set to middle, at 360 pixels). The estimate for the true radius of curvature is calculated from a 27 meters lane length (a bit shorter than the example in the tutorial) and 3.7 meters lane width. The smallest radius of curvature from the left and right lanes is used for the road curvature estimate, since for some frames the lanes appear straight on one side of the road and curved on the other. The radius of curvature is calculated in the `radius_of_curvature` method. 

The position on the lane is calculated from the 'destination' points used for warping. The distance between the lanes at y = 720 is added to the position of the left lane at y = 720, from which the "true center" is substracted (difference between the right and left lane positions from the warping destination points). Text was added to the image using the `putText` method, after formatting the numbers to a fixed size. The final result is shown below.


![alt text][image5]


### Discussion

The pipeline works well on the project video. There are some issues on the regions where there are a lot of shadows, which mostly affect the lane curvature, but not the general area of the lanes. This could be fixed by calculating the distance between the lanes from good previous frames. Then, in image slices where the lane distance is unusually large or small, this lane distance can be used to calculate a virtual location of the lane. For example, it the left lane is very visible but the right lane is impossible to detect, we could use the curvature of the left lane and the lane width information from the previous frames to estimate the location of the right lane. The final video can be found in `project_video_output.mp4`.


