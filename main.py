import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

###############
# settings
###############

crop_y = 300 #where to crop the image from
window_width = 50	# sliding window width for convolution
window_height = 100 # window slice height for convolution
margin = 40 #+- to the right and the left, in case the lane changes orientation
dy = 80 # by how much to move the sliding window

n_prev = 30
prev_window_left = np.array([]).reshape(0, 3)
prev_window_right = np.array([]).reshape(0, 3)
prev_l_radius = np.array([])
prev_r_radius = np.array([])

images = glob.glob('./camera_cal/calibration*.jpg')

######################################
### generate and save calibration data
nx = 9
ny = 6

objpts = []
imgpts = []

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


for img_file in images:
	### calibrate the camera in this section
	img = mpimg.imread(img_file)

	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	if ret == True:
		imgpts.append(corners)
		objpts.append(objp)

calib_data = {}
calib_file = "calib_data"
calib_file = open(calib_file, 'wb')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (1280, 720), None, None)

calib_data["mtx"] = mtx
calib_data["dist"] = dist
pickle.dump(calib_data, calib_file)
calib_file.close()

##############################################
# load the pickled camera data and undistort
##############################################

def crop_img(img, crop_y):
	return img[crop_y:, :, :]

def undistort_camera(img):
	calib_file = "calib_data"
	calib_file = open(calib_file, 'rb')
	calib_data = pickle.load(calib_file)
	
	dist = calib_data['dist']
	mtx = calib_data['mtx']

	return cv2.undistort(img, mtx, dist, None, mtx)

#############################
# transform the perspective
#############################
def warp_image(img, crop_y):
	#points on the real road
	src = np.float32([
					[285, 661 - crop_y],
					[586, 455 -  crop_y],
					[698, 455 - crop_y],
					[1015, 661 - crop_y]
					]);

	#points on the undistorted image
	dst = np.float32([
					[260, 720],
					[260, 0],  #100
					[900, 0], #100 #1062
					[900, 720]
					]);
	M = cv2.getPerspectiveTransform(src, dst)
	return cv2.warpPerspective(img, M, (1280, 720), flags = cv2.INTER_LINEAR)

###########################################
# apply color filter, gradient thresholds
###########################################
def color_edges(image):

	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	### color threshold
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

	s_ch = hls[:,:,2]
	min_s = 140 #150, 190
	max_s = 190

	s_binary = np.zeros_like(s_ch)
	s_binary[(s_ch >= min_s) & (s_ch <=max_s)] = 1

	## gradient mag and direction threshold
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 7)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 7)
	abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

	tot_sobel = np.sqrt(sobelx**2 + sobely**2) # + sobely**2)

	tot_sobel = tot_sobel/np.max(tot_sobel)*255.0
	tot_sobel = tot_sobel.astype(np.uint8)

	sobel_min = 60
	sobel_max = 150

	binary_sobel = np.zeros_like(tot_sobel)
	binary_sobel[(tot_sobel > sobel_min) & (tot_sobel<sobel_max) & (abs_grad_dir >= np.pi/3) & (abs_grad_dir<=np.pi/2)] = 1

	together_binary = np.zeros_like(tot_sobel)
	together_binary[(binary_sobel == 1) | (s_binary == 1)] = 1

	return together_binary


################################
#moving window to identify lanes
################################
def find_window_centroids(image, window_width, window_height, margin):
	min_pix = 50

	window_centroids = [] # for each level
	window_app = [] # for each level

	window = np.ones(window_width)

	img_y = image.shape[0]
	img_x = image.shape[1]

	l_sum = np.sum(image[int(3*img_y/4):,180:350], axis = 0)
	#find peak, and shift with half of window width to find the center of the lane
	l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2 + 180
	# print("initial l center", l_center)
	# print("where to look for right", int(img_x/2))
	r_sum = np.sum(image[int(3*img_y/4):,900:1100], axis = 0)
	r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + 900
	# print("initial r center", r_center)

	window_centroids.append((l_center, r_center))
	window_app.append((l_center, r_center))

	#why doing the above separately? because you don't look in the whole row, just around what you found so far
	for level in range(1, int((img_y - window_height)/dy)+1):

		image_layer = np.sum(image[int(img_y-((level)*dy+window_height)):int(img_y-level*dy),:], axis=0)
		conv_signal = np.convolve(window, image_layer) #shouldn't we convolve only in the regions of interest?

		offset = window_width/2
		l_min_index = int(max(l_center + offset - margin, 0))
		l_max_index = int(min(l_center + offset + margin, img_x))
		max_l = np.max(conv_signal[l_min_index:l_max_index])
		if (max_l > min_pix):
			l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
			l_center_app = l_center

		else: 
			l_center_app = np.nan
		# print(max_l)
		# print(l_center)

		r_min_index = int(max(r_center + offset - margin, 0))
		r_max_index = int(min(r_center + offset + margin, img_x))
		max_r = np.max(conv_signal[r_min_index:r_max_index])
		if (max_r > min_pix):
			r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
			r_center_app = r_center
		else:
			r_center_app = np.nan
		# print(r_center)
		# print(max_r)


		window_centroids.append((l_center, r_center))
		window_app.append((l_center_app, r_center_app))
		# print(r_center - l_center)

	return window_centroids, window_app

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-((level)*dy+height)):int(img_ref.shape[0]-level*dy),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

####################################
#draw the boxes for the lane region
####################################
def draw_lane_region(window_centroids, warped):
	if len(window_centroids) > 0:
		l_points = np.zeros_like(warped)
		r_points = np.zeros_like(warped)

		for level in range(0, len(window_centroids)):
			l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
			r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

			l_points[((l_mask == 1) ) ] = 255
			r_points[ ((r_mask == 1) ) ] = 255

		#draw the results
		template = np.array(l_points + r_points, np.uint8)
		zero_channel = np.zeros_like(template)
		template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) #make the mask green!
		warpage = np.dstack((warped, warped, warped))*255 #make image 3 channels
		output = cv2.addWeighted(warpage, 1.0, template, 0.5, 0.0)

	else:
		output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

	return output

################################################
# get the most frequent interpolation coefficien
# for left and right lanes in the previous frames 
################################################
def best_coeff(previous_window):
	#a
	hst = np.histogram(previous_window[:, 0], bins = 5)
	max_hist = np.argmax(hst[0])
	a = (hst[1][max_hist] + hst[1][max_hist + 1])/2.0
	#b
	hst = np.histogram(previous_window[:, 1], bins = 5)
	max_hist = np.argmax(hst[0])
	b = (hst[1][max_hist] + hst[1][max_hist + 1])/2.0
	#c
	hst = np.histogram(previous_window[:, 2], bins = 5)
	max_hist = np.argmax(hst[0])
	c = (hst[1][max_hist] + hst[1][max_hist + 1])/2.0

	return np.array([a, b, c])

def best_radius(prev_radii):
	#a
	hst = np.histogram(prev_radii)
	max_hist = np.argmax(hst[0])
	r = (hst[1][max_hist] + hst[1][max_hist + 1])/2.0

	return r

################################################
# interpolate the lanes with 2nd order polynomial
################################################
def interpolate_lanes(window_centroids):

	global prev_window_left, prev_window_right

	ploty = np.linspace(0, 720, int((720-window_height)/dy)+1)
	ploty = ploty[::-1]
	leftx = np.array([point[0] for point in window_centroids])
	rightx = np.array([point[1] for point in window_centroids])

	good_l_idx = np.isfinite(leftx)
	good_r_idx = np.isfinite(rightx)

	left_fit = np.polyfit(ploty[good_l_idx], leftx[good_l_idx], 2)
	right_fit = np.polyfit(ploty[good_r_idx], rightx[good_r_idx],  2)

	prev_window_left = np.vstack([prev_window_left, left_fit])
	prev_window_left = prev_window_left[-n_prev:]

	prev_window_right = np.vstack([prev_window_right, right_fit])
	prev_window_right = prev_window_right[-n_prev:]

	# left_fit = best_coeff(prev_window_left)
	# right_fit = best_coeff(prev_window_right)

	left_fit[0] = np.mean(prev_window_left[:,0])
	left_fit[1] = np.mean(prev_window_left[:,1])
	left_fit[2] = np.mean(prev_window_left[:,2])

	right_fit[0] = np.mean(prev_window_right[:,0])
	right_fit[1] = np.mean(prev_window_right[:,1])
	right_fit[2] = np.mean(prev_window_right[:,2])

	#for a smoother polygon
	ploty2 = np.linspace(0, 720, 20)
	ploty2 = ploty2[::-1]

	left_fitx = left_fit[0]*ploty2**2 + left_fit[1]*ploty2 + left_fit[2]
	right_fitx = right_fit[0]*ploty2**2 + right_fit[1]*ploty2 + right_fit[2]

	return ploty2, left_fitx, right_fitx

#############################
#calculate radii of curvature
#############################
def radius_of_curvature(ploty, left_fitx, right_fitx):
	global prev_l_radius, prev_r_radius

	ym_per_pix = 27/720 #warped image is 720 pixels tall
	xm_per_pix = 3.7/(900-260) #distance between lanes is 640 pixels

	#new interpolated values (Could do this from the beginning)
	left_fit_sc = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_sc = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

	y_eval = 360

	left_R = ((1 + (2*left_fit_sc[0]*y_eval*ym_per_pix + left_fit_sc[1])**2)**1.5)/2/np.absolute(left_fit_sc[0])
	right_R = ((1 + (2*right_fit_sc[0]*y_eval*ym_per_pix + right_fit_sc[1])**2)**1.5)/2/np.absolute(right_fit_sc[0])

	left_R = np.floor(left_R)
	right_R = np.floor(right_R)

	return left_R, right_R


######################################
#display the lane and road information
######################################
def display_lane_info(orig_image, plty, left_fitx, right_fitx):
	#now, make a polygon back to be drawn back onto the original image
	warp_zero = np.zeros_like(orig_image[:,:,1]).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	pts_left = np.array([np.transpose(np.vstack([left_fitx, plty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plty])))])
	pts = np.hstack((pts_left, pts_right))

	#radius of curvature 

	lr, rr = radius_of_curvature(plty, left_fitx, right_fitx)

	#points on original image
	src = np.float32([
					[285, 661],
					[586, 455],
					[698, 455],
					[1015, 661]
					]);

	#points on the undistorted image
	dst = np.float32([
					[260, 720],
					[260, 0],  #100
					[900, 0], #100 #1062
					[900, 720]
					]);

	Minv = cv2.getPerspectiveTransform(dst, src)

	cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
	newarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0]))
	result = cv2.addWeighted(orig_image, 1, newarp, 0.9, 0)
	# calculate the smallest radius
	radius = "{:4.0f}".format(np.min([lr, rr]))
	radius = 'Approximate radius ' + radius + ' m'
	xm_per_pix = 3.7/(900-260)

	#calculate the position on lane
	lane_pos = int((right_fitx[0] - left_fitx[0])/2 + left_fitx[0] - 580)*xm_per_pix
	lane_pos = "{:1.3f}".format(lane_pos)
	lane_pos = "lane position " + lane_pos + " m"

	#write the info to the image
	cv2.putText(result, radius, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.putText(result, lane_pos, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	return result

#######################
# this is the pipeline
#######################

def pipeline(img):
	#1. undistort, crop the image
	img_undistorted = undistort_camera(img)
	img_cropped = crop_img(img_undistorted, crop_y)
	#2. color and gradient thersholds
	thresholded = color_edges(img_cropped)
	#3. warp
	# thresholded = undistort_camera(thresholded)
	thresholded = warp_image(thresholded, crop_y)
	#4. find lanes
	window_centroids, wapp = find_window_centroids(thresholded, window_width, window_height, margin)
	#5. obtain polynomial
	#! add check too see if it is drastically different than previos one
	ploty, left_fitx, right_fitx = interpolate_lanes(wapp)
	# lr, rr = radius_of_curvature(ploty, left_fitx, right_fitx)
	#6. draw everything on original image
	result = display_lane_info(img_undistorted, ploty, left_fitx, right_fitx)
	return result

# test_file = './test_images/straight_lines1.jpg'
# img = mpimg.imread(test_file)
# undist = undistort_camera(img)
# undist_cropped = crop_img(undist, crop_y)
# cropped_img = crop_img(img, crop_y)
# undist_cropped = undistort_camera(cropped_img)

# plt.figure()
# plt.subplot(121)
# plt.imshow(undist, origin = 'upper')
# plt.plot(285, 661, 'r.')
# plt.plot(586, 455, 'r.')
# plt.plot(698, 455, 'r.')
# plt.plot(1015, 661, 'r.')
# plt.title('undistorted')
# plt.subplot(122)
# plt.imshow(warp_image(undist, 0),  origin = 'upper')
# plt.title('undistorted, warped')
# plt.savefig('./output_images/undistorted_perspective.png')
# plt.show()

# #process the movie
# from moviepy.editor import VideoFileClip

# fname = "project_video"
# output = fname + "_output.mp4"
# input_file = VideoFileClip(fname + ".mp4")
# cut_file = input_file.subclip(35, 44)
# cut_file = input_file.get_frame(30)
# processed_file = cut_file.fl_image(pipeline)
# processed_file.write_videofile(output, audio = False)
# result = pipeline(cut_file)

# plt.figure()
# plt.imshow(result, origin = 'upper')
# plt.savefig('./output_images/detected_lane.png')
# plt.show()
