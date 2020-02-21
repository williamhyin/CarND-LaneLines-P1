# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

**Step 1: Import the package and load the test image**

    #importing some useful packages
	import matplotlib.pyplot as plt # read the image as RGB
	import matplotlib.image as mpimg
	import numpy as np
	import cv2 # read the image as GBR
	%matplotlib inline
	# Import everything needed to edit/save/watch video clips
	from moviepy.editor import VideoFileClip
	from IPython.display import HTML
	#reading in an image
	image = mpimg.imread('test_images/solidWhiteRight.jpg')

**Step 2: Build a color filter to find the white and yellow lane lines**

Road lines are generally only available in white and yellow.To enhance robustness, we use both HSV and HSL color spaces to find yellow and white lines.


	def filter_colors(image):
	    # Filter the white and yellow lines in images.
	    # To reinforce the performance, we use both hls and hsv to find the yellow and white lines.
	    
	    # white BGR
	    lower_white = np.array([200, 200, 200])
	    upper_white = np.array([255, 255, 255])
	    white_mask = cv2.inRange(image, lower_white, upper_white)
	    white_RGB_image = cv2.bitwise_and(image, image, mask=white_mask)
	
	    # yellow HSV
	    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	    lower_yellow = np.array([90,100,100])
	    upper_yellow = np.array([110,255,255])
	    
	    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	    yellow_hsv_image = cv2.bitwise_and(image, image, mask=yellow_mask)
	
	    hsv_image = weighted_img(white_RGB_image, 1., yellow_hsv_image, 1., 0.)
	    
	    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	    
	    # white HLS
	    lower = np.uint8([  0, 200,   0])
	    upper = np.uint8([255, 255, 255])
	    
	    white_hls_mask = cv2.inRange(hls, lower, upper)
	    white_hls_image = cv2.bitwise_and(image, image, mask=white_hls_mask)
	
	    # yellow HLS
	    lower = np.uint8([ 10,   0, 100])
	    upper = np.uint8([ 40, 255, 255])
	    yellow_hls_mask = cv2.inRange(hls, lower, upper)
	    yellow_hls_image = cv2.bitwise_and(image, image, mask=yellow_hls_mask)
	
	    hls_image = weighted_img(white_hls_image, 1., yellow_hls_image, 1., 0.)
	
	    final_image = weighted_img(hls_image, 1., hsv_image, 1., 0.)
	    
    return final_image

	filtered_image = filter_colors(image)
	plt.figure()
	plt.imshow(filtered_image)

![](https://s2.ax1x.com/2019/09/04/nENqnU.png)

**Step 3: Convert color filtered image to gray image and smooth image (line_detect(image))**

	gray = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
    kernel_size = 3
    blur_gray = gaussian_blur(gray,kernel_size)

**Step 4: Canny edges detection(line_detect(image))**
	
	low_threshold = 50
    high_threshold = 180
    edges = canny(blur_gray,low_threshold,high_threshold)
![](https://s2.ax1x.com/2019/09/04/nE00zR.png)

**Step 5: Define the Region of ​​interest**

	def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
	    #defining a blank mask to start with
	    mask = np.zeros_like(img)   
	    
	    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	    if len(img.shape) > 2:
	        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
	        ignore_mask_color = (255,) * channel_count
	    else:
	        ignore_mask_color = 255
	        
	    #filling pixels inside the polygon defined by "vertices" with the fill color    
	    cv2.fillPoly(mask, vertices, ignore_mask_color)
	    
	    #returning the image only where mask pixels are nonzero
	    masked_image = cv2.bitwise_and(img, mask)
	    return masked_image

	imshape = image.shape
    vertices = np.array([[(imshape[1]*0.15,imshape[0]*0.95),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1]*0.9,imshape[0]*0.95)]], dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices)

![](https://s2.ax1x.com/2019/09/04/nEB1te.png)


**Step 6: Generate line by HoughLinesP function**
	
	def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
	    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	    draw_lines(line_img, lines)
	    return line_img

	rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50 #minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
  
    line_image = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
    
    result = weighted_img(line_image,0.8,image, 1.)

**Step 7: Find the left and right lane line and generate the start and end point of lane line**

* Calculate the start and end points of lane lines
		
		def get_endpoints(ymin,ymax,slope,intercept):
    		xmin=int((ymin-intercept)/slope)
    		xmax=int((ymax-intercept)/slope)
    		return xmin,xmax

* Distinguish the left and right lane lines according to the slope and the location of the line segment.


		def draw_lines(img, lines, color=[255, 0, 0],thickness=10):
	    """
	    NOTE: this is the function you might want to use as a starting point once you want to 
	    average/extrapolate the line segments you detect to map out the full
	    extent of the lane (going from the result shown in raw-lines-example.mp4
	    to that shown in P1_example.mp4).  
	    
	    Think about things like separating line segments by their 
	    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	    line vs. the right line.  Then, you can average the position of each of 
	    the lines and extrapolate to the top and bottom of the lane.
	    
	    This function draws `lines` with `color` and `thickness`.    
	    Lines are drawn on the image inplace (mutates the image).
	    If you want to make the lines semi-transparent, think about combining
	    this function with the weighted_img() function below
	    """
	
			# build lane lines slopes, intercepts and points cache
			# Then average them for every ten frames, in order to smooth the variation of lane lines
		    positive_slope_intercept=[]
		    negative_slope_intercept=[]
		    positive_lines=[]
		    negative_lines=[]
			
	 		# calculate half of the size of image in x-axis
		    size1=img.shape[1]/2 
		
		    
		    for line in lines:
		        for x1,y1,x2,y2 in line:
					# calculate the slope of lane line.
		            slope=np.float((y2-y1))/np.float((x2-x1))
					# tan 20' = 0.26 tan 80' = 5.67
					# positive line(left lane line)
		            if slope > 0.26 and slope<5.67 and x1>size1 and x2>size1 :
		
		                positive_slope_intercept.append([slope, y1-slope*x1])
		                positive_lines.append([x1,y1,x2,y2])
		            elif slope<-0.26 and slope>-5.67 and x1<size1 and x2<size1 :
					# negative line(right lane line)
		                negative_slope_intercept.append([slope,y1-slope*x1])
		                negative_lines.append([x1,y1,x2,y2])
		                
		    slope_p, intercept_p=filter_slope_intercept(positive_lines,positive_slope_intercept)
		    slope_n, intercept_n=filter_slope_intercept(negative_lines,negative_slope_intercept)

		    # In the caches, the image of the latest frame will be placed first, and every ten frames will discard the last picture.
		    if slope_p and intercept_p:
		        if len(slope_ps)<10:
		            slope_ps.insert(0,slope_p)
		        else:
		            slope_ps.pop(-1)
		            slope_ps.insert(0,slope_p)
		            
		        if len(intercept_ps)<10:
		            intercept_ps.insert(0,intercept_p)
		        else:
		            intercept_ps.pop(-1)
		            intercept_ps.insert(0,intercept_p)
		            
		    if slope_n and intercept_n:
		        if len(slope_ns)<10:
		            slope_ns.insert(0,slope_n)
		        else:
		            slope_ns.pop(-1)
		            slope_ns.insert(0,slope_n)
		            
		        if len(intercept_ns)<10:
		            intercept_ns.insert(0,intercept_n)
		        else:
		            intercept_ns.pop(-1)
		            intercept_ns.insert(0,intercept_n)
		    
			# Average slopes and intercepts every ten frames.         
		    s_p=np.mean(slope_ps)
		    i_p=np.mean(intercept_ps)
		    s_n=np.mean(slope_ns)
		    i_n=np.mean(intercept_ns)
		    
		    # Upper and lower limits of the region of interest.
		    ymin=int(0.6*img.shape[0])
		    ymax=img.shape[0]
		 	

		    try:
		        xmin_p,xmax_p=get_endpoints(ymin,ymax,s_p,i_p)
		        xmin_n,xmax_n=get_endpoints(ymin,ymax,s_n,i_n)
		        cv2.line(img, (xmin_p, ymin), (xmax_p, ymax), color, thickness)
		        cv2.line(img, (xmin_n, ymin), (xmax_n, ymax), color, thickness)
		        
				# Build caches of start and end points for road lines（size=3）
		        xmin_p_back.append(xmin_p)
		        xmax_p_back.append(xmax_p)
		        xmin_n_back.append(xmin_n)
		        xmax_n_back.append(xmax_n)

		        if len(xmin_p_back)>3:
		            xmin_p_back.pop(0)
		            xmax_p_back.pop(0)
		            xmin_n_back.pop(0)
		            xmax_n_back.pop(0)
		    # If the road line is not recognized in current frame, the result of the previous frame will be used.        
		    except Exception as e:
		        cv2.line(img, (xmin_p_back[-1], ymin), (xmax_p_back[-1], ymax), color, thickness)
		        cv2.line(img, (xmin_n_back[-1], ymin), (xmax_n_back[-1], ymax), color, thickness)

* Filter slopes and intercepts of lane line
	
	Calculate the mean and variance of all the lines in the image, and filter out the lines above or below the three-time standard deviations of the average. If there is no qualifying lines, calculate the distance of all lines, using the slope and intercept of the longest line.

			def filter_slope_intercept(lines,slope_intercepts):            
				legal_slope=[]
				legal_intercept=[]
				distance_max=0
				cache=[]
				slopes=[slo_int[0] for slo_int in slope_intercepts]
				slope_mean=np.mean(slopes)
				slope_std=np.std(slopes)
				
				# Calculate the means and standard deviations of all the lines

				for slo_int in slope_intercepts:
					if slo_int[0]-slope_mean<3*slope_std:
				    	legal_slope.append(slo_int[0])
				    	legal_intercept.append(slo_int[1])

				# Find the longest line of all lines in the image
    
				if not legal_slope:
					for line in lines:
					    x1=line[0]
					    y1=line[1]
					    x2=line[2]
					    y2=line[3]
					    v1=np.array([x1,y1])
					    v2=np.array([x2,y2])
					    op=np.linalg.norm(v2-v1)
					    if op>distance_max:
					        distance_max=op
					        cache=[x1,y1,x2,y2]
					
					        legal_slope=np.float((cache[3]-cache[1]))/np.float((cache[2]-cache[0]))        
					        legal_intercept=cache[1]-legal_slope*cache[0]       
				#     if not legal_slope:                                 
				#         legal_slope = slopes
				#         legal_intercept = [slo_int[1] for slo_int in slope_intercepts]
			
				slope=np.mean(legal_slope)
				intercept=np.mean(legal_intercept)
				
				return slope, intercept

**Step 8: Process the video image and draw line**

	
	def line_detect(image):
		
		filtered_image=filter_colors(image)
	    gray = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
	    kernel_size = 3
	    blur_gray = gaussian_blur(gray,kernel_size)
	    
	    low_threshold = 50
	    high_threshold = 180
	    edges = canny(blur_gray,low_threshold,high_threshold)
	 
	 
	    imshape = image.shape
	    vertices = np.array([[(imshape[1]*0.15,imshape[0]*0.95),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1]*0.9,imshape[0]*0.95)]], dtype=np.int32)
	    masked_edges = region_of_interest(edges,vertices)
	  
	    rho = 1 # distance resolution in pixels of the Hough grid
	    theta = np.pi/180 # angular resolution in radians of the Hough grid
	    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
	    min_line_length = 50 #minimum number of pixels making up a line
	    max_line_gap = 100  # maximum gap in pixels between connectable line segments
	
	    line_image = hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
	    
	    result = weighted_img(line_image,0.8,image, 1.)
    
    	return edges,masked_edges,result

	def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    	edges,masked_edges,result = line_detect(image)
    
    	return result


	white_output = 'test_videos_output/solidWhiteRight.mp4'
	## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
	## To do so add .subclip(start_second,end_second) to the end of the line below
	## Where start_second and end_second are integer values representing the start and end of the subclip
	## You may also uncomment the following line for a subclip of the first 5 seconds
	##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
	clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
	# Initialize and build caches
	slope_ps=[]
	slope_ns=[]
	intercept_ps=[]
	intercept_ns=[]
	xmin_p_back=[]
	xmax_p_back=[]
	xmin_n_back=[]
	xmax_n_back=[]

	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	%time white_clip.write_videofile(white_output, audio=False)

![](https://s2.ax1x.com/2019/09/04/nErPG4.png)


### 2. Identify potential shortcomings with your current pipeline

Potential shortcomings:

In the challenge.mp4, when the vehicle enters the light from the shadow or enters the shadow from the light, the angle of the generated road line will change greatly at that moment, and the reliable result cannot be obtained.

I solved this problem by caching the valid results of the previous frame, but how to always provide stable results? 


### 3. Suggest possible improvements to your pipeline

I think the main improvement points are using better color threshold for road lane color detection and more efficient edge detection.

In addition, we can use deep learning to improve the accuracy of traffic line identification.

At last, for some functions, maybe we still need try more times to get better parameters.
