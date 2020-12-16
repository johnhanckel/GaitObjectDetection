#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:52:54 2020

@author: pi
"""

## Using improve OD by averaging object contour, find centroid, and
## total length of object
# Code borrowed from depthlimiter.py 

### TO DO: create "average" contour around object to determine depth at various
### areas of the shape
import pyrealsense2 as rs
import numpy as np
import imutils
import cv2

# define the lower and upper boundaries of the "yellow"
# sock in HSV color space, then initialize the
# list of tracked points

# Try converting to HSV for yellow
lower = np.array([15, 10, 80],dtype="uint8")  #yellow
upper = np.array([40, 255, 255],dtype="uint8")
zcoord = [] #depth array 

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

# Align depth and color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale, "meters/mm")

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
##### CHECK DEPTH SCALE IN TERMS OF COLORMAP
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

try:
    while True:

        # Wait for a coherent pair of frames: depth and color & align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue 
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert color_image to hsv then back to 8bit
        bg_removed = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
        
        #restrict depth analysis to 1.2 meters or closer
        ### NEEDS ADJUSTMENT
        depth_image[depth_image>1200]=4000

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3),cv2.COLORMAP_HOT)
        
        # METHOD 1: Remove background to decrease liklihood of false positive objects
        # Remove background - Set pixels further than clipping_distance to black
        black_color = 40
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, bg_removed)
       
        # Apply color mask and apply contouring        
        #mask = cv2.inRange(color_image,lower,upper)
        mask = cv2.inRange(bg_removed,lower,upper)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        #output = cv2.bitwise_and(bg_removed,bg_removed, mask = mask)
        cnts = imutils.grab_contours(cnts)
        if len(cnts)>0:
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            #cv2.drawContours(color_image,cnts,0,(0,255,0), 2)
                #Create a mask image that contains the contour filled in
            cimg = np.zeros_like(depth_colormap)
            cv2.drawContours(cimg,cnts,0,(0,255,0), -1)
            cv2.drawContours(depth_colormap,cnts,0,(0,255,0), -1)
            #cv2.drawContours(output,cnts,0,(0,255,0), 2)
            cv2.drawContours(bg_removed,cnts,0,(60,255,255), 2)
            c = max(cnts, key=cv2.contourArea)
            
            # Test code for reading depth inside contour
            pts = np.where(cimg == 255)    #index 0 and 1 on var explorer are x &     y coordinates respectively. Only saves last frame for viewing
            zcoord.append(depth_image[pts[0],pts[1]])   #creates 1D array of depth data for each corresponding x,y coordinates. Saves all frames for viewing
                
                    
        # Find largest contour to isolate foot and centroid
            
            M = cv2.moments(c)
            if M["m00"]!=0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                #cv2.drawMarker(color_image,center,(0,0,0),cv2.MARKER_TILTED_CROSS, 8, 1)
                cv2.drawMarker(depth_colormap,center,(0,0,0), cv2.MARKER_TILTED_CROSS, 8, 1)
                cv2.drawMarker(bg_removed,center,(0,0,0),cv2.MARKER_TILTED_CROSS, 8, 1)
        bg_removed = cv2.cvtColor(bg_removed,cv2.COLOR_HSV2BGR)  
        # Use color_image & depth_colormap  
        images = np.hstack((bg_removed,depth_colormap))
        #images2 = np.hstack((output,bg_removed))
        #images3 = np.vstack((images,images2))
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        k = cv2.waitKey(1) & 0xFF

    # if the 'q' key or spacebar is pressed, stop the loop
        if k == ord("q") or k == ord("Q") or k == ord(" "):
            break

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()