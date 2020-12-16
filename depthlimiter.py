#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:40:33 2020

@author: pi
"""

## Using color and shape to improve Object Detection using RealSense camera
## to create green contour around an orange sock in motion
# Code borrowed from colorandshape.py 

### TO DO: create "average" contour around object to determine depth at various
### areas of the shape
import pyrealsense2 as rs
import numpy as np
import imutils
import cv2

# define the lower and upper boundaries of the "orange"
# sock in BGR color space, then initialize the
# list of tracked points
# yellow: lower = np.array([0, 154, 154],dtype="uint8"); upper = np.array([153, 255, 255],dtype="uint8")
# red/orange: lower = np.array([0, 0, 134],dtype="uint8"); upper = np.array([100, 150, 255],dtype="uint8") 
lower = np.array([0, 154, 154],dtype="uint8")  #yellow
upper = np.array([140, 225, 255],dtype="uint8")

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
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue 

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        #restrict depth analysis to 1.2 meters or closer
        depth_image[depth_image>1200]=4000
#        for i in depth_image:
#            for j in depth_image[i]:
#                if depth_image[i][j] > 1100:
#                    depth_image[i][j] = 4000
        
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3),cv2.COLORMAP_JET)
        #depth_colormap = cv2.bitwise_not(depth_colormap) 
        
        # METHOD 1: Remove background to decrease liklihood of false positive objects
         # Remove background - Set pixels further than clipping_distance to grey
        black_color = 40
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        #depth_image_3d = np.dstack((color_image[:][:][0],color_image[:][:][1],color_image[:][:][2]))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, color_image)

        
        # METHOD 2: Use ML algorithm (separate file) to teach foot detection  
        
        # Apply color mask and apply contouring        
        #mask = cv2.inRange(color_image,lower,upper)
        mask = cv2.inRange(bg_removed,lower,upper)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        output = cv2.bitwise_and(bg_removed,bg_removed, mask = mask)
        for c in cnts:
            cv2.drawContours(color_image,[c],-1,(0,255,0), 2)
            cv2.drawContours(depth_colormap,[c],-1,(0,255,0), 2)
            cv2.drawContours(output,[c],-1,(0,255,0), 2)
            cv2.drawContours(bg_removed,[c],-1,(0,255,0), 2)
        
        # Develop 'average' contour to define foot
        
        
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        images2 = np.hstack((output,bg_removed))
        images3 = np.vstack((images,images2))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images3)
        key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()