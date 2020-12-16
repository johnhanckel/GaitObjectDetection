#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:10:55 2020

@author: Jay Hanckel
"""

## Using color and shape to improve Object Detection using RealSense camera
## to create green contour around an orange sock in motion
# Code borrowed from opencv_viewer_example.py, previous balltrackercompressed2 from AMR,  
# and pyimagesearch.com


import pyrealsense2 as rs
import numpy as np
import imutils
import cv2

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# yellow: lower = np.array([0, 154, 154],dtype="uint8"); upper = np.array([153, 255, 255],dtype="uint8")
# red/orange: lower = np.array([0, 0, 134],dtype="uint8"); upper = np.array([100, 150, 255],dtype="uint8") 
lower = np.array([0, 154, 154],dtype="uint8")  #yellow
upper = np.array([140, 225, 255],dtype="uint8")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Align depth and color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
pipeline.start(config)

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

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HSV)
        depth_colormap = cv2.bitwise_not(depth_colormap)
        
        # Apply color mask and apply contouring        
        mask = cv2.inRange(color_image,lower,upper)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        output = cv2.bitwise_and(color_image,color_image, mask = mask)
        for c in cnts:
            cv2.drawContours(color_image,[c],-1,(0,255,0), 2)
            cv2.drawContours(depth_colormap,[c],-1,(0,255,0), 2)
            cv2.drawContours(output,[c],-1,(0,255,0), 2)
        
        # Stack both images horizontally
        images = np.hstack((output, color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()