## Testing Object Detection using RealSense camera
## to create rectangle identifier around yellow highlighter
# Code borrowed from opencv_viewer_example.py, previous object detection from AMR,  
# and pyimagesearch.com
# By Jay Hanckel

import pyrealsense2 as rs
import numpy as np
import cv2

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
lower = np.array([0, 154, 154],dtype="uint8")  # Need to change for Orange I think it needs to be something like 30
upper = np.array([153, 255, 255],dtype="uint8")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Function to swap colormap

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue 

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HSV)
        mask = cv2.inRange(color_image,lower,upper)
        output = cv2.bitwise_and(color_image,color_image, mask = mask)
        # Stack both images horizontally
        images = np.hstack((color_image, output, cv2.bitwise_not(depth_colormap)))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()