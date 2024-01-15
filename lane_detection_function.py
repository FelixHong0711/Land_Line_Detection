import streamlit as st
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
import concurrent.futures

# Function to download local files to user machine
def createDownloadLink(folder_path, file_extension):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'rb') as file_content:
            st.download_button(
                label=f'Download {file}',
                data=file_content.read(),
                file_name=file
            )

# Function to get vertices of the region of interest
def getVertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.15, rows]
    top_left     = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

# Function for Canny edge detection
def Canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Function to identify the region of interest
def identifyRegionInterest(image):
    polygons = getVertices(image)
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to transform color threshold
def transformColorThreshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    return result

# Function to find the equation of a line
def findLineEquation(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1 + 1e-5)
    intercept = y1 - slope * x1
    return slope, intercept

# Function to create an invisible line
def createInviLine(image, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    
    slope, intercept = findLineEquation(x1, y1, x2, y2)
    
    y1_inv = int(image.shape[0] * 0.75)
    x1_inv = int((y1_inv - intercept) / slope)

    y2_inv = image.shape[0]  # The y-coordinate of the bottom of the image
    x2_inv = int((y2_inv - intercept) / slope)
    
    invisible_line = [(x1_inv, y1_inv), (x2_inv, y2_inv)]
    return invisible_line

# Function to detect lane lines
def detectLaneLines(image):
    global last_invisible_left_line, last_invisible_right_line
    left_line = []
    right_line = []
    color_threshed = transformColorThreshold(image)
    cropped_img = identifyRegionInterest(color_threshed)
    edges = Canny(cropped_img)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, np.array([]), minLineLength=10, maxLineGap=5)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if lines is not None and len(lines) > 0: 
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-5)
                if abs(slope) > 0.5:
                    if slope < 0:
                        left_line.append((x1, y1))
                        left_line.append((x2, y2))
                    else:
                        right_line.append((x1, y1))
                        right_line.append((x2, y2))
        combo_img = cv2.addWeighted(image, 0.8, line_img, 1, 0)
        
    try:
        # Create invisible lines
        invisible_left_line = createInviLine(image, left_line)
        invisible_right_line = createInviLine(image, right_line)

        # Update the last successfully detected invisible lines
        last_invisible_left_line = invisible_left_line
        last_invisible_right_line = invisible_right_line
            
        # Draw the polygon between the invisible lines with light blue color and fade out
        if last_invisible_left_line is not None and last_invisible_right_line is not None:
            fade_combo_img = np.copy(image)
            cv2.fillPoly(fade_combo_img, [np.array([last_invisible_left_line[0], last_invisible_right_line[0], 
                                                     last_invisible_right_line[1], last_invisible_left_line[1]])],
                         color=(173, 216, 230))            
    
            cv2.polylines(fade_combo_img, [np.array(left_line)], isClosed=False, color=(255, 0, 0), thickness=10)
            cv2.polylines(fade_combo_img, [np.array(right_line)], isClosed=False, color=(255, 0, 0), thickness=10)
            combo_img = cv2.addWeighted(combo_img, 0.6, fade_combo_img, 0.4, 0)
        return combo_img
    except Exception:
        # Use the coordinates from the last successfully detected lines
        if last_invisible_left_line is not None and last_invisible_right_line is not None:
            fade_combo_img = np.copy(image)
            cv2.fillPoly(fade_combo_img, [np.array([last_invisible_left_line[0], last_invisible_right_line[0], 
                                                     last_invisible_right_line[1], last_invisible_left_line[1]])],
                         color=(173, 216, 230))
            cv2.polylines(fade_combo_img, [np.array(left_line)], isClosed=False, color=(255, 0, 0), thickness=10)
            cv2.polylines(fade_combo_img, [np.array(right_line)], isClosed=False, color=(255, 0, 0), thickness=10)
            combo_img = cv2.addWeighted(combo_img, 0.6, fade_combo_img, 0.4, 0)
            return combo_img
        else:
            return image
        
# Add this function to your code
def processBatch(video_path, start_time, end_time):
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    processed_clip = clip.fl_image(detectLaneLines)
    return processed_clip

# Modify the divideBatchProcess function
def divideBatchProcess(original_clip_path, batch_duration):
    # Get the total duration of the original video
    total_duration = VideoFileClip(original_clip_path).duration

    # Check if the video duration is valid
    if total_duration <= 0:
        st.error("Invalid video duration. Please upload a valid video.")
        return

    # Initialize start and end times for video processing batches
    start_time = 0
    end_time = batch_duration
    processed_clips = []

    # Use ThreadPoolExecutor for parallel video processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Process video in batches until the end of the original video
        while end_time <= total_duration:
            futures.append(executor.submit(processBatch, original_clip_path, start_time, end_time))
            start_time = end_time
            end_time += batch_duration

        # If the last batch extends beyond the total duration, process the remaining part
        if end_time > total_duration:
            futures.append(executor.submit(processBatch, original_clip_path, start_time, total_duration))

        # Wait for the threads to complete and collect the processed clips
        for future in concurrent.futures.as_completed(futures):
            processed_clips.append(future.result())

    # Check if any valid batches were processed
    if not processed_clips:
        st.error("No valid batches processed. Please check your video and batch duration.")
        return

    # Concatenate the processed clips into a consolidated video
    consolidated_clip = concatenate_videoclips(processed_clips)

    # Save the consolidated clip to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_result_file:
        temp_result_filename = temp_result_file.name
        consolidated_clip.write_videofile(temp_result_filename, codec="libx264", audio_codec="aac")

    # Return the filename of the consolidated result
    return temp_result_filename
