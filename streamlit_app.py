import streamlit as st
import numpy as np
import cv2
import tempfile
from lane_detection_function import detectLaneLines, createDownloadLink, divideBatchProcess
from moviepy.editor import VideoFileClip

# Streamlit app title and header
st.title("Lane Line Detection Project")
st.header("Context")

# Context caption
st.caption(
    body="As we navigate the road, our eyes serve as the key navigational tools, interpreting the environment around us. "
         "The lane lines on the road act as a dependable guide, offering crucial cues on steering direction. "
         "In the realm of self-driving car development, a pivotal objective is leveraging sophisticated algorithms to "
         "autonomously recognize these lane lines, enhancing the vehicle's ability to navigate with precision and safety. "
         "This project is designed to address these concerns by automatically detecting lane lines and identifying eligible driving areas. "
)

# Global variables for storing the last successfully detected invisible lines
last_invisible_left_line = None
last_invisible_right_line = None

st.header("Usage")
st.caption("For testing, please click the button for downloading examples from [Udacity](https://github.com/udacity/CarND-LaneLines-P1/tree/master)")

# Download files to user machines
col1, col2 = st.columns(2)

with col1:
    st.text("Images")
    createDownloadLink('test_images', '.jpg')

with col2:
    st.text("Videos")
    createDownloadLink('test_videos', '.mp4')

# Upload image or video for detection
uploaded_content = st.file_uploader("Upload an image or video for detection", type=["jpeg", "jpg", "mp4"])

if uploaded_content is not None:
    if uploaded_content.type == 'image/jpeg' or uploaded_content.type == 'image/jpg':
        # Process the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_content.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detectLaneLines(image)
        st.image(result, caption='Detected Lane Lines', use_column_width=True)

        # Convert the processed image (NumPy array) to bytes
        result_bytes = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes()

        # Add a download button for the processed image
        image_download_button = st.download_button(label="Download Processed Image", key="image_download", data=result_bytes, file_name="processed_image.jpg")

    elif uploaded_content.type == 'video/mp4':
        # Process the uploaded video
        video_bytes = uploaded_content.read()

        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_filename = temp_file.name

        # Create batch duration slider
        original_clip = VideoFileClip(temp_filename)
        st.info("Because of memory risks and processing speed, we will divide video into smaller parts for efficiency. "
                   "Please choose a batch duration for dividing videos into each smaller ones.")
        batch_duration = st.slider("Choose batch duration", 1, int(original_clip.duration), 1, 1)

        # Process the video in batches and consolidate
        output_video_path = divideBatchProcess(temp_filename, batch_duration=batch_duration)

        # Display the processed video using st.video
        st.video(output_video_path, format="video/mp4", start_time=0)

        # Convert the processed video to bytes
        video_bytes = open(output_video_path, "rb").read()

        # Add a download button for the processed video
        video_download_button = st.download_button(label="Download Processed Video", key="video_download", data=video_bytes, file_name="processed_video.mp4")
    else:
        st.error("Please provide an image of type .jpeg, .jpg or a video of type .mp4")

