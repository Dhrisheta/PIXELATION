import cv2
import numpy as np
import os
import yt_dlp
import streamlit as st
import matplotlib.pyplot as plt

# Ensure correct paths
weights_path = "D:/COLLEGE/SEM 5/LAB/ML/Project/yolov3.weights"
config_path = "D:/COLLEGE/SEM 5/LAB/ML/Project/yolov3.cfg"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to get YouTube stream URL using yt-dlp and cookies
def get_youtube_stream_url(video_url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Select the best video format with mp4 extension
        'noplaylist': True,              # Ensure only a single video is downloaded
        'cookiefile': 'D:/COLLEGE/SEM 5/LAB/ML/Project/youtube_cookies.txt',  # Path to your cookies file
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_url = info_dict.get("url", None)
        return video_url

# Function to process video
def process_video(stream_url):
    # Initialize the output video path first to avoid undefined variable issues
    output_video_path = "D:/COLLEGE/SEM 5/LAB/ML/Project/youtube_processed.mp4"
    final_accuracy = 0
    accuracy_history = []

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        st.error("Error: Could not open input video stream.")
        return None, None, None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for .mp4 format
    fps2 = 30
    out = cv2.VideoWriter(output_video_path, fourcc, fps2, (frame_width, frame_height))

    if not out.isOpened():
        st.error("Error: Could not open output video file for writing.")
        cap.release()
        return None, None, None

    # Function to detect blood (based on red color detection)
    def detect_blood(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Pixelation function
    def pixelate_region(frame, x, y, w, h):
        roi = frame[y:y+h, x:x+w]
        roi_small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
        roi_pixelated = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = roi_pixelated

    # Start processing the video frame by frame
    frame_counter = 0
    correct_detections = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        height, width, channels = frame.shape

        # Prepare the frame for YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Get the bounding boxes for detected objects (e.g., person)
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # confidence threshold
                    center_x, center_y, w, h = int(detection[0] * width), int(detection[1] * height), int(detection[2] * width), int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Ensure the bounding box is within the frame
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    total_detections += 1  # Increment total detections

        # Non-maximum suppression to eliminate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Pixelate blood regions
        blood_contours = detect_blood(frame)
        if blood_contours:
            for contour in blood_contours:
                x, y, w, h = cv2.boundingRect(contour)
                pixelate_region(frame, x, y, w, h)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                pixelate_region(frame, x, y, w, h)
                correct_detections += 1  # Increment correct detections

        # Calculate accuracy
        if total_detections > 0:
            accuracy = (correct_detections / total_detections) * 100
            accuracy_history.append(accuracy)

        # Write the processed frame to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Calculate final accuracy
    final_accuracy = (correct_detections / total_detections) * 100 if total_detections > 0 else 0

    return output_video_path, final_accuracy, accuracy_history

# Streamlit UI
st.title("YouTube Video Processor")

# Input for YouTube URL
youtube_video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if youtube_video_url:
        with st.spinner("Processing video..."):
            stream_url = get_youtube_stream_url(youtube_video_url)
            if stream_url:
                output_video_path, final_accuracy, accuracy_history = process_video(stream_url)
                st.success("Video processing complete!")

                # Display final accuracy
                st.metric("Final Accuracy", f"{final_accuracy:.2f}%")

                # Plot accuracy history
                plt.figure(figsize=(10, 5))
                plt.plot(accuracy_history, label='Accuracy (%)', color='blue')
                plt.title('Model Accuracy Over Time')
                plt.xlabel('Frame Number')
                plt.ylabel('Accuracy (%)')
                plt.grid()
                plt.legend()
                st.pyplot(plt)

                # Provide download link for the processed video
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",  # Button label
                        data=file,  # File to download
                        file_name="youtube_processed.mp4",  # Name of the downloaded file
                        mime="video/mp4"  # MIME type for the video
                    )

                # Display the processed video in the UI with a play/pause button
                st.video(output_video_path)

            else:
                st.error("Failed to retrieve video stream URL.")
    else:
        st.warning("Please enter a valid YouTube URL.")
