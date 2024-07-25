import re
import streamlit as st
import cv2
import numpy as np
from models.model_streamlit import process_video, model

# Initialize variables
roi = []
drawing = False

# Draw ROI function
def draw_roi(event, x, y, flags, param):
    global roi, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi.append((x, y))

# Streamlit interface
st.title("People Counting in Designated Zone")
video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if video_file is not None:
    # Save uploaded file to a temporary location
    video_path = 'data/uploaded_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(video_file.read())
        
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
      
    if not ret:
        st.error("Could not read the video file. Please upload a valid video.")
    else:
        st.text("Draw the ROI on the first frame")
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', draw_roi)
        
        while True:
            temp_frame = frame.copy()
            if len(roi) > 1:
                cv2.polylines(temp_frame, [np.array(roi)], True, (0, 255, 0), 2)
            cv2.imshow('Frame', temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        
        output_path = 'results/output_video.mp4'
        process_video(video_path, output_path, roi, model)
        st.video(output_path)
