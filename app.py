import streamlit as st
import cv2
import numpy as np
from models.model_streamlit import process_video, model

# Streamlit interface
st.title("People Counting in Video")
video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if video_file is not None:
    # Save uploaded file to a temporary location
    video_path = 'data/uploaded_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(video_file.read())
    
    output_path = 'results/output_video.mp4'
    
    batch_size = 4  # You can adjust this or even make it a user input
    
    with st.spinner('Processing video...'):
        processed_video_path = process_video(video_path, output_path, model, batch_size)
    
    st.success('Video processed successfully!')
    st.video(processed_video_path)