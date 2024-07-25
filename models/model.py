import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov9c.pt")

# Open video file
video_path = 'data/Video.mp4'
video = cv2.VideoCapture(video_path)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define output video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('results/output_video.mp4', fourcc, fps, (width, height))

# Initialize variables
roi = []
drawing = False

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

# Create window and set mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_roi)

# Read first frame
ret, frame = video.read()
if not ret:
    print("Error: Could not read video file.")
    exit()

# Let user draw ROI
while True:
    temp_frame = frame.copy()
    if len(roi) > 1:
        cv2.polylines(temp_frame, [np.array(roi)], True, (0, 255, 0), 2)
    cv2.imshow('Frame', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create mask from ROI
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
roi_corners = np.array(roi, dtype=np.int32)
cv2.fillPoly(mask, [roi_corners], (255))

# Function to predict and detect people
def predict_and_detect(model, img, classes=["person"], conf=0.5):
    results = model(img)
    detections = results[0]
    filtered_detections = [det for det in detections if det["class_name"] in classes and det["confidence"] > conf]
    return filtered_detections

# Initialize list to keep track of counted people
tracked_people = []

# Process video
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Detect people using the predict_and_detect method
    detections = predict_and_detect(model, frame, classes=["person"], conf=0.5)
    
    # Count people in ROI
    current_people = []
    for det in detections:
        x, y, w, h = det["bbox"]
        center_x, center_y = int(x + w / 2), int(y + h / 2)
        if mask[center_y, center_x] == 255:
            current_people.append((center_x, center_y))
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Update the tracked_people list
    for person in current_people:
        if person not in tracked_people:
            tracked_people.append(person)
    
    # Draw ROI and count
    cv2.polylines(frame, [roi_corners], True, (0, 255, 0), 2)
    cv2.putText(frame, f'People in area: {len(tracked_people)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write frame to output video
    out.write(frame)
    
    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
video.release()
out.release()
cv2.destroyAllWindows()
