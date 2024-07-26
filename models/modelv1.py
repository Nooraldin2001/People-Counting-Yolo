import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov9c.pt")

# Function to initialize the video
def initialize_video(video_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return video, width, height, fps

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

# Define prediction functions
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# Initialize video
video_path = "./data/Video.mp4"
video, width, height, fps = initialize_video(video_path)
ret, frame = video.read()

# Create window and set mouse callback for drawing ROI
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_roi)

# Allow user to draw ROI
while True:
    temp_frame = frame.copy()
    if len(roi) > 1:
        cv2.polylines(temp_frame, [np.array(roi)], True, (0, 255, 0), 2)
    cv2.imshow('Frame', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# Create mask from ROI
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
roi_corners = np.array(roi, dtype=np.int32)
cv2.fillPoly(mask, [roi_corners], (255, 255, 255))

# Define output video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('./results/output_video.mp4', fourcc, fps, (width, height))

# Process video frames
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame, results = predict_and_detect(model, frame, classes=[0], conf=0.5)
    people = [box for result in results for box in result.boxes if result.names[int(box.cls[0])] == "person"]
    
    count = 0  # Reset count for each frame
    for person in people:
        x1, y1, x2, y2 = person.xyxy[0]
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        if mask[centroid[1], centroid[0]] == 255:  # Check if the centroid is in the ROI
            count += 1  # Increment count for each person in the ROI
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
    
    cv2.polylines(frame, [roi_corners], True, (0, 255, 0), 2)
    cv2.putText(frame, f'People in area: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(frame)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved.")
