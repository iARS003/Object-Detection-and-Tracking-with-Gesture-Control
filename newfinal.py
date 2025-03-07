import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import time
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  

# Define the focal length of your camera 
focal_length = 700  

# Known real-world object size 
known_width = 0.5 

# Initialize the video capture object 
cap = cv2.VideoCapture(0)

# Initialize MediaPipe for hand tracking (for gesture control)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the CSV files to log the data
detection_log_file = 'detection_log.csv'
track_log_file = 'track_log.csv'

# Ensure that both CSV files exist and create them if they don't
if not os.path.exists(detection_log_file):
    with open(detection_log_file, 'w', newline='') as csvfile:
        detection_writer = csv.writer(csvfile)
        detection_writer.writerow(['Timestamp', 'Object Class', 'Object Confidence', 'Bounding Box', 'Gesture Detected'])

if not os.path.exists(track_log_file):
    with open(track_log_file, 'w', newline='') as csvfile:
        track_writer = csv.writer(csvfile)
        track_writer.writerow(['Timestamp', 'Tracked Object', 'X Position', 'Y Position', 'Bounding Box'])

# Initialize the tracker (CSRT for better accuracy)
tracker = None
roi = None

# Loop until 'q' is pressed or fist gesture is detected
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for better user interaction
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands (for gesture control)
    results = hands.process(rgb_frame)
    
    # Gesture detection: Check for closed fist (stop program on fist)
    fist_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for each finger (thumb, index, middle, etc.)
            landmarks = hand_landmarks.landmark
            
            # Check for closed fist (all fingers curled)
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            little_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
            
            index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
            little_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
            thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]

            # If all fingertips are curled, we detect a fist
            if (index_tip.y > index_mcp.y and
                middle_tip.y > middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                little_tip.y > little_mcp.y and
                thumb_tip.x < thumb_ip.x):  # Thumb is curled inward
                fist_detected = True
                break  # Stop processing further if fist is detected

    # Run the YOLOv8 model on the current frame for object detection
    results = model(frame)

    # If an object is detected and tracker is not initialized, initialize the tracker
    if results:
        # Get the detected boxes, confidences, and classes
        boxes = results[0].boxes.xyxy  
        confidences = results[0].boxes.conf  
        classes = results[0].boxes.cls  

        for i in range(len(boxes)):
            # Extract bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, boxes[i])  # Convert to integer pixel coordinates
            conf = confidences[i].item()  # Get confidence score for this detection
            cls = int(classes[i].item())  # Get class index

            # Calculate the width of the object in pixels
            object_width_pixels = x2 - x1

            # Calculate the distance (in meters) using the formula
            distance = (known_width * focal_length) / object_width_pixels

            # Log the detection data to the detection CSV file
            with open(detection_log_file, 'a', newline='') as csvfile:
                detection_writer = csv.writer(csvfile)
                timestamp = time.time()  # Get the current timestamp
                log_data = [timestamp, model.names[cls], conf, f'[{x1},{y1},{x2},{y2}]', 'Fist Detected' if fist_detected else 'No Gesture']
                detection_writer.writerow(log_data)

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f} Dist: {distance:.2f}cm"  # Display object name, confidence, and distance
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Initialize tracker if it's not already initialized
            if tracker is None:
                roi = (x1, y1, x2 - x1, y2 - y1)  # Define the region of interest (ROI)
                tracker = cv2.TrackerCSRT_create()  # Initialize the CSRT tracker
                tracker.init(frame, roi)

    # If the tracker is initialized, update it in subsequent frames
    if tracker is not None:
        success, roi = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in roi]
            # Log the tracking data to the track CSV file
            with open(track_log_file, 'a', newline='') as csvfile:
                track_writer = csv.writer(csvfile)
                timestamp = time.time()  # Get the current timestamp
                log_data = [timestamp, 'Tracked Object', x, y, f'[{x},{y},{x+w},{y+h}]']
                track_writer.writerow(log_data)

            # Draw the bounding box of the tracked object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # If fist gesture is detected, stop the program
    if fist_detected:
        cv2.putText(frame, "Fist detected - Stopping...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Fist detected! Exiting program.")
        break  # Exit the loop when fist gesture is detected

    # Display the frame with bounding boxes, distance, and possible gesture control
    cv2.imshow("YOLOv8 Object Detection with Gesture Control and Logging", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
