# This code captures live video from your webcam,
# detects a human face using cvzone (MediaPipe wrapper),
# then estimates and displays the full head region based on the face position.
# No servos are controlled – it's purely for visualization.

import cv2  # OpenCV for image processing and camera access
from cvzone.FaceDetectionModule import FaceDetector  # Simplified face detection
import numpy as np  # For calculations like interpolation and scaling

# Start webcam capture (camera index 0 = default camera)
cap = cv2.VideoCapture(0)

# Define webcam resolution
ws, hs = 1280, 720  # Width and Height
cap.set(3, ws)  # Set frame width
cap.set(4, hs)  # Set frame height

# Check if the camera opened successfully
if not cap.isOpened():
    print("Camera couldn't Access!!!")  # Error message if webcam not found
    exit()  # Exit program

# Initialize the face detector from cvzone
detector = FaceDetector()

# Start an infinite loop to continuously read and process video frames
while True:
    success, img = cap.read()  # Capture one frame from the webcam
    img, bboxs = detector.findFaces(img, draw=False)  # Detect faces in the frame (returns bounding boxes)

    # If at least one face is detected
    if bboxs:
        # Get the center of the first detected face
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]  # fx = X-coordinate, fy = Y-coordinate
        pos = [fx, fy]  # Store center position

        # Get the face bounding box: x, y, width, height
        x, y, w, h = bboxs[0]["bbox"]

        # Estimate the head box by expanding the face bounding box
        head_x = x - int(0.2 * w)  # Shift left
        head_y = y - int(0.4 * h)  # Shift up
        head_w = int(1.4 * w)      # Make wider
        head_h = int(1.6 * h)      # Make taller

        # Draw a large red circle around the face
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)

        # Draw a small red filled dot at the face center
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)

        # Show the (x, y) face center coordinates on screen
        cv2.putText(img, str(pos), (fx + 15, fy - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display "TARGET LOCKED" when a face is found
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Draw crosshair lines across the face center
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)   # Horizontal line
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)   # Vertical line

        # Draw a green box around the estimated head region
        cv2.rectangle(img, (head_x, head_y), (head_x + head_w, head_y + head_h), (0, 255, 0), 2)

        # Label the green box as "HEAD"
        cv2.putText(img, "HEAD", (head_x, head_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    else:
        # If no face is detected, display a warning message
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        # Draw a static red circle at the center of the screen
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)

        # Draw crosshair lines through the center
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)   # Horizontal center line
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)   # Vertical center line

    # Display the frame with drawings
    cv2.imshow("Image", img)

    # Wait 1ms and check for key press (used for smooth loop)
    cv2.waitKey(1)
