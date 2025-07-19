"""
Aim of the Code:
This program captures live video from the webcam, detects a human face using cvzone's FaceDetector,
calculates the face position relative to the center of the video frame (treated as the origin in a Cartesian coordinate system),
estimates the full head bounding box based on the face location, and
displays the relative coordinates both on the video feed and prints them to the console for monitoring.
"""

import cv2  # OpenCV for video capture and image processing
from cvzone.FaceDetectionModule import FaceDetector  # Face detection module based on MediaPipe
import numpy as np  # For numerical operations (imported for possible extensions)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use default camera

# Set camera frame width and height
ws, hs = 1280, 720
cap.set(3, ws)  # Width
cap.set(4, hs)  # Height

# Check if webcam opened successfully
if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# Initialize the face detector object
detector = FaceDetector()

# Main loop for processing frames
while True:
    success, img = cap.read()  # Read a frame from webcam
    if not success:
        print("Failed to grab frame")
        break  # Exit loop if frame is not captured

    # Detect faces in the frame, bboxs contains face data
    img, bboxs = detector.findFaces(img, draw=False)

    # Draw Cartesian coordinate crosshair lines at the center of the frame
    cv2.line(img, (640, 0), (640, hs), (100, 100, 100), 1)  # Vertical center line
    cv2.line(img, (0, 360), (ws, 360), (100, 100, 100), 1)  # Horizontal center line

    # If at least one face detected
    if bboxs:
        # Get face center coordinates
        fx, fy = bboxs[0]["center"]
        # Get face bounding box (top-left x,y and width,height)
        x, y, w, h = bboxs[0]["bbox"]

        # Calculate relative coordinates to center (Cartesian system)
        rel_x = int(fx - 640)   # X positive right, negative left
        rel_y = int(360 - fy)   # Y positive up, negative down (invert Y axis)

        # Print relative position in console for debugging/tracking
        print(f"Relative X: {rel_x}, Relative Y: {rel_y}")

        # Estimate head box by expanding face bounding box
        head_x = x - int(0.2 * w)  # Expand left
        head_y = y - int(0.4 * h)  # Expand up
        head_w = int(1.4 * w)      # Wider
        head_h = int(1.6 * h)      # Taller

        # Draw face detection markers
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)          # Large circle on face center
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)  # Small filled circle on center
        cv2.putText(img, f'X: {rel_x}, Y: {rel_y}', (fx + 15, fy - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # Relative coordinates near face
        cv2.putText(img, "TARGET LOCKED", (850, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Status message

        # Draw guide lines through face center
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)  # Horizontal line
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)  # Vertical line

        # Draw estimated head bounding box and label it
        cv2.rectangle(img, (head_x, head_y), (head_x + head_w, head_y + head_h),
                      (0, 255, 0), 2)
        cv2.putText(img, "HEAD", (head_x, head_y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    else:
        # No face detected case
        cv2.putText(img, "NO TARGET", (880, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        # Draw a default red circle at frame center
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        # Draw center crosshair lines
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)

    # Display the frame with drawings
    cv2.imshow("Image", img)

    # Wait 1 ms and continue; add 'q' key press break if desired
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
