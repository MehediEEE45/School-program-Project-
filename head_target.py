import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# Initialize face detector
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # Get face center
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]

        # Get bounding box for face
        x, y, w, h = bboxs[0]["bbox"]

        # Estimate head bounding box (expand face box)
        head_x = x - int(0.2 * w)
        head_y = y - int(0.4 * h)
        head_w = int(1.4 * w)
        head_h = int(1.6 * h)

        # Draw face circle and face info
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str(pos), (fx + 15, fy - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Draw reference lines
        cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)
        cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)

        # Draw head bounding box
        cv2.rectangle(img, (head_x, head_y), (head_x + head_w, head_y + head_h), (0, 255, 0), 2)
        cv2.putText(img, "HEAD", (head_x, head_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    else:
        # If no face detected
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)

    # Show output
    cv2.imshow("Image", img)
    cv2.waitKey(1)
