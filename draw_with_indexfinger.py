import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# Initialize color points deque for each color
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# Indexes for each color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for drawing
kernel = np.ones((5, 5), np.uint8)

# Canvas colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default is blue

# Create a blank canvas for drawing
paintWindow = np.ones((471, 636, 3)) * 255
cv2.putText(paintWindow, "Finger Drawing - Press ESC to Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
cv2.imshow("Paint", paintWindow)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    # Draw color selection buttons
    frame[0:65, 0:636] = (255, 255, 255)
    cv2.rectangle(frame, (40, 1), (140, 65), colors[0], -1)
    cv2.rectangle(frame, (160, 1), (255, 65), colors[1], -1)
    cv2.rectangle(frame, (275, 1), (370, 65), colors[2], -1)
    cv2.rectangle(frame, (390, 1), (485, 65), colors[3], -1)
    cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 0), -1)

    cv2.putText(frame, "BLUE", (60, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "GREEN", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "RED", (295, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "YELLOW", (400, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    cv2.putText(frame, "CLEAR", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    center = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            index_finger_tip = lmList[8]  # Index finger tip landmark
            print(index_finger_tip)
            center = index_finger_tip

            # Draw circle on fingertip
            cv2.circle(frame, center, 8, colors[colorIndex], -1)

            if center[1] <= 65:
                # Color selection area
                if 40 <= center[0] <= 140:
                    colorIndex = 0
                elif 160 <= center[0] <= 255:
                    colorIndex = 1
                elif 275 <= center[0] <= 370:
                    colorIndex = 2
                elif 390 <= center[0] <= 485:
                    colorIndex = 3
                elif 505 <= center[0] <= 600:
                    # Clear canvas
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255
            else:
                # Drawing mode
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    else:
        # If no hand detected, append new deque to avoid connecting strokes
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines on frame and canvas
    points = [(bpoints, (255, 0, 0)), (gpoints, (0, 255, 0)),
              (rpoints, (0, 0, 255)), (ypoints, (0, 255, 255))]

    for pointList, color in points:
        for i in range(len(pointList)):
            for j in range(1, len(pointList[i])):
                if pointList[i][j - 1] is None or pointList[i][j] is None:
                    continue
                cv2.line(frame, pointList[i][j - 1], pointList[i][j], color, 2)
                cv2.line(paintWindow, pointList[i][j - 1], pointList[i][j], color, 2)

    # Show frames
    cv2.imshow("Air Drawing", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
