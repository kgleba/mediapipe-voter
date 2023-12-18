import argparse
import cv2
import mediapipe as mp
import numpy as np

parser = argparse.ArgumentParser(description='MediaPipe Voter Server')

parser.add_argument('host', help='Host to connect to', type=str)
parser.add_argument('port', help='Port to connect to', type=int)
parser.add_argument('--live', action='store_true', help='Stream real-time hands detection')

args = parser.parse_args()

with mp.solutions.hands.Hands() as handsDetector:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(1) == ord('q') or not ret:
            break

        flipped = np.fliplr(frame)
        flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results = handsDetector.process(flippedRGB)

        if results.multi_hand_landmarks is not None:
            x_tip = int(results.multi_hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * flippedRGB.shape[0])
            cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
            print(results.multi_hand_landmarks[0])

        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

        if args.live:
            cv2.imshow('Hands', res_image)
