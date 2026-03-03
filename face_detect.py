import cv2
import numpy as np
import os
import time

import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# face filters
filters = [
    cv2.imread("bday_hat.png", cv2.IMREAD_UNCHANGED),  # RGBA images with transparency
    cv2.imread("bike.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("chef_hat.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("pats_eyes.png", cv2.IMREAD_UNCHANGED),
]
current_filter = 0

# Hand swipe variables

prev_wrist_x = None
swipe_cooldown = 0.5  # seconds
last_swipe_time = 0

# overlay PNG function
def overlay_transparent(background, overlay, x, y, w=None, h=None):
    if w is not None and h is not None:
        overlay = cv2.resize(overlay, (w, h))

    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r)) 
    mask = cv2.merge((a, a, a))
    h, w, _ = overlay_color.shape
    roi = background[y:y+h, x:x+w]

    img1_bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, mask)
    background[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    return background


# primary camera was "0" --> start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect face and hands
    face_results = face_mesh.process(rgb_frame)

    # if face detected, apply filter
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            #EDIT FOR PLACEMENT OF DIFFERENT FILTERS

            # overlay current filter on detected face region
            # filter_img = filters[current_filter]

    # if hand detected, check for swipe gesture to change filter
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist_x = hand_landmarks.landmark[0].x  # Wrist landmark
            now = time.time()

            if prev_wrist_x is not None and (now - last_swipe_time) > swipe_cooldown:
                diff = wrist_x - prev_wrist_x
                if diff > 0.2:  # Swipe right
                    current_filter = (current_filter + 1) % len(filters)
                    last_swipe_time = now
                elif diff < -0.2:  # Swipe left
                    current_filter = (current_filter - 1) % len(filters)
                    last_swipe_time = now

            prev_wrist_x = wrist_x

    # display the resulting frame
    cv2.imshow('Face Filter', frame)

    cv2.imshow("Face Filter App", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite(f"snapshot_{int(time.time())}.png", frame)
        print("Snapshot saved!")
    elif key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()