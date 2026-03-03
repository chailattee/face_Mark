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

### filter placements ###

# hat placement
def place_hat(frame, landmarks, img, w, h):
    left_eye = landmarks.landmark[33]  # Left eye landmark
    right_eye = landmarks.landmark[263]  # Right eye landmark
    forehead = landmarks.landmark[10]  # Forehead landmark

    eye_width = abs(right_eye.x - left_eye.x) * w

    fx = int(forehead.x * w)
    fy = int(forehead.y * h) 

    hat_width = int(eye_width * 2.5)
    hat_height = int(hat_width * 0.9)

    return overlay_transparent(
        frame, 
        img, 
        fx - hat_width // 2, 
        fy - hat_height, 
        hat_width, 
        hat_height)

# cheeks placement
def place_cheeks(frame, landmarks, img, w, h):
    # Nose tip
    nose = landmarks.landmark[1]
    
    # Eye distance for scaling
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    face_width = abs((right_eye.x - left_eye.x) * w)

    size = int(face_width * 1.2)  # scale multiplier

    cx = int(nose.x * w)
    cy = int(nose.y * h)

    # Shift slightly downward from nose
    cy += int(size * 0.2)

    return overlay_transparent(
        frame,
        img,
        cx - size // 2,
        cy - size // 2,
        size,
        size
    )

# glasses placement
def place_glasses(frame, landmarks, img, w, h):
    left = landmarks.landmark[33]
    right = landmarks.landmark[263]

    x1 = int(left.x * w)
    y1 = int(left.y * h)
    x2 = int(right.x * w)

    width = abs(x2 - x1)
    height = int(width * 0.5)

    return overlay_transparent(
        frame,
        img,
        x1,
        y1 - height // 2,
        width,
        height
    )


# face filters

filters = [
    {
        "name": "bday_hat",
        "image": cv2.imread("bday_hat.png", cv2.IMREAD_UNCHANGED),
        "placement": place_hat
    },
    {
        "name": "pats_eyes",
        "image": cv2.imread("pats_eyes.png", cv2.IMREAD_UNCHANGED),
        "placement": place_cheeks
    },
    {
        "name": "bike",
        "image": cv2.imread("bike.png", cv2.IMREAD_UNCHANGED),
        "placement": place_glasses
    },
    {
        "name": "chef_hat",
        "image": cv2.imread("chef_hat.png", cv2.IMREAD_UNCHANGED),
        "placement": place_hat
    }
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

    # Prevent crash if filter goes off screen
    if y < 0 or x < 0 or y+h > background.shape[0] or x+w > background.shape[1]:
        return background
    
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
            h, w, _ = frame.shape

            selected_filter = filters[current_filter]

            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            face_width = abs((right_eye.x - left_eye.x) * w)

            size = int(face_width * 1.2)

            cx = int(nose.x * w)
            cy = int(nose.y * h)

            cy += int(size * 0.2)  # adjust vertical position

            frame = overlay_transparent(
                frame,
                selected_filter,
                cx - size // 2,
                cy - size // 2,
                size,
                size
            )

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

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite(f"snapshot_{int(time.time())}.png", frame)
        print("Snapshot saved!")
    elif key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()