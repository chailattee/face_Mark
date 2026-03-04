import cv2
import numpy as np
import time
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import mediapipe as mp

print("Imports complete")

# overlay PNG function
def overlay_transparent(background, overlay, x, y, w=None, h=None):
    if overlay is None: 
        return background
    if w is not None and h is not None:
        overlay = cv2.resize(overlay, (w, h))

    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r)) 
    mask = cv2.merge((a, a, a))

    h_o, w_o, _ = overlay_color.shape
    h_b, w_b, _ = background.shape

    # Prevent crash if filter goes off screen
    if y < 0 or x < 0 or y+h > background.shape[0] or x+w > background.shape[1]:
        return background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(x + w_o, w_b), min(y + h_o, h_b)

    overlay_crop = overlay_color[y1 - y:y2 - y, x1 - x:x2 - x]
    mask_crop = mask[y1 - y:y2 - y, x1 - x:x2 - x]

    roi = background[y1:y2, x1:x2]
    img1_bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_crop))
    img2_fg = cv2.bitwise_and(overlay_crop, mask_crop)
    background[y1:y2, x1:x2] = cv2.add(img1_bg, img2_fg)
    return background

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

print("Before face mesh init")
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
print("after face mesh init")

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

    width = int(abs(x2 - x1) * 1.3)       # scale width
    height = int(width * 0.5)               # maintain aspect ratio

    return overlay_transparent(
        frame,
        img,
        x1,
        y1 - height // 2,
        width,
        height
    )

# Map placement type to function
PLACEMENT_FUNCS = {
    "hat": place_hat,
    "cheeks": place_cheeks,
    "glasses": place_glasses
}

ASSETS_DIR = Path(__file__).parent / "assets"
filters = [
    {"name": "bday_hat", "image": cv2.imread(str(ASSETS_DIR / "bday_hat.png"), cv2.IMREAD_UNCHANGED), "placement": "hat"},
    {"name": "pats_eyes", "image": cv2.imread(str(ASSETS_DIR / "pats_eyes.png"), cv2.IMREAD_UNCHANGED), "placement": "cheeks"},
    {"name": "bike", "image": cv2.imread(str(ASSETS_DIR / "bike.png"), cv2.IMREAD_UNCHANGED), "placement": "glasses"},
    {"name": "chef_hat", "image": cv2.imread(str(ASSETS_DIR / "chef_hat.png"), cv2.IMREAD_UNCHANGED), "placement": "hat"},
]

class FaceFilterTransformer(VideoTransformerBase):
    def __init__(self):

        self.mp_face_mesh= mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        self.current_filter = 0
        self.prev_wrist_x = None
        self.swipe_cooldown = 0.5
        self.last_swipe_time = 0

    def transform(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Face detection
        face_results = self.face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                f = filters[self.current_filter]
                func = PLACEMENT_FUNCS.get(f["placement"])
                if func:
                    frame = func(frame, face_landmarks, f["image"], w, h)

        # Hand swipe detection
        hand_results = self.hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            now = time.time()
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[0].x
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if self.prev_wrist_x is not None and (now - self.last_swipe_time) > self.swipe_cooldown:
                    diff = wrist_x - self.prev_wrist_x
                    if diff > 0.2:
                        self.current_filter = (self.current_filter + 1) % len(filters)
                        self.last_swipe_time = now
                    elif diff < -0.2:
                        self.current_filter = (self.current_filter - 1) % len(filters)
                        self.last_swipe_time = now

                self.prev_wrist_x = wrist_x

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🎭 AR Face Filters")

webrtc_streamer(
    key="face-filter",
    video_transformer_factory=FaceFilterTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

st.info("Swipe your hand left/right to change filters!")