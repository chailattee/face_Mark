import cv2
import numpy as np
import time
from pathlib import Path

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

    width = int(abs(x2 - x1) * 1.9)       # scale width
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
print(f"[DEBUG] Assets directory: {ASSETS_DIR}")
filters = [
    {"name": "bday_hat", "image": cv2.imread(str(ASSETS_DIR / "bday_hat.png"), cv2.IMREAD_UNCHANGED), "placement": "hat"},
    {"name": "pats_eyes", "image": cv2.imread(str(ASSETS_DIR / "pats_eyes.png"), cv2.IMREAD_UNCHANGED), "placement": "cheeks"},
    {"name": "bike", "image": cv2.imread(str(ASSETS_DIR / "bike.png"), cv2.IMREAD_UNCHANGED), "placement": "glasses"},
    {"name": "chef_hat", "image": cv2.imread(str(ASSETS_DIR / "chef_hat.png"), cv2.IMREAD_UNCHANGED), "placement": "hat"},
]
print(f"[DEBUG] Loaded {len(filters)} filters")
for i, f in enumerate(filters):
    print(f"[DEBUG] Filter {i}: {f['name']} (placement: {f['placement']}, image loaded: {f['image'] is not None})")

current_filter = 0
prev_wrist_x = None
swipe_cooldown = 0.5
last_swipe_time = 0

# -------------------------------
# Mediapipe initialization
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()
print("[DEBUG] Camera initialized successfully")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[DEBUG] Processing frame #{frame_count}")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)
    h, w, _ = frame.shape


    # Face detection
    if face_results and face_results.multi_face_landmarks:
        print(f"[DEBUG] Detected {len(face_results.multi_face_landmarks)} face(s)")
        for face_landmarks in face_results.multi_face_landmarks:

            f = filters[current_filter]
            func = PLACEMENT_FUNCS.get(f["placement"])
            if func:
                print(f"[DEBUG] Applying filter: {f['name']} (placement: {f['placement']})")
                frame = func(frame, face_landmarks, f["image"], w, h)
    else:
        if frame_count % 30 == 0:
            print("[DEBUG] No face detected in frame")

    # Hand swipe detection
    hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hand_results.multi_hand_landmarks:
        print(f"[DEBUG] Hand(s) detected: {len(hand_results.multi_hand_landmarks)}")
        now = time.time()
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist_x = hand_landmarks.landmark[0].x
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if prev_wrist_x is not None and (now - last_swipe_time) > swipe_cooldown:
                diff = wrist_x - prev_wrist_x
                print(f"[DEBUG] Swipe detected - diff: {diff:.3f}, prev_x: {prev_wrist_x:.3f}, curr_x: {wrist_x:.3f}")
                if diff > 0.2:
                    current_filter = (current_filter + 1) % len(filters)
                    last_swipe_time = now
                    print(f"[DEBUG] Swiped right → changed filter to: {filters[current_filter]['name']}")
                elif diff < -0.2:
                    current_filter = (current_filter - 1) % len(filters)
                    last_swipe_time = now
                    print(f"[DEBUG] Swiped left → changed filter to: {filters[current_filter]['name']}")

            prev_wrist_x = wrist_x

    # Display current filter name
    cv2.imshow("Filter: " + filters[current_filter]["name"], frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        print("[DEBUG] ESC key pressed - exiting application")
        break
    elif key == ord(' '):
        filename = f"snapshot_{filters[current_filter]['name']}.png"
        cv2.imwrite(filename, frame)
        print(f"[DEBUG] Snapshot saved as {filename}")

cap.release()
cv2.destroyAllWindows()