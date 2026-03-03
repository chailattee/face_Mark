import cv2
import numpy as np
import os
import time

import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# primary camera was "0" 
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)