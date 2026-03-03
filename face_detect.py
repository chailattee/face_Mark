import cv2
import numpy as np
import os
import time

import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# primary camera was "0" 
cap = cv2.VideoCapture(0)
