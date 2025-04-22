import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []   
labels = []
label_map = {}  # To store label mapping (e.g., 0 -> 'A', 26 -> 'hello')

# 1️⃣ Process Letter Data (A-Z) for Left & Right Hand
print("Processing letter images (A-Z)...")
letter_base_index = 0

for hand_type in ['left_hand', 'right_hand']:  # Loop through both hands
    HAND_DIR = os.path.join(DATA_DIR, hand_type)

    for label in os.listdir(HAND_DIR):  # 0-25 folders for A-Z
        label_path = os.path.join(HAND_DIR, label)
        label_int = int(label)

        if label_int not in label_map:
            label_map[label_int] = chr(65 + label_int)  # Map 0 → A, 1 → B, ..., 25 → Z

        for img_path in os.listdir(label_path):
            data_aux = []
            x_, y_ = [], []

            img = cv2.imread(os.path.join(label_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                data.append(data_aux)
                labels.append(label_int)

# 2️⃣ Process Phrase Data
print("Processing phrase images...")
phrase_base_index = 26  # Letters go from 0-25, so phrases start at 26
PHRASE_DIR = os.path.join(DATA_DIR, 'phrases')

for idx, phrase in enumerate(os.listdir(PHRASE_DIR)):  
    phrase_path = os.path.join(PHRASE_DIR, phrase)
    phrase_label = phrase_base_index + idx  # Assign unique number to phrases

    label_map[phrase_label] = phrase  # e.g., 26 -> 'hello', 27 -> 'thank_you'

    for img_path in os.listdir(phrase_path):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(phrase_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            data.append(data_aux)
            labels.append(phrase_label)

# 3️⃣ Save dataset with both letters & phrases
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)

print("Dataset created successfully! ✅")
