import os
import cv2

# Define original and new dataset directories
left_hand_dir = os.path.join("data", "left_hand")
right_hand_dir = os.path.join("data", "right_hand")
phrases_dir = os.path.join("data", "phrases")
flipped_phrases_dir = os.path.join("data", "flipped_phrases")

# Create right_hand directory if not exists
if not os.path.exists(right_hand_dir):
    os.makedirs(right_hand_dir)

# Create flipped_phrases directory if not exists
if not os.path.exists(flipped_phrases_dir):
    os.makedirs(flipped_phrases_dir)

# ========== FLIP left_hand -> right_hand ==========
for class_name in os.listdir(left_hand_dir):
    original_folder = os.path.join(left_hand_dir, class_name)
    flipped_folder = os.path.join(right_hand_dir, class_name)

    if not os.path.exists(flipped_folder):
        os.makedirs(flipped_folder)

    for filename in os.listdir(original_folder):
        img_path = os.path.join(original_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        flipped_image = cv2.flip(image, 1)
        flipped_img_path = os.path.join(flipped_folder, filename)
        cv2.imwrite(flipped_img_path, flipped_image)

# ========== FLIP phrases -> flipped_phrases ==========
for class_name in os.listdir(phrases_dir):
    original_folder = os.path.join(phrases_dir, class_name)
    flipped_folder = os.path.join(flipped_phrases_dir, class_name)

    if not os.path.exists(flipped_folder):
        os.makedirs(flipped_folder)

    for filename in os.listdir(original_folder):
        img_path = os.path.join(original_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        flipped_image = cv2.flip(image, 1)
        flipped_img_path = os.path.join(flipped_folder, filename)
        cv2.imwrite(flipped_img_path, flipped_image)

print("Flipped right_hand and flipped_phrases datasets created successfully!")
