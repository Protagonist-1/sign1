import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Ask user whether to collect Letters or Phrases
mode = input("Do you want to collect letters (A-Z) or phrases? (Enter 'letters' or 'phrases'): ").strip().lower()

if mode == 'letters':
    DATA_DIR = os.path.join(DATA_DIR, 'left_hand')  # Store letters inside 'left_hand'
    classes = [str(i) for i in range(26)]  # 0-25 for A-Z
elif mode == 'phrases':
    DATA_DIR = os.path.join(DATA_DIR, 'phrases')  # Store phrases separately
    classes = ['hello', 'thank_you', 'i_love_you', 'help', 'sorry']  # Add more as needed
else:
    print("Invalid choice! Exiting...")
    exit()

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 50  # Number of images per class

cap = cv2.VideoCapture(0)

for label in classes:
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for "{label}" gesture...')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Get Ready for {label}! Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Data collection completed! ✅")
