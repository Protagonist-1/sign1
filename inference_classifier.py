import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Updated labels dictionary to include both letters and phrases
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE',
    27: 'HELLO', 28: 'HELP', 29: 'I LOVE YOU', 30: 'SORRY', 31: 'THANK YOU'
}

sentence = ""
last_prediction = None
letter_start_time = None
confirmed_prediction = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            data_aux = data_aux * 2  # Double features if needed
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = labels_dict[int(prediction[0])]

            current_time = time.time()

            # Letter confirmation logic (Hold for 3 seconds)
            if predicted_label == last_prediction:
                if letter_start_time is None:
                    letter_start_time = current_time
                elif current_time - letter_start_time > 2.5:  # 2.5 seconds confirmation
                    confirmed_prediction = predicted_label
                    letter_start_time = None
            else:
                letter_start_time = None
            last_prediction = predicted_label

            # Add confirmed letter or phrase to sentence
            if confirmed_prediction:
                if confirmed_prediction == "SPACE":
                    sentence += " "
                elif confirmed_prediction in ["HELLO", "HELP", "I LOVE YOU", "SORRY", "THANK YOU"]:
                    sentence += " " + confirmed_prediction + " "  # Add phrase directly
                else:
                    sentence += confirmed_prediction
                confirmed_prediction = None  

            # Draw bounding box and predicted text
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display sentence
    cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display exit message in bottom right
    cv2.putText(frame, "Press 'Q' to exit", (W - 200, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    
    # Exit logic when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nFinal Predicted Sentence: ", sentence)  # Print final text in console
        
        # Save sentence to history file
        with open("prediction_history.txt", "a") as file:
            file.write(sentence + "\n")
        
        break


cap.release()
cv2.destroyAllWindows()
