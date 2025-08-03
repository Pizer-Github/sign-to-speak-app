import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time

with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_prediction = ""
current_letter = ""
predicted_word = ""
stable_count = 0
stable_threshold = 22

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data)[0]


            if prediction == current_letter:
                stable_count += 1
            else:
                current_letter = prediction
                stable_count = 0

            if stable_count == stable_threshold and prediction != prev_prediction:
                predicted_word += prediction
                prev_prediction = prediction
                stable_count = 0  

            cv2.putText(image, f"Current: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f"Word: {predicted_word}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if predicted_word != "":
            engine.say(predicted_word)
            engine.runAndWait()
            predicted_word = ""
            prev_prediction = ""
            stable_count = 0

    if key == ord('q'):
        break

    cv2.imshow("Sign Language to Text", image)

cap.release()
cv2.destroyAllWindows()