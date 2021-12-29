import cv2
import mediapipe as mp
import numpy as np
import time, os
import csv
actions = ['come', 'away', 'spin']
seq_length = 30
secs_for_action = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('./dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if result.multi_hand_landmarks is not None:
                for hand_landmark in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmark, result.mp_hands.HAND_CONNECTIONS)

                    try:
                        # Extract Pose landmarks
                        hand = result.hand_landmarks.landmark
                        hand_row = list(np.array(
                            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand]).flatten())

                        row = hand_row

                        row.insert(0, action)

                        with open('coords.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)


                    except:
                        pass

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

