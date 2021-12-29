import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result


def draw_landmarks(image, result):
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(100, 0, 0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(100, 0, 0), thickness=2, circle_radius=2))


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    img, result = mediapipe_detection(frame, holistic)
    draw_landmarks(img, result)
    cv2.imshow('Feed', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() \
        if result.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() \
        if result.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() \
        if result.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() \
        if result.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


result_test = extract_keypoints(result)
np.save("0", result_test)

DATA_PATH = os.path.join('MP_DATA')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequence = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_sequence):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
for action in actions:
    for sequence in range(no_sequence):
        for frame_num in range(sequence_length):

            ret, frame = cap.read()

            img, result = mediapipe_detection(frame, holistic)
            draw_landmarks(img, result)

            if frame_num == 0:
                cv2.putText(img, 'starting collecting', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(img, 'collecting {} VIDEO NUMBER {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feed', img)
                cv2.waitKey(2000)
            else:
                cv2.putText(img, 'collecting {} VIDEO NUMBER {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feed', img)

            keypoint = extract_keypoints(result)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoint)

            cv2.imshow('Feed', img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
