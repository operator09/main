import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('model.h5')
actions = np.array(['hello', 'thanks', 'iloveyou'])
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

res = model.predict(x_test)
sequence = []
sentence = []
threshold = 0.4

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


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    img, result = mediapipe_detection(frame, holistic)
    draw_landmarks(img, result)
    keypoints = extract_keypoints(result)
    sequence.insert(0, keypoints)
    sequence = sequence[:30]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]

    if res[np.argmax(res)] > threshold:
        if len(sentence) > 0:
            if actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
        else:
            sentence.append(actions[np.argmax(res)])

    if len(sentence) > 5:
        sentence = sentence[-5:]

    cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(img, ''.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Feed', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
