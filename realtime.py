# realtime.py
import cv2, mediapipe as mp, numpy as np, os
from tensorflow.keras.models import load_model
from collections import deque, Counter

if not os.path.exists('hand_landmark_model.h5'):
    raise SystemExit("No model file found. Run train_model.py first.")
model = load_model('hand_landmark_model.h5')
labels = list(np.load('label_classes.npy', allow_pickle=True))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pred_queue = deque(maxlen=7)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        pred = model.predict(arr.reshape(1, -1), verbose=0)
        idx = pred.argmax()
        pred_queue.append(labels[idx])
        if len(pred_queue) == pred_queue.maxlen:
            letter = Counter(pred_queue).most_common(1)[0][0]
        else:
            letter = labels[idx]
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Pred: {letter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No hand', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('realtime', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
