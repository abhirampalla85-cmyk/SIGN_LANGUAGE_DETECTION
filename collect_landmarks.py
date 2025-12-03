# collect_landmarks.py  (A-Z ready)
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create labels A-Z
labels = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
save_base = 'data'
for l in labels:
    os.makedirs(os.path.join(save_base, l), exist_ok=True)

def counts_for_labels():
    out = {}
    for l in labels:
        path = os.path.join(save_base, l)
        try:
            files = [f for f in os.listdir(path) if f.endswith('.npy')]
            out[l] = len(files)
        except FileNotFoundError:
            out[l] = 0
    return out

# Optional start letter argument: python collect_landmarks.py D
start_label = None
if len(sys.argv) > 1:
    arg = sys.argv[1].upper()
    if arg in labels:
        start_label = arg

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Close other apps using camera and retry.")

print("Camera open. SPACE=save sample | n=next | b=prev | q=quit | g=go to letter (type in terminal)")

current_idx = labels.index(start_label) if start_label else 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    # draw landmarks if detected
    if res.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # UI overlay
    cts = counts_for_labels()
    label = labels[current_idx]
    cv2.putText(frame, f'Label: {label}   Count: {cts[label]}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # show a few following labels and counts to help navigation
    y = 60
    for l in labels[current_idx:current_idx+6]:
        cv2.putText(frame, f'{l}: {cts[l]}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        y += 22

    cv2.imshow('collect', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # save sample on SPACE
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            arr = np.array([[p.x, p.y, p.z] for p in lm]).flatten().astype(float)
            fname = os.path.join(save_base, label, f'{int(time.time()*1000)}.npy')
            np.save(fname, arr)
            print(f"Saved {fname} -> {label} (total now {cts[label]+1})")
        else:
            print("No hand detected - move your hand fully into the camera.")
    elif key == ord('n'):  # next label
        current_idx = (current_idx + 1) % len(labels)
        print("Switched to", labels[current_idx])
    elif key == ord('b'):  # previous label
        current_idx = (current_idx - 1) % len(labels)
        print("Switched to", labels[current_idx])
    elif key == ord('q'):  # quit
        break
    elif key == ord('g'):  # quick jump: press g then type letter in terminal
        sys.stdout.write("\nType letter to go to (A-Z) and press Enter in the terminal: ")
        sys.stdout.flush()
        dest = sys.stdin.readline().strip().upper()
        if dest in labels:
            current_idx = labels.index(dest)
            print("Going to", dest)
        else:
            print("Invalid letter:", dest)

cap.release()
cv2.destroyAllWindows()
