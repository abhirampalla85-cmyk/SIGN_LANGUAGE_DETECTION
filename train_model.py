# train_model.py
import numpy as np
import os, glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = 'data'

# discover labels (sorted for stable order)
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not labels:
    raise SystemExit("No data found in 'data' folder. Run collect_landmarks.py first.")
print("Found labels:", labels)

# load samples
X, y = [], []
for lab in labels:
    pattern = os.path.join(DATA_DIR, lab, '*.npy')
    for f in glob.glob(pattern):
        X.append(np.load(f).astype(float))
        y.append(lab)

X = np.array(X)
y = np.array(y)
print("Loaded samples:", X.shape)  # e.g. (1506, 63)

# === Per-sample normalization (important: matches realtime) ===
# Normalize each sample separately (axis=1) so training and realtime preprocessing match.
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

# Encode labels and save mapping
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
np.save('label_classes.npy', encoder.classes_)
print("Saved label_classes.npy ->", encoder.classes_)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
print("Train/Val sizes:", X_train.shape[0], X_val.shape[0])

# Build a simple dense network
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

# Save model
model.save('hand_landmark_model.h5')
print("Saved model: hand_landmark_model.h5")
