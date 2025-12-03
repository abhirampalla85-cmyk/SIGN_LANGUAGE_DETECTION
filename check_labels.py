import numpy as np

labels = np.load('label_classes.npy', allow_pickle=True)
print("Labels loaded (index -> label):")
for i, l in enumerate(labels):
    print(i, "->", l)
