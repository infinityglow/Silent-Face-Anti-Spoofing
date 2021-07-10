import os
import cv2

for img in os.listdir("/Users/fenlai/Desktop/Silent-Face-Anti-Spoofing/datasets/org_1_300_300/remake"):
    try:
        cv2.imread(os.path.join("/Users/fenlai/Desktop/Silent-Face-Anti-Spoofing/datasets/org_1_300_300/remake", img))
    except:
        print("error")

