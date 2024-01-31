from ultralytics import YOLO
import cv2
import glob
import random

model = YOLO('./runs/detect/train/weights/best.pt')
files = glob.glob('./Drones-1/valid/images/*.jpg')

for _ in range(10):
    file = random.choice(files)
    results = model.predict(file, device='mps')
    plots = results[0].plot()
    cv2.imshow('plot', plots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
