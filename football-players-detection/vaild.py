from ultralytics import YOLO
import cv2
import glob
import random

model = YOLO("./runs/detect/train3/weights/best.pt")
files = glob.glob("./valid/images/*.jpg")

if int(input('Press key: ')) == 1:
    for _ in range(10):
        file = random.choice(files)
        results = model.predict(file, device='mps')
        plots = results[0].plot()
        cv2.imshow('plot', plots)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    file = './valid/images/a9f16c_8_10_png.rf.3028b2b8a61ac775683f2f11f6293053.jpg'
    results = model.predict(file, device='mps')
    plots = results[0].plot()
    cv2.imshow('plot', plots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()