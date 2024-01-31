import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("/Users/j213h/Documents/test_ObjectDetection/Tutorials/dogs.mp4")
img = "/Users/j213h/Documents/test_ObjectDetection/Tutorials/people.jpg"
model = YOLO("yolov8m.pt")  # YOLO 모델 불러오기

"""
오류 발생!!
WARNING ⚠️ NMS time limit 0.550s exceeded
객체 탐지 계산에 걸린 시간이 0.550초를 넘어갔다는 의미.
즉, 객체 탐지에 필요한 성능이 부족함을 의미한다.
초반에만 오류가 발생하는 경우 크게 상관없는듯..? 🤔
"""

with open("/Users/j213h/Documents/test_ObjectDetection/Tutorials/classes.txt", "r") as f:
    class_names = [i.rstrip() for i in f.readlines()]

option = int(input())

if option == 1:
    results = model.predict(img)
    plots = results[0].plot()
    cv2.imshow('plot', plots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

#cap1 = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #ret, frame = cap1.read()

    if not ret:  # 비디오 재생 시간이 끝나면
        break

    results = model.predict(frame, device="mps")  # enable mps
    result = results[0]  # 어떤 클래스로 탐지했는지 확인
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # 바운딩박스
    classes = np.array(result.boxes.cls.cpu(), dtype="int")  # 클래스
    for bbox, cls in zip(bboxes, classes):
        (x, y, x2, y2) = bbox
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225))
        # 탐지 결과 어떤 클래스인지 나타내기
        cv2.putText(frame, class_names[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1100)  # 1이라면 키가 눌리기 전까지 계속 비디오 재생

    if key == 27:  # ESC 키가 눌리면
        break

cap.release()
#cap1.release()
cv2.destroyAllWindows()

"""
MAC M2와 3060GTX 실행시간 비교
MAC M2 평균 : 25 ~ 28 ms
RTX 3060 평균 : 17 ~ 19 ms
생각보다 시간 차이 별로안난다! 👍
"""
