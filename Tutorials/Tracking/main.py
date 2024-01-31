from ultralytics import YOLO
import cv2

# load model
model = YOLO('/Users/j213h/Documents/test_ObjectDetection/yolov8n.pt')

# load video
cap = cv2.VideoCapture('./test.mp4')

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    # detect objects
    # track objects
    results = model.track(frame, persist=True)

    # plot results
    frame_ = results[0].plot()

    # visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
