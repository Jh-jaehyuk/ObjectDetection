from ultralytics import YOLO

model = YOLO("/Users/j213h/Documents/test_ObjectDetection/yolov8n.pt")

results = model.train(data="/Users/j213h/Documents/test_ObjectDetection/Easyocr/License Plate Recognition/data.yaml",
                      epochs=5, device='mps')