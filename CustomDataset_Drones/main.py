from ultralytics import YOLO

model = YOLO('../yolov8s.pt')

results = model.train(data='/Users/j213h/Documents/test_ObjectDetection/CustomDataset_Drones/Drones-1/data.yaml', epochs=10, device='mps')