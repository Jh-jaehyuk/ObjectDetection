from ultralytics import YOLO

# Load a model
model = YOLO('/Users/j213h/Documents/test_ObjectDetection/yolov8s.pt')

# Use the model
results = model.train(data='/Users/j213h/Documents/test_ObjectDetection/CustomDataset_FootballPlayer/football-players-detection/data.yaml', epochs=20, device='mps') # train