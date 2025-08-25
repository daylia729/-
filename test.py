from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
model.predict('work.mp4',save=True,classes = [0,2],line_width = 3)
