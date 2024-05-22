from ultralytics import YOLO


#yolov8 is convinient than 9
model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save = True)
print(results[0])
print(['x' for i in range(10)])
for box in results[0].boxes:
    print(box)
