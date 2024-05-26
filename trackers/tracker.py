from ultralytics import YOLO
import supervision as sv 

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20         #instead of detecting every frame we will detect in every 20 frame
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)   #predict funtion detects the objects
            detections += detections_batch
            break
        return detections
    
    def get_object_track(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {value:key for key, value in cls_names.items()}     #inversing the key and value pairs

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #Convert goalkeeper to player object
            for obj_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_id] = cls_names_inv["player"]

            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            print(detection_supervision)