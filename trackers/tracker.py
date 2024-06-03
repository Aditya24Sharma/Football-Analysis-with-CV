from ultralytics import YOLO
import supervision as sv  #supervision helps to seamlessly track the objects that our model detected 
import pickle
import os


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()   #track objects motion and path by assigning them with tracker id

    def detect_frames(self, frames):
        batch_size = 20         #instead of detecting every frame we will detect in every 20 frame
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)   #predict funtion detects the objects
            detections += detections_batch

        return detections
    
    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        
        # to load the tracks if it is already there
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)

        tracks = {
            # object: [{frame1},{frame2}, {tracker_id:{bbox:[0,0,0,0]}}]
            'players': [],
            'referees': [],
            'ball':[],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {value:key for key, value in cls_names.items()}     #inversing the key and value pairs

            #creating a new dictionary for each frame
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #Convert goalkeeper to player object
            for obj_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_id] = cls_names_inv["player"]

            #Track Objects
            #update_with_detections is used to hold the current position of the player and connect it with past trajectories
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # print('+'*70)

            # print(detection_with_tracks)

            for frame_detections in detection_with_tracks:  #provides a list of the detections with bbox, cls, trk_id, etc for each detections in a frame

                bbox = frame_detections[0].tolist()
                cls_id = frame_detections[3]
                tracker_id = frame_detections[4]


                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][tracker_id] = {'bbox':bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][tracker_id] = {'bbox':bbox}

             
            #We don't need to track the ball for its data therefore there is no need to use detection_with_tracks
            for frame_detections in detection_supervision:
                bbox = frame_detections[0].tolist()
                cls_id = frame_detections[3]
                tracker_id = frame_detections[4]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox} #no use of tracking id of a ball so we put just a constant
            
            #store tracks for first creation
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks,f) 
        
        return tracks
