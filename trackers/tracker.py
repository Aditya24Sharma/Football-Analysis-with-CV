from ultralytics import YOLO
import supervision as sv  #supervision helps to seamlessly track the objects that our model detected 
import pickle
import os
import numpy as np
import cv2
import sys
import pandas as pd
sys.path.append('../')  #allowing python to look for parent directories for the modules

from utils.bbox_utils import get_center_of_bbox, get_bbox_width



class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()   #track objects motion and path by assigning them with tracker id

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()   #back filling for if the first value is missing

        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


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

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)  # '_' is a throwback variable that we are intentionally ignoring
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,  #where ellipse is to be drawn
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle = 0.0,
            startAngle=-45,
            endAngle= 235,
            color=color,
            thickness= 2,
            lineType=cv2.LINE_4, 
        )

        rectange_width = 40
        rectangle_height = 20
        
        x1_rect = x_center - rectange_width//2
        x2_rect = x_center + rectange_width//2

        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (x1_rect, y1_rect),
                          (x2_rect, y2_rect),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(frame, 
                        f'{track_id}', 
                        (x1_text, y1_rect + 15), 
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6, 
                        (0,0,0),
                        2
                        )
            

        # print('Ran the ellipse function')

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,(y-2)],
            [(x - 10), (y - 20)],
            [(x+ 10), (y - 20)]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)    # the inside color
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)   # the border

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num] 

            #Draw players
            for track_id, player in player_dict.items(): #converting the dictionary to lists
                color = player.get('team_color', (0,0,255)) #get funtion for a dictionary returns the value of given key. Also able to provide a placeholder color
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id) #will be sending different colors to different teams

                if player.get('has_ball', False):
                    self.draw_triangle(frame, player['bbox'], (0,0,255) )
            
            #Draw referees
            for track_id, referee in referee_dict.items(): 
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255)) 

            #Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255,0))

            output_video_frames.append(frame)

        return output_video_frames