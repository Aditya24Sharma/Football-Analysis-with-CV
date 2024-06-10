from utils import read_video, save_video
from trackers import Tracker
import time

def main():
    #read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize Tracker
    tracker = Tracker('models/best.pt')


    print('Ran Tracker class')
    
    tracks = tracker.get_object_track(video_frames, 
                                      read_from_stub=True,
                                      stub_path = 'stubs/track_stubs.pkl')


    print('Ran get object track')
    print(type(video_frames))
    print(len(video_frames[0:300]))
    time.sleep(1)

    #Draw output
    output_video_frames = tracker.draw_annotations(video_frames[0:300], tracks) 


    print('got output_video_frames')

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
    