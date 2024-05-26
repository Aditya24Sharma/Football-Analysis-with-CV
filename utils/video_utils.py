import cv2

def read_video(video_path):
    vid = cv2.VideoCapture(video_path)  #creates a video object
    frames = []
    while True:


        ret, frame = vid.read()    #returns a boolean(check if frame was read) and numpy array(image data)
        if not ret:
            break
        frames.append(frame)

    #important to release video to avoid release leaks
    vid.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  #four character code for the given format XVID
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()             
