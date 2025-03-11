from transformers import pipeline
from PIL import Image
import cv2
import yaml
import os

def get(video_path):
    # Use cv2 to read the video at fps=30
    cap = cv2.VideoCapture(video_path)
    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, {
        "fps": fps,
        "width": frame.shape[1],
        "height": frame.shape[0]
    }


def preprocess(frames, fps_old, fps_new):
    # Load the image from the URL
    ratio = fps_old // fps_new
    frames = frames[::ratio]
    return frames

def segment(frames, pipeline):
    results = []
    for frame in frames:
        results.append(pipeline(frame))
    return results

def connect_save(results, config, weight, height):
    # Connect the segments
    masks = [results[i]['mask'] for i in range(len(results))]
    # Make the frames of the masks into video
    video = cv2.VideoWriter(config['output']['outtput_path'], cv2.VideoWriter_fourcc(*'mp4v'), config['output']['fps'], (weight, height))
    for image in results:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    print("Video saved at", config['output']['output_path'])
    

def main():
    config = yaml.safe_load(open("config.yaml"))
    semantic_segmentation = pipeline("image-segmentation", "briaai/RMBG-2.0")
    frames, etc = get(config["video_path"])
    frames = preprocess(frames, etc['fps'], config['output']['fps'])
    segments = segment(frames, semantic_segmentation)
    connect_save(segments, config, etc['width'], etc['height'])

if __name__ == "__main__":
    main()
    
    

