from transformers import pipeline
from PIL import Image
import cv2
import yaml
import torch
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

    # Make the frames of the masks into video
    video = cv2.VideoWriter(config['output']['output_path'], cv2.VideoWriter_fourcc(*'mp4v'), config['output']['fps'], (weight, height))

    for i in range(len(results)):
        video.write(results[i]['mask'])

    cv2.destroyAllWindows()
    video.release()
    print("Video saved at", config['output']['output_path'])
    
def main():
    config = yaml.safe_load(open("config.yaml"))
    output_dir = os.path.dirname(config['output']['output_path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.basename(config['video_path'])
    config['output']['output_path'] = os.path.join(output_dir, video_name)

    # as of now, only "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" is supported for this usage
    semantic_segmentation = pipeline("image-segmentation", config['model'])
    frames, etc = get(config["video_path"])
    frames = preprocess(frames, etc['fps'], config['output']['fps'])
    with torch.no_grad():
        segments = segment(frames, semantic_segmentation)
    connect_save(segments, config, etc['width'], etc['height'])

if __name__ == "__main__":
    main()
    
    

