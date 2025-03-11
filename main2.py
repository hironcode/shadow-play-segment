from transformers import pipeline
from PIL import Image
import cv2
import yaml
import os
import torch
from wasabi import msg
from tqdm import tqdm

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
        frame = Image.fromarray(frame)
        frames.append(frame)
    result = {
        "fps": fps,
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    }
    cap.release()

    return frames, result

def preprocess(frames, fps_old, fps_new):
    # Load the image from the URL
    ratio = int(fps_old) // fps_new
    frames = frames[::ratio]
    return frames

def segment(frames, pipeline):
    results = []
    for frame in tqdm(frames):
        results.append(pipeline(frame))
    return results

def save_frames(results, path):
    frame_len = len(results)

    for i in range(frame_len):
        objects_num = len(results[i])
        for j in range(objects_num):
            mask = results[i][j]['mask']
            label = results[i][j]['label']
            mask.save(os.path.join(path, f'{label}{j}_{i}.png'))

def save_video(config, weight, height):

    frame_dir = config['output']['frame_dir']

    # Make the frames of the masks into video
    video = cv2.VideoWriter(config['output']['output_path'], cv2.VideoWriter_fourcc(*'mp4v'), config['output']['fps'], (weight, height))

    target_label = config['model']['label']

    for file in os.listdir(frame_dir).sort(key=lambda x: int(x.split('_')[1].split('.')[0])):
        if target_label not in file:
            continue
        else:
            img = Image.open(os.path.join(frame_dir, file)).convert('RGB')
            video.write(img)

    cv2.destroyAllWindows()
    video.release()

def main(config_path):
    config = yaml.safe_load(open(config_path))
    output_dir = os.path.dirname(config['output']['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.basename(config['input']['video_path'])
    config['output']['output_dir'] = os.path.join(output_dir, video_name)

    # as of now, only "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" is supported for this usage
    semantic_segmentation = pipeline("image-segmentation", config['model'])
    frames, etc = get(config['input']["video_path"])
    frames = preprocess(frames, etc['fps'], config['output']['fps'])

    msg.info("Segmentation Started...")
    with torch.no_grad():
        segments = segment(frames, semantic_segmentation)

    frame_dir = os.path.join(config['output']['output_dir'], 'frames')
    config['output']['frame_dir'] = frame_dir
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    save_frames(segments, frame_dir)

    config['output']['width'] = etc['width']
    config['output']['height'] = etc['height']

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    save_video(config, etc['width'], etc['height'])
    msg.info("Video saved at", config['output']['output_path'])

main("shadow-play-segment/config.yaml")

