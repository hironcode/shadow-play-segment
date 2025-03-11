
from transformers import pipeline, AutoModelForImageSegmentation
from PIL import Image
import cv2
import yaml
import os
import torch
from wasabi import msg
from tqdm import tqdm
from torchvision import transforms
import gc
import subprocess

def init_model(config):
    model = AutoModelForImageSegmentation.from_pretrained(config['model']['model'], trust_remote_code=True)
    # torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to('cuda')
    model.eval()
    model.to(torch.bfloat16)


    image_size = tuple(config['model']['image_size'])
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, transform_image

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
    del cap
    return frames, result

def preprocess(frames, fps_old, fps_new, transform_image=None):
    # Load the image from the URL
    ratio = int(fps_old) // fps_new
    frames = frames[::ratio]
    if transform_image:
        for i in range(len(frames)):
            frames[i] = transform_image(frames[i]).unsqueeze(0).to('cuda').to(torch.bfloat16)
    return frames

def segment(frames, model, is_model=True):
    results = []
    if is_model:
        batch = 1
        for i in tqdm(range(0, len(frames), batch)):
            preds = model(frames[i])[-1].sigmoid().cpu()
            results.append(preds)
            del preds
        return results
    else:
        for frame in tqdm(frames):
            results.append(pipeline(frame))
        return results

def save_frames(preds, path, frames=None, is_model=True):

    if is_model:
        if frames is None:
            raise ValueError("frames must be provided if is_model is True")
        for i in tqdm(range(len(preds))):
            pred = preds[i].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(frames[i].size)
            frames[i].putalpha(mask)
            mask.save(os.path.join(path, f'frame_{i}.png'))

    else:
        frame_len = len(preds)

        for i in range(frame_len):
            objects_num = len(preds[i])
            for j in range(objects_num):
                mask = preds[i][j]['mask']
                label = preds[i][j]['label']
                mask.save(os.path.join(path, f'{label}{j}_{i}.png'))

def save_video(config, weight, height):

    frame_dir = config['output']['frame_dir']

    # Make the frames of the masks into video
    video = cv2.VideoWriter(config['output']['output_path'], cv2.VideoWriter_fourcc(*'mp4v'), config['output']['fps'], (weight, height))

    # target_label = config['model']['label']

    for file in os.listdir(frame_dir).sort(key=lambda x: int(x.split('_')[1].split('.')[0])):
        # if target_label not in file:
        #     continue
        # else:
        #     img = Image.open(os.path.join(frame_dir, file)).convert('RGB')
        #     video.write(img)
        img = Image.open(os.path.join(frame_dir, file)).convert('RGB')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    del video

def main(config_path):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Efficient matrix multiplies
    subprocess.run("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")

    torch.cuda.empty_cache()
    gc.collect()
    # load config
    config = yaml.safe_load(open(config_path))
    output_dir = os.path.dirname(config['output']['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.basename(config['input']['video_path'])
    config['output']['output_dir'] = os.path.join(output_dir, video_name)

    print(config)

    # as of now, only "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" is supported for this usage
    # semantic_segmentation = pipeline("image-segmentation", config['model'])

    # init model
    model, transform_image = init_model(config)

    frames, etc = get(config['input']["video_path"])
    frames = preprocess(frames, etc['fps'], config['output']['fps'], transform_image)

    msg.info("Segmentation Started...")
    with torch.no_grad():
        segments = segment(frames, model)
        torch.cuda.empty_cache()

    frame_dir = os.path.join(config['output']['output_dir'], 'frames')
    config['output']['frame_dir'] = frame_dir
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    save_frames(segments, frame_dir)

    # delete big tensor
    del segments

    config['output']['width'] = etc['width']
    config['output']['height'] = etc['height']

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    save_video(config, etc['width'], etc['height'])
    msg.info("Video saved at", config['output']['output_path'])
    del model, frames

if __name__ == "__main__":
    main("shadow-play-segment/config.yaml")