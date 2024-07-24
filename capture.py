import argparse
import glob
import os
import pandas as pd
import cv2
from models import *
import torch
import numpy as np
from config import Config
from torch.nn import DataParallel
from mtcnn import MTCNN
from albumentations.pytorch import ToTensorV2
import albumentations

THRES = 0.75

def preprocess_frame(image):
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32)
    image = (image - 127.5) / 127.5
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimension for grayscale
    return image

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def process_video(model, device, vid_path, df_list, vid_dir):
    cap = cv2.VideoCapture(vid_path)
    detector = MTCNN()
    idx = 0
    if not cap.isOpened():
        print(f"Error: Could not open video stream {vid_path}.")
        return df_list
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 3)  # Process one frame per second
    while True:
        color = (255,0,0)
        ret, frame = cap.read()
        if not ret:
            break
        w, h = frame.shape[0], frame.shape[1]
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if frame_id % frame_interval == 0:
            detect_box = detector.detect_faces(frame)
            for b in detect_box:
                box = b['box']
                face = frame[ max(0, box[1]) : min(w-1, box[1]+box[3]), max(0, box[0]) : min(h-1, box[0]+box[2]) ]
                input_face = preprocess_frame(face).to(device)
                path  = f'img/{vid_dir}/capture_image_{idx}.jpg'
                img_embeddings = model(input_face).detach().numpy()[0]
                df_list.append([path, str(img_embeddings.tolist())])
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                cv2.imwrite(path, face)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 4)
                idx += 1

        key = cv2.waitKey(1)
        if key == 27:
            break
        # cv2.imshow('image', frame)
    cap.release()
    cv2.destroyAllWindows()
    return df_list

def run_cam(vid_paths, name):
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()   
     
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path, map_location=device))
    model.to(device=device)
    model.eval()

    df_list = []   

    for vid_path in vid_paths:
        base = os.path.basename(vid_path)
        df_list = process_video(model, device, vid_path, df_list, base)
        print(f'done {base}')
    
    df = pd.DataFrame(df_list, columns=['path', 'vector'])
    os.makedirs('df', exist_ok=True)
    df.to_csv(f'df/{name}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facenet capture face')
    parser.add_argument("--name", default='p2v', type=str, help="dataframe csv save name")
    parser.add_argument("--vid_dir", default='vid', type=str, help="Video directory")
    # parser.add_argument("--vid_paths", nargs='+', default=[], type=str, help="List of video paths")
    args = parser.parse_args()
    vid_paths = glob.glob(os.path.join(args.vid_dir, '*'))
    run_cam(vid_paths, args.name)
