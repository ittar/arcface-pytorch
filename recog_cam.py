import os
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

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def run_cam(model, device):
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    detector = MTCNN()
    img_taken = False
    sim = -1
    img_embeddings = []
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        color = (255,0,0)
        ret, frame = cap.read()
        w, h = frame.shape[0], frame.shape[1]
        if not ret:
            break
        detect_box = detector.detect_faces(frame)
        for b in detect_box:
            box = b['box']
            face = frame[ max(0, box[1]) : min(w-1, box[1]+box[3]), max(0, box[0]) : min(h-1, box[0]+box[2]) ]
            input_face = preprocess_frame(face).to(device)
            if cv2.waitKey(2) & 0xFF == ord('1'):
                print('Taken picture')
                cv2.imwrite('img/capture.jpg', face)
                img_embeddings = model(input_face).detach()
                img_taken = True
            if (img_taken):
                embeddings = model(input_face).detach()
                sim = cosin_metric(embeddings[0], img_embeddings[0])
                if sim > THRES : color = (0,255,0)
                else : color = (0,0,255)
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 4)
            cv2.putText(frame, str(sim), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(2) & 0xFF == ord('2'):
            print('Release picture')
            img_taken = False
        cv2.putText(frame, 'img taken' if img_taken else '' , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(2) & 0xFF == 27:  # esc key
            break
        cv2.imshow('image', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

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

    run_cam(model, device)



