import os
import re
import cv2
import tqdm
import torch
import random
import argparse
import numpy as np

from model import Conv3DDetector

from torchvision import transforms
from PIL import Image


INPUT_SHAPE = (3, 10, 150, 150)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ Sort in 'human' order. """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def preprocess_image(img, cuda):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    preproc_img = tf(img)
    
    if cuda:
        preproc_img = preproc_img.cuda()
    return preproc_img

def get_stacked_frames(video_dir, cuda):
    frames = []
    file_names = os.listdir(video_dir)
    file_names.sort(key=natural_keys)
    for im in file_names:
        if not im.endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(video_dir, im))
        preproc_img = preprocess_image(img, cuda)
        frames.append(img)
    
    frame_tensor = torch.tensor(frames).permute(3, 0, 1, 2)

    assert frame_tensor.shape == INPUT_SHAPE, 'Wrong shape of frames in dir ' + video_dir

    if cuda:
        frame_tensor = frame_tensor.cuda()
    return frame_tensor

def get_batch(source_dir, cuda, batch_size=64):
    video_name_list = []
    stacked_frames_list = []
    label_list = []
    for i in range(batch_size):
        # Get random dir name
        dir_name = None
        while dir_name is None or dir_name in video_name_list:
            dir_name = random.choice(os.listdir(source_dir))
            if not os.path.isdir(os.path.join(source_dir, dir_name)):
                dir_name = None
        video_name_list.append(dir_name)
        
        # Get frames of then chosen video in a tensor
        stacked_frames_list.append(get_stacked_frames(os.path.join(source_dir, dir_name)))
        
        # Get label of video
        label_str = dir_name[-1]
        if label_str not in ['0', '1']:
            print(f'ERROR: {dir_name} has bad label.')
        label = np.zeros((2))
        label[int(label_str)] = 1.0
        label_list.append(label)

    

    return np.array(stacked_frames_list)

def train(model, source_dir, cuda, nb_epochs=10, batch_size=64):
    train_path = os.path.join(source_dir, 'train')
    nb_video_train = sum(os.path.isdir(os.path.join(train_path, i)) for i in os.listdir(train_path))

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'CPU')

    assert nb_video_train > batch_size
    nb_batch = int(nb_video_train / batch_size)
    for ep in range(nb_epochs):
        for b in range(nb_batch):
            video_batch = get_batch(train_path, cuda, batch_size)

            video_batch = video_batch.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a 3d Convolutional model for Deepfake Detection.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing all training and testing data.',
                        default=None, type=str)
    args = parser.parse_args()
    source_dir = args.source

    sf = get_stacked_frames('C:\\Users\\maxim\\Desktop\\PFE\\test_output\\test\\manipulated_DeepFakeDetection_c23_videos_25_12__walking_down_street_outside_angry__MA71PDNV_face0_1_1', True)
    print(sf.shape)