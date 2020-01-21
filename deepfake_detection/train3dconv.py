import os
import re
import cv2
import random
import argparse
import numpy as np

# from model import get_3dConv_model


INPUT_SHAPE = (10, 150, 150, 3)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ Sort in 'human' order. """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_stacked_frames(video_dir):
    frames = []
    file_names = os.listdir(video_dir)
    file_names.sort(key=natural_keys)
    for im in file_names:
        if not im.endswith('.jpg'):
            continue
        frames.append(cv2.imread(os.path.join(video_dir, im)))

    assert np.array(frames).shape == INPUT_SHAPE, 'Wrong number of images in dir ' + video_dir

    return np.array(frames)

def get_batch(source_dir, batch_size=64):
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
        
        stacked_frames_list.append(get_stacked_frames(os.path.join(source_dir, dir_name)))
        
        label_str = dir_name[-1]
        if label not in ['0', '1']:
            print(f'ERROR: {dir_name} has bad label.')
        label = np.zeros((2))
        label[int(label_str)] = 1.0
        label_list.append(label)

    return np.array(stacked_frames_list)

def train(model, source_dir, nb_epochs=10):
    train_path = os.path.join(source_dir, 'train')
    nb_video_train = sum(os.path.isdir(os.path.join(train_path, i)) for i in os.listdir(train_path))
    
    batch_size = 64

    assert nb_video_train > batch_size
    nb_batch = int(nb_video_train / batch_size)
    for ep in range(nb_epochs):
        for b in range(nb_batch):
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a 3d Convolutional model for Deeofake Detection.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing all training and testing data.',
                        default=None, type=str)
    args = parser.parse_args()
    source_dir = args.source

    b = get_batch(os.path.join(source_dir, 'train'))
    print(b.shape)
