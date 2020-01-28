import os
import re
import cv2
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import Conv3DDetector

from torchvision import transforms
from PIL import Image
from tqdm import tqdm


INPUT_SHAPE = (3, 10, 150, 150)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ Sort in 'human' order. """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tf = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    preproc_img = tf(img)

    return preproc_img

def get_stacked_frames(video_dir):
    """ Stack all frames of a video directory and return them in a tensor. """
    frames = []
    file_names = os.listdir(video_dir)
    file_names.sort(key=natural_keys)
    for im in file_names:
        if not im.endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(video_dir, im))
        preproc_img = preprocess_image(img)
        frames.append(preproc_img.unsqueeze(1))

    frame_tensor = torch.cat(frames, dim=1)

    assert frame_tensor.shape == INPUT_SHAPE, 'Wrong shape of frames in dir ' + video_dir

    return frame_tensor

def get_batch(source_dir, cuda, batch_size=64, all_videos=False):
    # Get all video directories and randomly shuffle the list
    video_dirs = os.listdir(source_dir)
    random.shuffle(video_dirs)

    if all_videos:
        batch_size = len(video_dirs)

    stacked_frames_list = []
    label_list = []
    for dir_name in video_dirs[:batch_size]:
        # Get frames of then chosen video in a tensor
        stacked_frames = get_stacked_frames(os.path.join(source_dir, dir_name))
        stacked_frames_list.append(stacked_frames.unsqueeze(0))
        
        # Get label of video
        label_str = dir_name[-1]
        if label_str not in ['0', '1']:
            print(f'ERROR: {dir_name} has bad label.')
            exit(0)
        label_list.append(int(label_str))

    batch_tensor = torch.cat(stacked_frames_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)

    # if cuda:
    #     batch_tensor = batch_tensor.cuda()
    #     label_tensor = label_tensor.cuda()

    return batch_tensor, label_tensor
    
def train(model, source_dir, cuda, nb_epochs=10, batch_size=64, mini_batch_size=20):
    # Get number of training examples
    train_path = os.path.join(source_dir, 'train')
    nb_video_train = sum(os.path.isdir(os.path.join(train_path, i)) for i in os.listdir(train_path))

    # Check for cuda support
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'CPU')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Get number of mini-batches
    nb_mini_batches = int(batch_size / mini_batch_size)
    batch_size = nb_mini_batches * mini_batch_size

    # Get the number of batches of training data
    assert nb_video_train > batch_size, 'Not enough data to train'
    nb_batch = int(nb_video_train / batch_size)

    print(f'Training with {nb_video_train} training videos from {train_path},')
    print(f'              {nb_mini_batches} mini-batches of size {mini_batch_size},')
    print(f'              {nb_batch} batches of size {batch_size}')

    for ep in range(nb_epochs):
        print(f'\nEpoch # {ep}')
        batch_loss = 0.0
        for b in tqdm(range(nb_batch)):
            video_batch, label_batch = get_batch(train_path, cuda, batch_size)

            mini_batch_loss = 0.0
            for mb in range(nb_mini_batches):
                video_mini_batch = video_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size]
                label_mini_batch = label_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size]

                if cuda:
                    video_mini_batch = video_mini_batch.cuda()
                    label_mini_batch = label_mini_batch.cuda()
                    # video_mini_batch = video_mini_batch.to(device)
                    # label_mini_batch = label_mini_batch.to(device)

                # Zero the parameters gradients
                optimizer.zero_grad()

                # Forward prop
                outputs = model(video_mini_batch)

                # Compute loss
                loss = criterion(outputs, label_mini_batch)

                # Backward prop
                loss.backward()
                optimizer.step()

                # Save stats
                mini_batch_loss += loss.item()

            batch_loss += mini_batch_loss / nb_mini_batches
            if b % 10 == 9:
                print('\n[%d, %5d] loss: %.3f' % (ep + 1, b + 1, batch_loss / (b + 1)))
        
        loss_hist = 0.0




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a 3d Convolutional model for Deepfake Detection.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing all training and testing data.',
                        type=str, required=True)
    args = parser.parse_args()
    source_dir = args.source

    print('Create model...')
    model = Conv3DDetector().cuda()

    train(model, source_dir, True, nb_epochs=1, batch_size=256)