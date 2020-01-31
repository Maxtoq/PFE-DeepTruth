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

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


INPUT_SHAPE = (3, 10, 150, 150)

CHECKPOINT_PATH = './deepfake_detection/checkpoints/3DConv_model_cp.pth'

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
    label_tensor = torch.tensor(label_list, dtype=torch.float).unsqueeze(1)#, dtype=torch.long)

    return batch_tensor, label_tensor
    
def train(model, source_dir, cuda, nb_epochs=10, batch_size=64, mini_batch_size=8, load=False):
    # Get number of training examples
    train_path = os.path.join(source_dir, 'train')
    nb_video_train = sum(os.path.isdir(os.path.join(train_path, i)) for i in os.listdir(train_path))

    # Get number of testing examples
    test_path = os.path.join(source_dir, 'test')
    nb_video_test = sum(os.path.isdir(os.path.join(test_path, i)) for i in os.listdir(test_path))

    # Check for cuda support
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'CPU')

    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Load if necessary
    if load:
        print('Loading model... ', end='')
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('end.')

    # Get number of mini-batches
    nb_mini_batches = int(batch_size / mini_batch_size)
    batch_size = nb_mini_batches * mini_batch_size

    # Get the number of batches of training data
    assert nb_video_train > batch_size, 'Not enough data to train'
    nb_batch_train = int(nb_video_train / batch_size)

    # Get the number of batches of training data
    assert nb_video_test > batch_size, 'Not enough data to test'
    nb_batch_test = int(nb_video_test / batch_size)

    print(f'Training with {nb_video_train} training videos from {train_path},')
    print(f'              {nb_mini_batches} mini-batches of size {mini_batch_size},')
    print(f'              {nb_batch_train} training batches of size {batch_size},')
    print(f'              {nb_video_test} testing videos from {test_path},')
    print(f'              {nb_batch_test} testing batches of size {batch_size}')

    for ep in range(nb_epochs):
        # if ep==1:
        #     break
        print(f'\nEpoch # {ep + 1}')
        batch_loss = 0.0
        for b in tqdm(range(nb_batch_train)):
            # if b==0:
            #     break
            video_batch, label_batch = get_batch(train_path, cuda, batch_size)

            mini_batch_loss = 0.0
            for mb in range(nb_mini_batches):
                video_mini_batch = Variable(video_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size])
                label_mini_batch = Variable(label_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size])

                if cuda:
                    video_mini_batch = video_mini_batch.cuda()
                    label_mini_batch = label_mini_batch.cuda()

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
            if b % 50 == 49:
                print('\n[%d, %5d] loss: %.3f' % (ep + 1, b + 1, batch_loss / (b + 1)))

        # Testing
        with torch.no_grad():
            print('\nTesting...', end='')
            batch_loss = 0.0
            batch_acc = 0.0
            for b in tqdm(range(nb_batch_test)):
                # if b==1:
                #     break
                video_batch, label_batch = get_batch(test_path, cuda, batch_size)

                mini_batch_loss = 0.0
                mini_batch_acc = 0.0
                for mb in range(nb_mini_batches):
                    # if mb==3:
                    #     break
                    video_mini_batch = video_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size]
                    label_mini_batch = label_batch[mb * mini_batch_size:(mb + 1) * mini_batch_size]

                    if cuda:
                        video_mini_batch = video_mini_batch.cuda()
                        label_mini_batch = label_mini_batch.cuda()

                    # Forward prop
                    outputs = model(video_mini_batch)

                    # Compute loss
                    mini_batch_loss += criterion(outputs, label_mini_batch).item()

                    # Compute accuracy on mini-batch
                    mini_batch_acc += int(torch.sum((outputs.round() == label_mini_batch))) / label_mini_batch.size(0)

                batch_loss += mini_batch_loss / nb_mini_batches
                batch_acc += mini_batch_acc / nb_mini_batches
            
            print(f'\nTest loss = {batch_loss / nb_batch_test}.')
            print(f'Test accuracy = {batch_acc / nb_batch_test}.')
    
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a 3d Convolutional model for Deepfake Detection.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing all training and testing data.',
                        type=str, required=True)
    parser.add_argument('-l', '--load', help='load a model from a checkpoint file',
                        action='store_true', default=False)
    args = parser.parse_args()
    source_dir = args.source
    load = args.load

    print('Create model... ', end='')
    model = Conv3DDetector().cuda()
    print('end.')

    train(model, source_dir, True, nb_epochs=10, batch_size=64, load=load)