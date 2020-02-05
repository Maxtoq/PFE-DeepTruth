import os
import re
import cv2
import time
import torch
import shutil
import argparse

from model import Conv3DDetector

from torchvision import transforms
from PIL import Image
from tqdm import tqdm


INPUT_SHAPE = (3, 10, 150, 150)
MIN_NB_FRAMES = 10

CHECKPOINT_PATH = './deepfake_detection/checkpoints/3DConv_model_cp.pth'


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ Sort in 'human' order. """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_frame_id(frame_name):
    return int(re.split(r'(\d+)', frame_name)[1])

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

def get_stacked_frames(video_dir, frame_list):
    """ Stack all frames of a list and return them in a tensor. """
    frames = []
    for im in frame_list:
        if not im.endswith('.jpg'):
            continue
        img = cv2.imread(os.path.join(video_dir, im))
        preproc_img = preprocess_image(img)
        frames.append(preproc_img.unsqueeze(1))

    frame_tensor = torch.cat(frames, dim=1)

    assert frame_tensor.shape == INPUT_SHAPE, 'Wrong shape of frames in dir ' + video_dir

    return frame_tensor.unsqueeze(0)

def test_model(video_dir, model, nb_test, cuda):
    # Get list of frames and sort it the 'human' way
    list_frames = os.listdir(video_dir)
    list_frames.sort(key=natural_keys)
  
    if len(list_frames) < MIN_NB_FRAMES:
        return None

    max_test = int(len(list_frames) / INPUT_SHAPE[1])
    if nb_test is None or nb_test > max_test:
        nb_test = max_test

    preds = []
    for i in range(nb_test):
        sample_frames = []
        while len(list_frames) >= INPUT_SHAPE[1] - len(sample_frames):
            next_frame = list_frames.pop(0)
            if len(sample_frames) > 0 and get_frame_id(next_frame) == get_frame_id(sample_frames[-1]) + 1:
                sample_frames.append(next_frame)
                if len(sample_frames) == INPUT_SHAPE[1]:
                    break
            elif len(list_frames) >= INPUT_SHAPE[1] - 1:
                sample_frames = [next_frame]
            else:
                break
        
        if len(sample_frames) != INPUT_SHAPE[1]:
            print('Bad number of frames.')
            continue
        # Use them to predict class
        frame_tensor = get_stacked_frames(video_dir, sample_frames)
        if cuda:
            frame_tensor = frame_tensor.cuda()

        pred = model(frame_tensor)

        preds.append(pred)
    
    return sum(preds) / len(preds)

def wait_for_videos(input_dir, model, max_test, cuda):
    print('Waiting for videos to predict...')
    while True:
        dir_list = os.listdir(input_dir)

        for vid_dir in dir_list:
            print('Prediction for new video', vid_dir)
            time.sleep(0.1)
            video_path = os.path.join(input_dir, vid_dir)
            pred = test_model(video_path, model, max_test, cuda)

            if pred is None:
                continue
            
            print(pred)

            # Remove dir
            shutil.rmtree(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a 3d Convolutional model for Deepfake Detection.')
    parser.add_argument('-i', '--input', help='-i <input_dir_path> : input directory containing all video directories.',
                        type=str, required=True)
    parser.add_argument('-mi', '--model_input', help='model path.',
                        type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--max_test', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    input_dir = args.input
    model_path = args.model_input
    max_test = args.max_test
    cuda = args.cuda

    print('Create model... ')
    model = Conv3DDetector()
    print('Loading model... ')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model = model.cuda()
    model.eval()
    torch.set_grad_enabled(False)
    print('done.')

    wait_for_videos(input_dir, model, max_test, cuda)