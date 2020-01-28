import os
import re
import random
import shutil
import argparse

from tqdm import tqdm


NB_FRAMES = 10


def cut_video_name(name):
    id_manip = name.find('manipulated')
    if id_manip != -1:
        return name[id_manip:]
    id_orig = name.find('original')
    if id_orig != -1:
        return name[id_orig:]
    return name

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ Sort in 'human' order. """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_frame_id(frame_name):
    return int(re.split(r'(\d+)', frame_name)[1])

def init_output_dir(output_dir):
    if not os.path.exists(os.path.join(output_dir, 'train')):
        os.makedirs(os.path.join(output_dir, 'train'))
    if not os.path.exists(os.path.join(output_dir, 'test')):
        os.makedirs(os.path.join(output_dir, 'test'))

def check_source_dir(source_dir):
    if os.listdir(source_dir) != ['manipulated', 'originals']:
        print('ERROR: Source directory must contain only two directories: \'manipulated\' and \'originals\'.')
        exit(0)

def create_data(source_dir, output_dir, label, train_split=0.8, nb_sample=15):
    for video_dir in tqdm(os.listdir(source_dir)):
        # Determine if this video will be for training of testing
        if random.random() < train_split:
            type_data = 'train'
        else:
            type_data = 'test'

        # Get list of frames and sort it the 'human' way
        list_frames = os.listdir(os.path.join(source_dir, video_dir))
        list_frames.sort(key=natural_keys)

        # Delete first and last 15 to avoid badly cropped images
        list_frames = list_frames[15:-15]

        for i in range(nb_sample):
            sample_frames = []
            while len(list_frames) >= NB_FRAMES - len(sample_frames):
                next_frame = list_frames.pop(0)
                if len(sample_frames) > 0 and get_frame_id(next_frame) == get_frame_id(sample_frames[-1]) + 1:
                    sample_frames.append(next_frame)
                    if len(sample_frames) == NB_FRAMES:
                        break
                else:
                    sample_frames = [next_frame]
            # Save them in a new dir in output_dir with label in dir_path
            dir_name = os.path.join(output_dir, type_data, cut_video_name(video_dir) + f'_{str(i)}_{label}')
            # Create dir
            os.makedirs(dir_name)
            # Save all frames
            for f in sample_frames:
                shutil.copyfile(os.path.join(source_dir, video_dir, f), os.path.join(dir_name, f))
            
            if len(list_frames) < NB_FRAMES:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training and testing a 3d Convolutional model.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing \'originals\' and \'manipulated\' directories',
                        default=None, type=str)
    parser.add_argument('-o', '--output', help='-o <output_dir_path> : output directory (must be empty)',
                        default=None, type=str)
    args = parser.parse_args()
    source_dir = args.source
    output_dir = args.output

    check_source_dir(source_dir)
    
    init_output_dir(output_dir)

    print('Preparing original data...')
    create_data(os.path.join(source_dir, 'originals'), output_dir, 0)
    print('\n\nPreparing manipulated data...')
    create_data(os.path.join(source_dir, 'manipulated'), output_dir, 1, nb_sample=3)