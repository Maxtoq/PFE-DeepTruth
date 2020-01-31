import os
import re
import cv2
import dlib
import math
import argparse
import numpy as np
import pandas as pd

SHAPE_PREDICTOR_FILE = "face_alignement/shape_predictor_5_face_landmarks.dat"


class Face(object):

    """ Detected face in the video. """

    def __init__(self, pos1, pos2, face_im, frame_size, frame_num):
        # Frame size tuple
        self.frame_size = frame_size
        # Position in the video
        self.pos1 = pos1
        self.pos2 = pos2

        # Length of the face
        self.l = (pos2.x - pos1.x, pos2.y - pos1.y)

        # Dict containing all frames of face
        self.frames = { frame_num: face_im }

    def get_dist(self, new_pos1, frame_num):
        if frame_num in self.frames:
            return 10000
        return math.sqrt((self.pos1.x - new_pos1.x)**2 + (self.pos1.y - new_pos1.y)**2)
    
    def add_frame(self, new_pos1, new_pos2, face_im, frame_num):
        if self.get_dist(new_pos1, frame_num) > self.l[1] * 2:
            return False
        self.pos1 = new_pos1
        self.pos2 = new_pos2
        self.l = (new_pos2.x - new_pos1.x, new_pos2.y - new_pos1.y)
        self.frames[frame_num] = face_im
        return True


def align_video(video_file, detector, shape_predictor, output_dir, label, nb_frames=100):
    persons = []

    vid_raw = cv2.VideoCapture(video_file)
    success, image_raw = vid_raw.read()

    min_height = None
    frame_num = 0
    while success:
        #print(f"Frame #{frame_num}")
        dets = detector(image_raw, 0)

        for detection in dets:
            if min_height is None:
                min_height = detection.height() * (2 / 3)
            if detection.height() < min_height:
                break
            elif detection.height() > min_height * (5 / 3):
                min_height = min_height * (5 / 3)

            # Align face and crop
            face_im_raw = dlib.get_face_chip(image_raw, shape_predictor(image_raw, detection))
            # Assign face to person
            is_known = False
            dist = []
            # Compute distances to all persons
            for p in persons_raw:
                dist.append(p.get_dist(detection.tl_corner(), frame_num))
            # Add frame to closest
            if len(dist) > 0 and min(dist) < 10000:
                is_known = persons[np.argmin(dist)].add_frame(detection.tl_corner(), detection.br_corner(), face_im, frame_num)
                
            if not is_known:
                persons_raw.append(Face(
                                detection.tl_corner(), 
                                detection.br_corner(), 
                                face_im_raw, 
                                image_raw.shape[:-1],
                                frame_num
                            ))

        success, image_raw = vid_raw.read()
        frame_num += 1

    count = 0
    for i, p in enumerate(persons_raw):
        print(f"Person {i} with {len(p.frames)} frames")
        if len(p.frames) >= 80:
            count += 1
            video_name = video_file.replace('\\', '_')[3:-4] + '_face' + str(i) + "_{}".format(label)
            # Create dir
            dir_path = os.path.join(output_dir, video_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            print(f'Saving frames in {dir_path}...')
            # Save pictures TO MODIFY WITH nb_frames
            for k, f in p.frames.items():
                cv2.imwrite(os.path.join(dir_path, f'frame{k}.jpg'), f)

    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face alignement on multiple input videos.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing all input videos',
                        default=None, type=str)
    parser.add_argument('-o', '--output', help='-o <output_dir_path> : output directory',
                        default=None, type=str)
    args = parser.parse_args()
    source_dir = args.source
    output_dir = args.output

    if source_dir is None:
        print('ERROR: Source directory (-s) must be specified. Enter \'python face_detection.py -h\' to prompt help.')
        exit(0)
    if output_dir is None:
        print('ERROR: Output directory (-o) must be specified. Enter \'python face_detection.py -h\' to prompt help.')
        exit(0)

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_FILE)
    metadat_f = os.path.join(source_dir, "metadata.json")
    metadata = pd.read_json(metadat_f).T

    nb_files = 0
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        nb_files += len(filenames)
    count_files = 0
    for (dirpath, _, filenames) in os.walk(source_dir):
        for f in filenames:
            count_files += 1
            print(f'\nFile {count_files}/{nb_files}')
            video_file = os.path.join(dirpath, f)
            if video_file[-4:] in ['.mp4', '.avi']:
                print(f'\'{video_file}\' is not a video file.')
                continue
            print('Analysing', video_file)
            if metadata.loc[f,"label"] == "REAL":
             label = 0
             output_dir = os.path.join(output_dir, 'originals')
            elif metadata.loc[f,"label"] =="FAKE":
             label=1 
             output_dir = os.path.join(output_dir, 'manipulated')
            count_faces = align_video(video_file, detector, sp, output_dir, label)
            if count_faces != 1:
                print(f'WARNING : Video \'{video_file}\' has {count_faces} faces.')