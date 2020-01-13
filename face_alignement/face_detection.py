import os
import re
import cv2
import dlib
import argparse


SHAPE_PREDICTOR_FILE = "face_alignement/shape_predictor_5_face_landmarks.dat"


class Face(object):

    """ Detected face in the video. """

    def __init__(self, pos1, pos2, face_im, frame_size):
        # Frame size tuple
        self.frame_size = frame_size
        # Position in the video
        self.pos1 = pos1
        self.pos2 = pos2

        # Length of the face
        self.l = (pos2.x - pos1.x, pos2.y - pos1.y)

        self.frames = [face_im]

        self.frame = 0

    def is_me(self, new_pos1, new_pos2, face_im):
        """ Verifies if the given attributes correspond to this face. """
        # Compute the criteria for knowing if the faces are the same
        # (percentage of the frame size)
        crit_x = (self.l[0] / self.frame_size[0]) * 1000
        crit_y = (self.l[1] / self.frame_size[1]) * 1000
        print(crit_x, crit_y)

        new_l = (new_pos2.x - new_pos1.x, new_pos2.y - new_pos1.y)

        # Check that the size of the new face corresponds to self
        if new_l[1] < self.l[1] - crit_y or new_l[1] > self.l[1] + crit_y:
            self.frame += 1
            return False
        
        # Check that the position corresponds to self
        if new_pos1.x < self.pos1.x - crit_x or new_pos1.x > self.pos1.x + crit_x:
            self.frame += 1
            return False
        if new_pos1.y < self.pos1.y - crit_y or new_pos1.y > self.pos1.y + crit_y:
            self.frame += 1
            return False

        # The new face corresponds to self, update the attributes
        self.pos1 = new_pos1
        self.pos2 = new_pos2
        self.l = new_l
        self.frames.append(face_im)
        self.frame =0
        return True


def align_video(video_file, detector, shape_predictor, output_dir, nb_frames=100):
    persons = []

    vid = cv2.VideoCapture(video_file)
    success, image = vid.read()
    frame = 0
    while success:
        print(f"Frame #{frame}")
        dets = detector(image, 0)

        for detection in dets:
            # Align face and crop
            face_im = dlib.get_face_chip(image, shape_predictor(image, detection))
            # Assign face to person
            is_known = False
            for p in persons:
                if p.is_me(detection.tl_corner(), detection.br_corner(), face_im):
                    is_known = True
            if not is_known:
                persons.append(Face(
                                detection.tl_corner(), 
                                detection.br_corner(), 
                                face_im, image.shape[:-1]
                            ))

        success, image = vid.read()
        frame += 1

        print(len(persons))

    count = 0
    for i, p in enumerate(persons):
        print(f"Person {i} with {len(p.frames)} frames")
        if len(p.frames) >= 100:
            count += 1
            video_name = video_file.replace('\\', '_')[3:-4] + '_face' + str(i)
            # Create dir
            dir_path = os.path.join(output_dir, video_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            print(f'Saving frames in {dir_path}...')
            # Save pictures TO MODIFY WITH nb_frames
            for j, f in enumerate(p.frames):
                cv2.imwrite(os.path.join(dir_path, f'frame{j}.jpg'), f)

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
        print('Source directory (-s) must be specified. Enter \'python face_detection.py -h\' to prompt help.')
        exit(0)
    if output_dir is None:
        print('Output directory (-o) must be specified. Enter \'python face_detection.py -h\' to prompt help.')
        exit(0)

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_FILE)

    nb_files = 0
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        nb_files += len(filenames)
    count_files = 0
    for (dirpath, _, filenames) in os.walk(source_dir):
        for f in filenames:
            count_files += 1
            print(f'\nFile {count_files}/{nb_files}')
            video_file = os.path.join(dirpath, f)
            if video_file[-4:] != '.mp4':
                print(f'\'{video_file}\' is not a video file.')
                continue
            print('Analysing', video_file)
            count_faces = align_video(video_file, detector, sp, output_dir)
            if count_faces != 1:
                print(f'WARNING : Video \'{video_file}\' has {count_faces} faces.')