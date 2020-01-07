import cv2
import dlib

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

    def is_me(self, new_pos1, new_pos2, face_im):
        """ Verifies if the given attributes correspond to this face. """
        # Compute the criteria for knowing if the faces are the same
        # (percentage of the frame size)
        crit_x = (self.l[0] / self.frame_size[0]) * 100
        crit_y = (self.l[1] / self.frame_size[1]) * 100

        new_l = (new_pos2.x - new_pos1.x, new_pos2.y - new_pos1.y)
        # Check that the size of the new face corresponds to self
        if new_l[1] < self.l[1] - crit_y or new_l[1] > self.l[1] + crit_y:
            return False
        
        # Check that the position corresponds to self
        if new_pos1.x < self.pos1.x - crit_x or new_pos1.x > self.pos1.x + crit_x:
            return False
        if new_pos1.y < self.pos1.y - crit_y or new_pos1.y > self.pos1.y + crit_y:
            return False

        # The new face corresponds to self, update the attributes
        self.pos1 = new_pos1
        self.pos2 = new_pos2
        self.l = new_l
        self.frames.append(face_im)

        return True


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("face_alignement/shape_predictor_5_face_landmarks.dat")

    persons = []

    vid = cv2.VideoCapture("face_alignement/033.mp4")
    success, image = vid.read()
    frame = 0
    while success:
        print(f"Frame #{frame}")
        dets = detector(image, 0)

        for detection in dets:
            face_im = dlib.get_face_chip(image, sp(image, detection))

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

    for i, p in enumerate(persons):
        print(f"Person {i} with {len(p.frames)} frames")
        if len(p.frames) >= 100:
            print('Saving 100 frames...')
            for j, f in enumerate(p.frames[:5]):
                cv2.imwrite(f'frame{j}.jpg', f)