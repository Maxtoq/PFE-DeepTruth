import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.xception import Xception, decode_predictions
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model


INPUT_SHAPE = (150, 150, 3)

CHECKPOINT_PATH = 'deepfake_detection/checkpoints/model_checkpoint.h5'

class DeepFaceDetector(object):

    def __init__(self, model_checkpoint=None):
        # Initiate model
        self.model = self._init_model(model_checkpoint)
        self.model.summary()

    def _init_model(self, model_checkpoint):
        if model_checkpoint is None:
            xception = Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='avg')
            output = Dense(2, activation='softmax')(xception.layers[-2].output)
            return Model(xception.inputs, output)
        else:
            if os.path.isfile(model_checkpoint):
                return load_model(CHECKPOINT_PATH)
            else:
                print('ERROR: Checkpoint path is not a valid path.')
                exit(0)

    def save_model(self):
        self.model.save(CHECKPOINT_PATH)
        

if __name__ == '__main__':
    detector = DeepFaceDetector(CHECKPOINT_PATH)



    detector.save_model()