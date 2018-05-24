from wisard.discriminator import Discriminator
from utils import random_indexes
import numpy as np
from PIL import Image
from skimage import filters
import cv2

class Detector(Discriminator):
    def __init__(self, name, data_shape, number_of_rams):
        super().__init__(name, random_indexes(data_shape[0]*data_shape[1], number_of_rams))
        # super().__init__(name, np.arange(data_shape[0]*data_shape[1]).reshape((number_of_rams, int((data_shape[0]*data_shape[1])/number_of_rams))))
        self.data_shape = data_shape

    def train(self, files, bleaching=True):
        i = 0
        for filename in files:
            img = np.array(Image.open(filename).convert('LA'))
            if len(np.shape(img)) > 2:
                img = img[:, :, 0]
            img = (img <= filters.threshold_sauvola(img)).astype(int)
            i += 1
            super().train(img.ravel(), bleaching)

    def classify(self, image, step_size=1):
        max_score = 0
        best_fit = []
        pos = None
        for y in range(self.data_shape[0], image.shape[0], step_size):
            for x in range(self.data_shape[1], image.shape[1], step_size):
                crop_img = image[y - self.data_shape[0]:y, x - self.data_shape[1]:x]
                # binary_img = (crop_img <= filters.threshold_sauvola(crop_img)).astype(int)
                if len(np.shape(crop_img)) > 2:
                    crop_img = crop_img[:, :, 0]
                score = super().classify(crop_img.ravel())
                if(score > max_score):
                    max_score = score
                    best_fit = crop_img.copy()
                    pos = [y - self.data_shape[0],y,x - self.data_shape[1],x]

        return best_fit, pos
