import glob
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import sparse


class CKP:
    """docstring for CKP."""

    def __init__(self, labels_dir='FACS/', imgs_dir='cohn-kanade-images/'):
        self.labels_dir = labels_dir
        self.imgs_dir = imgs_dir
        self.imgs = []
        self.labels = {}

    def read_data(self, all_imgs=False, threshold_func=None):
        for filename in glob.iglob(self.imgs_dir + '**/*.png', recursive=True):
            label_path = Path(
                self.labels_dir + filename[len(self.imgs_dir):-len('.png')] + '_facs.txt')

            if label_path.is_file() or all_imgs:
                img = np.array(Image.open(filename))
                if threshold_func == None:
                    self.imgs.append(img.ravel())
                else:
                    self.imgs.append(np.multiply(
                        (img > threshold_func(img)), 1).ravel())

            if label_path.is_file() and not all_imgs:
                labels_file = open(str(label_path), "r").read()
                temp_labels = [i for i in labels_file.replace(
                    '\n', ' ').split(' ') if i != '' and i != '\n']
                img_labels = []
                for i in range(0, len(temp_labels), 2):
                    img_labels.append([temp_labels[i], temp_labels[i + 1]])
                self.labels[len(self.imgs) - 1] = img_labels
