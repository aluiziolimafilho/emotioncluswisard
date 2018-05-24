import glob
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import sparse
import cv2

class CKP:
    @staticmethod
    def get_imgs_and_labels(threshold_func=None, labels_dir='FACS/', imgs_dir='cohn-kanade-images/', reshape=None):
        imgs = []
        labels = []
        for filename in glob.iglob(labels_dir + '**/*.txt', recursive=True):
            img_path = Path(
                imgs_dir + filename[len(labels_dir):-len('_facs.txt')] + '.png')

            if img_path.is_file():
                img = np.array(Image.open(img_path).convert('L'))
                if reshape != None:
                    img = cv2.resize(img, reshape)

                if threshold_func == None:
                    imgs.append(img)
                else:
                    imgs.append((img <= threshold_func(img)).astype(np.uint8))

                labels_file = open(str(filename), "r").read()
                temp_labels = [i for i in labels_file.replace(
                    '\n', ' ').split(' ') if i != '' and i != '\n']
                img_labels = []
                for i in range(0, len(temp_labels), 2):
                    img_labels.append([temp_labels[i], temp_labels[i + 1]])
                labels.append(img_labels)

        return [imgs, labels]

    @staticmethod
    def get_imgs_labels_and_emotions(threshold_func=None, labels_dir='FACS/', emotions_dir='Emotion/', imgs_dir='cohn-kanade-images/', reshape=None):
        imgs = []
        labels = []
        emotions = []
        for filename in glob.iglob(emotions_dir + '**/*.txt', recursive=True):
            img_path = Path(
                imgs_dir + filename[len(emotions_dir):-len('_emotion.txt')] + '.png')

            if img_path.is_file():
                img = np.array(Image.open(img_path))
                if reshape != None:
                    img = cv2.resize(img, reshape)

                if threshold_func == None:
                    imgs.append(img)
                else:
                    imgs.append((img <= threshold_func(img)).astype(np.uint8))

                labels_file = open(labels_dir+str(filename[len(emotions_dir):-len('_emotion.txt')] + '_facs.txt'), "r").read()
                temp_labels = [i for i in labels_file.replace(
                    '\n', ' ').split(' ') if i != '' and i != '\n']
                img_labels = []
                for i in range(0, len(temp_labels), 2):
                    img_labels.append([temp_labels[i], temp_labels[i + 1]])
                labels.append(img_labels)

                emotion_file = open(str(filename), "r").read()
                temp_emotion = [i for i in emotion_file.replace(
                    '\n', ' ').split(' ') if i != '' and i != '\n']

                emotions.append(temp_emotion)

        return [imgs, labels, emotions]

    @staticmethod
    def get_all_imgs(threshold_func=None, labels_dir='FACS/', imgs_dir='cohn-kanade-images/', emotions_dir='Emotion/', reshape=None):
        imgs = []
        labels = {}
        emotions = {}
        for filename in glob.iglob(imgs_dir + '**/*.png', recursive=True):
            label_path = Path(
                labels_dir + filename[len(imgs_dir):-len('.png')] + '_facs.txt')

            emotion_path = Path(
                emotions_dir + filename[len(imgs_dir):-len('.png')] + '_emotion.txt'
            )

            img = np.array(Image.open(filename).convert('L'))
            if reshape != None:
                img = cv2.resize(img, reshape)
            img = img.ravel()

            if threshold_func == None:
                imgs.append(img)
            else:
                imgs.append((img <= threshold_func(img)).astype(np.uint8))

            if emotion_path.is_file():
                emo_file = open(str(emotion_path), "r")
                emotions[len(imgs)-1] = int(float(emo_file.readline()))

            if label_path.is_file():
                labels_file = open(str(label_path), "r").read()
                temp_labels = [i for i in labels_file.replace(
                    '\n', ' ').split(' ') if i != '' and i != '\n']
                img_labels = []
                for i in range(0, len(temp_labels), 2):
                    img_labels.append([int(float(temp_labels[i])), int(float(temp_labels[i + 1]))])
                labels[len(imgs) - 1] = img_labels

        return [imgs, labels, emotions]
