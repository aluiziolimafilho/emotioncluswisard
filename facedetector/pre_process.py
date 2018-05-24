from ckp import CKP
from skimage import filters
import numpy as np
import glob
from detector import Detector
from PIL import Image
import random

FACE_NUM_RAMS = 1000
EYE_NUM_RAMS = 5
MOUTH_NUM_RAMS = 15
# NUM = 20
# THRESHOLD_TRAINING_FILES = 0.9

labels_dir = "data/FACS/"
imgs_dir = "data/cohn-kanade-images/"
emotions_in_dir="dataset/Emotion/"
emotions_out_dir="dataset/emotions-process/"
out_dir = "data/out/"
eye_dir = "data/eyes/"
mouth_dir = "data/mouth/"
faces_dir = "../dataset/faces/"

print("== DATASET ==")
imgs, labels, emotions = CKP.get_all_imgs(threshold_func=filters.threshold_sauvola, labels_dir=labels_dir, imgs_dir=imgs_dir, emotions_dir=emotions_in_dir, reshape=(400,400))
# indexes = random.sample(range(len(imgs)), NUM)

print("== DETECTOR FILES ==")
face_training_files = []
for filename in glob.iglob(faces_dir + '*.png'):
    face_training_files.append(filename)

# mouth_training_files = []
# for filename in glob.iglob(mouth_dir + '*.png'):
#     mouth_training_files.append(filename)
#
# eye_training_files = []
# for filename in glob.iglob(eye_dir + '*.png'):
#     eye_training_files.append(filename)
print(np.shape(np.array(Image.open(face_training_files[0]))))

print("== TRAINING ==")
face_detector = Detector(name="faces", data_shape=np.shape(np.array(Image.open(face_training_files[0]))), number_of_rams=FACE_NUM_RAMS)
face_detector.train(files=face_training_files)

# eye_detector = Detector(name="eye", data_shape=np.shape(np.array(Image.open(eye_training_files[0]))), number_of_rams=EYE_NUM_RAMS)
# eye_detector.train(files=eye_training_files)
#
# mouth_detector = Detector(name="mouth", data_shape=np.shape(np.array(Image.open(mouth_training_files[0]))), number_of_rams=MOUTH_NUM_RAMS)
# mouth_detector.train(files=mouth_training_files)

print("== DETECTOR ==")
for i in range(len(imgs)):
    face_img, face_pos = face_detector.classify(imgs[i], step_size=10)
    if len(face_img) > 0:
        Image.fromarray(np.multiply(face_img, 255)).save("{}FACE/{}.png".format(out_dir, i))
        # eye_img, eye_pos = eye_detector.classify(face_img, step_size=10)
        # Image.fromarray(np.multiply(eye_img, 255)).save("{}EYE/{}.png".format(out_dir, i))
        # mouth_img, mouth_pos = mouth_detector.classify(face_img, step_size=10)
        # Image.fromarray(np.multiply(mouth_img, 255)).save("{}MOUTH/{}.png".format(out_dir, i))
        file = open("{}LABEL/{}.txt".format(out_dir, i),'w')
        for elem in labels[i]:
            file.write("{} {}\n".format(int(float(elem[0])), int(float(elem[1]))))
        file.close()
