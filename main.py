from ckp import CKP
from skimage import filters
from wisardpkg import ClusWisard, Wisard
from png import Writer
from random import random

# things of lime
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

# loading data
print("loading...")
size=(100,100)
imgs_o, aus_o, emotions_o, imgs_raw = CKP.get_imgs_labels_and_emotions(threshold_func=filters.threshold_sauvola, labels_dir='dataset/FACS/',  emotions_dir='dataset/Emotion/', imgs_dir='dataset/cohn-kanade-images/', reshape=size)

# instantiate the ClusWisard
clus = ClusWisard(size[0], 0.3, 1000, 5, verbose=True)
wsd = Wisard(size[0], verbose=True)

print("training...")
clus.train(imgs_o, emotions_o)
wsd.train(imgs_o, emotions_o)

# print("classifing...")
# out=clus.classify(imgs_o)
#
# count = 0
# total = len(imgs_test)
# for i,e in enumerate(out):
#     if e == emotions_o[i]:
#         count += 1
#
# print(" "+str(count)+" of "+str(total))
# print(" "+str((count/total)*100)+"%")

#
# print("generating mental images...")
# w = Writer(size[0],size[1], greyscale=True)
# mentalImages = wsd.getMentalImages()


# for aClass in mentalImages:
#     d = mentalImages[aClass]
#     m=max(d)
#     mentalImage = [ [ int((1.0-(d[(r*size[0])+c]/float(m)))*255) for c in range(size[0])] for r in range(size[1])]
#     f = open("mentalImages/wsd_mental_"+aClass+".png", "wb")
#     w.write(f, mentalImage)
#     f.close()

# mentalImages = clus.getMentalImages()

# for aClass in mentalImages:
#     groupMentalImages = mentalImages[aClass]
#     for key,d in enumerate(groupMentalImages):
#         m=max(d)
#         mentalImage = [ [ int((1.0-(d[(r*size[0])+c]/float(m)))*255) for c in range(size[0])] for r in range(size[1])]
#         f = open("mentalImages/clus_mental_"+aClass+"_"+str(key)+".png", "wb")
#         w.write(f, mentalImage)
#         f.close()
#
def s(e):
    e.sort(key=lambda x: x['class'])
    return [ c['degree'] for c in e ]

def classifier_clus(entries):
    bins = []
    for img in entries:
        bin = (img <= filters.threshold_sauvola(img)).astype(np.uint8)
        bin = bin.ravel()
        bin=[v for i,v in enumerate(bin) if i%3==0]
        bins.append(bin)

    output = clus.classify(bins, returnConfidence=True, returnClassesDegrees=True)
    output = [ s(o['classesDegrees']) for o in output]
    return output

def classifier_wsd(entries):
    bins = []
    for img in entries:
        bin = (img <= filters.threshold_sauvola(img)).astype(np.uint8)
        bin = bin.ravel()
        bin=[v for i,v in enumerate(bin) if i%3==0]
        bins.append(bin)

    output = wsd.classify(bins, returnConfidence=True, returnClassesDegrees=True)
    output = [ s(o['classesDegrees']) for o in output]
    return output

explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

explanation_wsd = explainer.explain_instance(imgs_raw[0],
                                        classifier_fn = classifier_wsd,
                                        top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

explanation_clus = explainer.explain_instance(imgs_raw[0],
                                        classifier_fn = classifier_clus,
                                        top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)


def show(i,e,filename):
    temp, mask = e.get_image_and_mask(i, positive_only=False, num_features=10, hide_rest=False)
    image = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(np.array(image)/255)
    plt.savefig(filename+'.png')

show(0, explanation_wsd, 'wsd')
show(0, explanation_clus, 'clus')
