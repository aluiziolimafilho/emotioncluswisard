from wisardpkg import ClusWisard, Wisard
from loader import MNIST
from png import Writer
import numpy as np
from skimage.color import gray2rgb, rgb2gray, label2rgb

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
y_vec = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test2, y_train, y_test2 = train_test_split(X_vec, y_vec, train_size=0.55)


mndata = MNIST('./')
print("loading...")
X,y = mndata.load_training()
X_test,y_test = mndata.load_testing()


cut=0
Xb = [ [ (1 if i>cut else 0) for i in a] for a in X]
X_testb = [ [ (1 if i>cut else 0) for i in a] for a in X_test]

y = [ str(a) for a in y]
y_test = [ str(a) for a in y_test]

# clus = ClusWisard(28, 0.3, 6000, 5, verbose=True)
clus = Wisard(28)

print("training...")
clus.train(Xb, y)

print("classifing...")
out=clus.classify(X_testb)

count = 0
for i,oneout in enumerate(out):
    if oneout == y_test[i]:
        count += 1
print("pontos: "+str(count)+" de "+str(len(out)))
print("acertos: "+str(float(count)/len(out)*100)+"%")

def classifier_fn(entry):
    cut=0
    o=entry.reshape((-1,1,784))
    bin = [ (1 if (o[0][i]+o[1][i]+o[2][i])>cut else 0) for i,v in enumerate(o[0])]
    out = clus.classify(bin)
    m = sum(lambda x: out[x], out)
    return [ float(i)/m for i in out ]


explanation = explainer.explain_instance(X_test2[0],
                                        classifier_fn = classifier_fn,
                                        top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

# print("generating mental images...")
# mentalImages = clus.getMentalImages()
#
# w = Writer(28,28, greyscale=True)
#
# m = 0
# # for aClass in mentalImages:
# #     groupMentalImages = mentalImages[aClass]
# #     for d in groupMentalImages:
# #         lm = max(d)
# #         if lm > m:
# #             m = lm
# #
# # for aClass in mentalImages:
# #     groupMentalImages = mentalImages[aClass]
# #     for key,d in enumerate(groupMentalImages):
# #         mentalImage = [ [ int((d[(r*28)+c]/float(m))*255) for c in range(28)] for r in range(28)]
# #         f = open("mentalImages/mental_"+aClass+"_"+str(key)+".png", "wb")
# #         w.write(f, mentalImage)
# #         f.close()
#
# for aClass in mentalImages:
#     d = mentalImages[aClass]
#     lm = max(d)
#     if lm > m:
#         m = lm
#
# for aClass in mentalImages:
#     d = mentalImages[aClass]
#     mentalImage = [ [ int((d[(r*28)+c]/float(m))*255) for c in range(28)] for r in range(28)]
#     f = open("mentalImages/mental_"+aClass+".png", "wb")
#     w.write(f, mentalImage)
#     f.close()
