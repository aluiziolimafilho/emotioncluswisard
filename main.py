from ckp import CKP
from skimage import filters
from wisardpkg.cluswisard import ClusWisard

# loading data
print("loading...")
imgs_o, aus_o, emotions_o = CKP.get_imgs_labels_and_emotions(threshold_func=filters.threshold_sauvola, labels_dir='FACS/',  emotions_dir='Emotion/', imgs_dir='cohn-kanade-images/', reshape=(400,400))

imgs_o = list(map(lambda x: x.ravel(), imgs_o))

imgs = []
aus = []
emotions = []

for i,img in enumerate(imgs_o):
    if len(img) == (400*400):
        imgs.append(img)
        aus.append(aus_o[i])
        emotions.append(emotions_o[i])
# instantiate the ClusWisard
clus = ClusWisard(addressSize=100, minScore=0.01, threshold=100)

print("training...")
clus.trainall(imgs[:250], emotions[:250])

print("classifing...")
out=clus.classifyall(imgs[250:])

count = 0
for i,e in enumerate(emotions[250:]):
    if e == out[i]:
        count += 1
print(" "+str(count)+" of "+str(len(out)))
