from ckp import CKP
from skimage import filters
from wisard import ClusWisard
from png import Writer
from random import random
# loading data
print("loading...")
size=(100,100)
imgs_o, aus_o, emotions_o = CKP.get_all_imgs(threshold_func=filters.threshold_sauvola, labels_dir='dataset/FACS/',  emotions_dir='dataset/Emotion/', imgs_dir='dataset/cohn-kanade-images/', reshape=size)

imgs_train = []
imgs_test = []

aus_train = []
aus_test = []

emotions_train = {}
emotions_test = []

for i,img in enumerate(imgs_o):
    if i in emotions_o:
        if random() < 0.1:
            imgs_test.append(img)
            emotions_test.append(str(emotions_o[i]))
            if i in aus_o:
                aus_test.append(aus_o[i])
        else:
            imgs_train.append(img)
            k = len(imgs_train)-1
            emotions_train[k] = str(emotions_o[i])
            if i in aus_o:
                aus_train.append(aus_o[i])

    else:
        imgs_train.append(img)
        if i in aus_o:
            aus_train.append(aus_o[i])

# instantiate the ClusWisard
clus = ClusWisard(size[0], 0.3, 1000, 5)
clus.verbose=True

print("training...")
clus.train(imgs_train, emotions_train)

print("classifing...")
out=clus.classify(imgs_test)

count = 0
total = len(imgs_test)
for i,e in enumerate(out):
    if e == emotions_test[i]:
        count += 1

print(" "+str(count)+" of "+str(total))
print(" "+str((count/total)*100)+"%")


print("generating mental images...")
mentalImages = clus.getMentalImages()


w = Writer(size[0],size[1], greyscale=True)
m=0
for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for d in groupMentalImages:
        lm = max(d)
        if lm > m:
            m = lm

for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for key,d in enumerate(groupMentalImages):
        mentalImage = [ [ int((d[(r*size[0])+c]/float(m))*255) for c in range(size[0])] for r in range(size[1])]
        f = open("mentalImages/e_mental_"+aClass+"_"+str(key)+".png", "wb")
        w.write(f, mentalImage)
        f.close()
