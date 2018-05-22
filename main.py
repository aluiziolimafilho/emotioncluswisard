from ckp import CKP
from skimage import filters
from wisard import ClusWisard
from png import Writer

# loading data
print("loading...")
size=(400,400)
imgs_o, aus_o, emotions_o = CKP.get_imgs_labels_and_emotions(threshold_func=filters.threshold_sauvola, labels_dir='FACS/',  emotions_dir='Emotion/', imgs_dir='cohn-kanade-images/', reshape=size)

imgs_o = list(map(lambda x: x.ravel(), imgs_o))

imgs = []
aus = []
emotions = []

for i,img in enumerate(imgs_o):
    if len(img) == (size[0]*size[1]):
        imgs.append(img)
        aul = ':'
        for au in aus_o[i]:
            aul += str(float(au[0]))+":"
        aus.append(aul)
        emotions.append(str(int(float(emotions_o[i][0]))))

# instantiate the ClusWisard
clus = ClusWisard(size[0], 0.5, 100, 2)
clus.verbose=True

print("training...")
clus.train(imgs[:1], emotions[:1])

print("classifing...")
out=clus.classify(imgs)
#
# count = 0
# total = 0
# for i,e in enumerate(aus[200:]):
#     aus_e = e.split(':')[1:-1]
#     aus_out = out[i].split(':')[1:-1]
#     total += len(aus_e)
#     for au in aus_e:
#         if au in aus_out:
#             count += 1

count = 0
total = len(emotions)
for i,e in enumerate(emotions):
    if e == out[i]:
        count += 1

print(" "+str(count)+" of "+str(total))
print(" "+str((count/total)*100)+"%")


print("generating mental images...")
mentalImages = clus.getMentalImages()


w = Writer(size[0],size[1], greyscale=True)
out = [ [ int(imgs[0][(r*28)+c]*255) for c in range(size[0])] for r in range(size[1])]
f = open("mentalImages/test.png", "wb")
w.write(f, out)
f.close()
m = 0
for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for key in groupMentalImages:
        d = groupMentalImages[key]
        lm = max(d)
        if lm > m:
            m = lm

for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for key in groupMentalImages:
        d = groupMentalImages[key]
        mentalImage = [ [ int((d[(r*28)+c]/float(m))*255) for c in range(size[0])] for r in range(size[1])]
        f = open("mentalImages/e_mental_"+aClass+"_"+str(key)+".png", "wb")
        w.write(f, mentalImage)
        f.close()
