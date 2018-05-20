from ckp import CKP
from skimage import filters
from wisard import ClusWisard

# loading data
print("loading...")
imgs_o, aus_o, emotions_o = CKP.get_imgs_labels_and_emotions(threshold_func=filters.threshold_sauvola, labels_dir='FACS/',  emotions_dir='Emotion/', imgs_dir='cohn-kanade-images/', reshape=(120,160))

imgs_o = list(map(lambda x: x.ravel(), imgs_o))

imgs = []
aus = []
emotions = []

for i,img in enumerate(imgs_o):
    if len(img) == (120*160):
        imgs.append(img)
        aul = ':'
        for au in aus_o[i]:
            aul += str(float(au[0]))+":"
        aus.append(aul)
        emotions.append(float(emotions_o[i][0]))

def verbose(fase='',index=None, total=None, end=False):
    out = '\r '+fase+': '+str(index)+' of '+str(total)
    print(out,end='')
    if end:
        print('\n')
# instantiate the ClusWisard
clus = ClusWisard(160, 0.01, 100)

print("training...")
clus.train(imgs[:100], aus[:100])

print("classifing...")
out=clus.classify(imgs[280:])

count = 0
total = 0
for i,e in enumerate(aus[280:]):
    aus_e = e.split(':')[1:-1]
    aus_out = out[i].split(':')[1:-1]
    total += len(aus_e)
    for au in aus_e:
        if au in aus_out:
            count += 1
print(" "+str(count)+" of "+str(total))
