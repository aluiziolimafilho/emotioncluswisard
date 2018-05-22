from wisard import ClusWisard
from png import Reader, Writer
import glob

# loading data
print("loading...")

labels_dir="dataset/LABEL"
face_dir="dataset/FACE"
labels = []
aus = []
images = []

for face_filename in glob.iglob(face_dir + '**/*.png', recursive=True):
    label_filename = labels_dir + '/' + face_filename[len(labels_dir):-len(".png")] + ".txt"

    label = open(label_filename, "r")
    lines = label.readlines()
    local_aus = list(map(lambda x: x.split()[0],lines))
    label.close()
    local_label = ":".join(local_aus)
    aus.append(local_aus)
    labels.append(local_label)

    image = open(face_filename, "rb")
    r = Reader(image)
    local_image = []
    for r in r.asDirect()[2]:
        l = list(map(lambda x: 1 if x>0 else 0, r))
        local_image += l
    images.append(local_image)
    image.close()

print("training...")
clus = ClusWisard(160, 0.3, 100, 2)
clus.verbose = True
clus.train(images, labels)

print("classifing...")
out = clus.classify(images)

count = 0
total = float(len(labels))
for i,l in enumerate(out):
    if l == labels[i]:
        count += 1

print("taxa: "+str((count/total)*100)+"% "+str(count)+" of "+str(total))

print("generating mental images...")
mentalImages = clus.getMentalImages()

w = Writer(160,200, greyscale=True)

size=(160,200)

m = 0
for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for d in groupMentalImages:
        lm = max(d)
        if lm > m:
            m = lm

for aClass in mentalImages:
    groupMentalImages = mentalImages[aClass]
    for key,d in enumerate(groupMentalImages):
        mentalImage = [ [ int((1-(d[(r*28)+c]/float(m)))*255) for c in range(size[0])] for r in range(size[1])]
        f = open("mentalImages/aus_mental_"+aClass+"_"+str(key)+".png", "wb")
        w.write(f, mentalImage)
        f.close()
