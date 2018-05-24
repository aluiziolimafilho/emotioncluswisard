from wisard import ClusWisard, Wisard
from loader import MNIST
from png import Writer

mndata = MNIST('./')
print("loading...")
X,y = mndata.load_training()
X_test,y_test = mndata.load_testing()


cut=0
X = [ [ (1 if i>cut else 0) for i in a] for a in X]
X_test = [ [ (1 if i>cut else 0) for i in a] for a in X_test]


w = Writer(28,28, greyscale=True)
mentalImage = [ [ 255 if X[0][(r*28)+c] > 0 else 0 for c in range(28)] for r in range(28)]
f = open("mentalImages/mental_test.png", "wb")
w.write(f, mentalImage)
f.close()

y = [ str(a) for a in y]
y_test = [ str(a) for a in y_test]

clus = ClusWisard(28, 0.3, 6000, 5)
# clus = Wisard(4)
clus.verbose = True

print("training...")
clus.train(X[:1], y[:1])

print("classifing...")
out=clus.classify(X_test)

count = 0
for i,oneout in enumerate(out):
    if oneout == y_test[i]:
        count += 1
print("pontos: "+str(count)+" de "+str(len(out)))
print("acertos: "+str(float(count)/len(out)*100)+"%")

print("generating mental images...")
mentalImages = clus.getMentalImages()

w = Writer(28,28, greyscale=True)

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
        mentalImage = [ [ int((d[(r*28)+c]/float(m))*255) for c in range(28)] for r in range(28)]
        f = open("mentalImages/mental_"+aClass+"_"+str(key)+".png", "wb")
        w.write(f, mentalImage)
        f.close()
