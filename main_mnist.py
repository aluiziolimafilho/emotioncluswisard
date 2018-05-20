from wisard import ClusWisard, Wisard
from loader import MNIST

mndata = MNIST('./')
print("loading...")
X,y = mndata.load_training()
X_test,y_test = mndata.load_testing()

cut=0
X = [ [ (1 if i>cut else 0) for i in a] for a in X]
X_test = [ [ (1 if i>cut else 0) for i in a] for a in X_test]

y = [ str(a) for a in y]
y_test = [ str(a) for a in y_test]

# clus = ClusWisard(28, 0.01, 100)
clus = Wisard(4)
clus.verbose = False

print("training...")
clus.train(X, y)

print("classifing...")
out=clus.classify(X_test)

count = 0
for i,oneout in enumerate(out):
    if oneout == y_test[i]:
        count += 1
print("pontos: "+str(count)+" de "+str(len(out)))
print("acertos: "+str(float(count)/len(out)*100)+"%")
