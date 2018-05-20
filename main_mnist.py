from wisard import ClusWisard
from loader import MNIST

mndata = MNIST('./')
X,y = mndata.load_training()
X_test,y_test = mndata.load_testing()

clus = ClusWisard(28, 0.01, 100)
print(dir(clus))

print("training...")
clus.train(X, y)

print("classifing...")
out=clus.classify(X_test)

count = 0
for i,oneout in enumerate(out):
    if oneout[1] == str(y_test[i]):
        count += 1
print("pontos: "+str(count)+" de "+str(len(out)))
print("acertos: "+str(float(count)/len(out)*100)+"%")
