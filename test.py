from wisard import Wisard, ClusWisard

print("### Input ###")

X = [
    [1,1,1,0,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,0,1,1,1]
]

y = [
    "cold",
    "cold",
    "hot",
    "hot"
]
for i,d in enumerate(X):
    print(y[i],d)


print("\n\n")

print("### WiSARD ###")
addressSize = 3 # tamanho do endereçamento das rams
wsd = Wisard(addressSize)

print("training...")
wsd.train(X,y)

print("classifing...")
out=wsd.classify(X)

print("out:")
for i,d in enumerate(X):
    print(out[i],d)

print("\n\n")

print("### ClusWiSARD ###")
addressSize = 3 # tamanho do endereçamento das rams
minScore = 0.1 # score mínimo do processo de treino
threshold = 10 # limite de treinos por discriminador
discriminatorLimit = 5 # limit de discriminadores por cluster
clus = ClusWisard(addressSize,minScore,threshold,discriminatorLimit)


print("training...")
wsd.train(X,y)

print("classifing...")
out=wsd.classify(X)

print("out:")
for i,d in enumerate(X):
    print(out[i],d)
