
cc=g++
flags=-Wall -O2 -std=c++11
bin=cluswisard

all:
	$(cc) $(flags) -o $(bin) main.cc
