#include <string>
#include <iostream>
#include "mnist.cc"
#include "cluswisard.cc"

using namespace std;

int main(int argsize, char* args[]){

  cout << endl;
  ClusWisard cluswisard(3,0.5,5);

  return 0;
}
