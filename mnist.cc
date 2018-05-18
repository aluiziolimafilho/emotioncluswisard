#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <tuple>

using namespace std;

class DataSet{
public:
  DataSet(string fileName){
    vector<vector<string>> lines = readLines(fileName);
    insertItems(lines);
  }

  tuple<vector<int>,string>& operator[](int index){
    tuple<vector<int>,string>* entry = new tuple<vector<int>,string>;
    *entry = make_tuple(this->images[index],this->labels[index]);
    return *entry;
  }

  const vector<vector<int>>& getImages() const{
    return this->images;
  }

  const vector<string>& getLabels() const{
    return labels;
  }

private:
  vector<vector<int>> images;
  vector<string> labels;

  vector<string>& splitString(string input, char delimiter){
    stringstream ss(input);
    string item;
    vector<string>* items = new vector<string>();
    while(getline(ss,item, delimiter)){
      items->push_back(item);
    }
    return *items;
  }

  vector<vector<string>>& readLines(string fileName){
    ifstream file(fileName);
    string line;
    getline ( file, line );
    vector<vector<string>>* lines = new vector<vector<string>>;
    while ( file.good() )
    {
      getline (file, line);
      if(line.size()>0)
        lines->push_back(this->splitString(line, ','));
    }
    file.close();
    return *lines;
  }

  void insertItems(vector<vector<string>>& lines){
    int size = splitString(lines[0][0], ' ').size();
    this->images = vector<vector<int>>(lines.size(), vector<int>(size));
    this->labels = vector<string>(lines.size());

    for(unsigned int j=0; j<lines.size(); j++){
      cout << "\rreading " << j+1 << " of " << lines.size();
      vector<string> pixels = splitString(lines[j][0], ' ');
      for(unsigned int i=0; i<pixels.size(); i++){
        this->images[j][i] = stoi(pixels[i]);
      }
      this->labels[j] = lines[j][1];
    }
    cout << "\r" << endl;
  }
};

class MNIST{
public:
  MNIST(string trainFileName, string testFileName):
    train(DataSet(trainFileName)),
    test(DataSet(testFileName)){}

  DataSet train;
  DataSet test;
};
