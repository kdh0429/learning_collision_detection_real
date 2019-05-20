#include "UtilFunction.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

int main(int argc, char* argv[])
{
    double Input[180];
    double Output[2];
    CEstimateCollisionNN NN;

    fstream classFile("./TestingDivide/Testing_raw_data_5.csv");
    string line;

    string data_string;
    
    ofstream writeFile("Result.txt");

    int line_idx = 0;
    int idx = 0;
    while (getline(classFile, line,'\n')) // there is input overload classfile
    {
        istringstream iss(line);
        while (getline(iss, data_string,','))
        {
            if (idx < 180)
            {
            Input[idx] = stod(data_string);
            idx++;
            }
            else if (idx == 180)
            {
                idx++;
            }
            else if (idx == 181)
            {
                idx = 0;
                line_idx++;
                //if(line_idx == 1)
                    NN.EstimateCollision(Input, Output);
                writeFile<<Output[0]<<'\n';
            }
        }

    }
	writeFile.close();
    return 0;
}
