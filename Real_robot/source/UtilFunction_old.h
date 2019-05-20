
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

class CEstimateCollisionNN
{
public:
	
	ifstream file[4];

    CEstimateCollisionNN();
    ~CEstimateCollisionNN();

	void EstimateCollision(double dInput[], double dOutput[]);
	void SetNNWeight(unsigned int nIndex, double dWeight); ///< ����� �ۼ�
	void SetNNBias(unsigned int nIndex, double dBias); ///< ����� �ۼ�
	void SetBNGamma(unsigned int nIndex, double dWeight); ///< ����� �ۼ�
	void SetBNBeta(unsigned int nIndex, double dWeight); ///< ����� �ۼ�
	double GetNNWeight(unsigned int nIndex); ///< ����� �ۼ�
	double GetNNBias(unsigned int nIndex); ///< ����� �ۼ�

    /********* Notch filter **********/
private:

	double m_dWeight[10000];
	double m_dBias[300];

	void InputNomalization(double dInput[], double dInputN[]);
	double m_dInputN[180];
	double m_dInputNormalMin[180];
	double m_dInputNormalMax[180];

	void CalLayer1(double dInput[], double dOutput[]);
	double m_dLayer1[90];

	void CalLayer2(double dInput[], double dOutput[]);
	double m_dLayer2[90];

	void CalLayer3(double dInput[], double dOutput[]);
	double m_dLayer3[6];

	void CalLayer4(double dInput[], double dOutput[]);
	double m_dLayer4[15];

	void CalLayer5(double dInput[], double dOutput[]);
	double m_dLayer5[15];

	void CalOutput(double dInput[], double dOutput[]);
};
