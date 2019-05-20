
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>

class CEstimateCollisionNN
{
public:
	
	ifstream file[7];

    CEstimateCollisionNN();
    ~CEstimateCollisionNN();

	void EstimateCollision(double dInput[], double dOutput[]);
	void SetNNWeight(); ///< ����� �ۼ�
	void SetNNBias(); ///< ����� �ۼ�
	void SetBNGamma(); ///< ����� �ۼ�
	void SetBNBeta(); ///< ����� �ۼ�
	void SetBNMean(); ///< ����� �ۼ�
	void SetBNVariance(); ///< ����� �ۼ�
	void SetInputNormalMinMax(); ///< ����� �ۼ�

    /********* Notch filter **********/
private:

	double m_dWeight[10000];
	double m_dBias[300];
	double m_dGamma[300];
	double m_dBeta[300];
	double m_dMean[300];
	double m_dVariance[300];

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
