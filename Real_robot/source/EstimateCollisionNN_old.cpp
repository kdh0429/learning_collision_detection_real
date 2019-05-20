
#include "UtilFunction.h"

CEstimateCollisionNN::CEstimateCollisionNN()
{
	for (unsigned int i=0;i<10000;i++){
		m_dWeight[i] = 0.0;
	}
	for (unsigned int i=0;i<300;i++){
		m_dBias[i] = 0.0;
	}

	file[0].open('Weight.txt', ios::in);
	file[1].open('Bias.txt', ios::in);
	file[2].open('Gamma.txt', ios::in);
	file[3].open('Beta.txt', ios::in);
}

CEstimateCollisionNN::~CEstimateCollisionNN()
{

}

void CEstimateCollisionNN::EstimateCollision(double dInput[], double dOutput[])
{
	///< Input nomalization
	InputNomalization(dInput, m_dInputN);

	///< Layer 1
	CalLayer1(m_dInputN,m_dLayer1);

	///< Layer 2
	CalLayer2(m_dLayer1,m_dLayer2);

	///< Layer 3
	CalLayer3(m_dLayer2,m_dLayer3);

	///< Layer 4
	CalLayer4(m_dLayer3,m_dLayer4);

	///< Layer 5
	CalLayer5(m_dLayer4,m_dLayer5);

	///< Output
	CalOutput(m_dLayer5,dOutput);
}

void CEstimateCollisionNN::InputNomalization(double dInput[], double dInputN[])
{
	for (unsigned int i=0;i<180;i++){
		dInputN[i] = (dInput[i] - m_dInputNormalMin[i])/(m_dInputNormalMax[i] - m_dInputNormalMin[i]);
	}
}

void CEstimateCollisionNN::CalLayer1(double dInput[], double dOutput[])
{
	double dOutputNN[90] = {0.0};
	double dOutputBN[90] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[nJointDOF * 15 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 30; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[nJointDOF * 450 + nOutputNode * 30 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 90; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 90; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::CalLayer2(double dInput[], double dOutput[])
{
	double dOutputNN[90] = {0.0};
	double dOutputBN[90] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[90 + nJointDOF * 15 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[2700 + nJointDOF * 225 + nOutputNode * 30 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 90; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 90; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::CalLayer3(double dInput[], double dOutput[])
{
	double dOutputNN[6] = {0.0};
	double dOutputBN[6] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		dOutputNN[nJointDOF] = m_dBias[180 + nJointDOF];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nJointDOF] += m_dWeight[4050 + nJointDOF * 15 + nInputNode] * dInput[nJointDOF * 15 + nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 6; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 6; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::CalLayer4(double dInput[], double dOutput[])
{
	double dOutputNN[15] = {0.0};
	double dOutputBN[15] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[186 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 6; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[4140 + nOutputNode * 15 + nInputNode] * dInput[nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 15; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 15; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::CalLayer5(double dInput[], double dOutput[])
{
	double dOutputNN[15] = {0.0};
	double dOutputBN[15] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[201 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[4230 + nOutputNode * 15 + nInputNode] * dInput[nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 15; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 15; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::CalOutput(double dInput[], double dOutput[])
{
	double dOutputNN[2] = {0.0};
	double dOutputBN[2] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 2; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[216 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[4455 + nOutputNode * 2 + nInputNode] * dInput[nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int i = 0;i < 2; i++){
		dOutputBN[i] = dOutputNN[i];
	}

	///< Calculation Activation Function (����� �ۼ�, Sigmode�� ���� ��û)
	for (unsigned int i = 0;i < 2; i++){
		if (dOutputBN[i] < 0.0){
			dOutput[i] = 0.0;
		}
		else{
			dOutput[i] = dOutputBN[i];
		}
	}
}


void CEstimateCollisionNN::SetNNWeight(unsigned int nIndex, double dWeight)
{
	if(!file[0].is_open())
	{
		std::cout<<"can not found the Weight file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[0].eof())
	{
		file[0] >> temp;
		if(temp != '\n')
			m_dWeight[index%10000] = temp;
		index ++; 
	}
	// m_dWeight[nIndex%10000] = dWeight;
}

void CEstimateCollisionNN::SetNNBias(unsigned int nIndex, double dBias)
{
	if(!file[1].is_open())
	{
		std::cout<<"can not found the Bias file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[1].eof())
	{
		file[1] >> temp;
		if(temp != '\n')
			m_dBias[index%300] = temp;
		index ++; 
	}
	// m_dBias[nIndex%300] = dBias;
}

void CEstimateCollisionNN::SetBNGamma(unsigned int nIndex, double dWeight)
{
	if(!file[2].is_open())
	{
		std::cout<<"can not found the Gamma file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[2].eof())
	{
		file[2] >> temp;
		if(temp != '\n')
			m_dWeight[index%10000] = temp;
		index ++; 
	}
	// m_dWeight[nIndex%10000] = dWeight;
}

void CEstimateCollisionNN::SetBNBeta(unsigned int nIndex, double dBias)
{
	if(!file[3].is_open())
	{
		std::cout<<"can not found the Beta file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[3].eof())
	{
		file[3] >> temp;
		if(temp != '\n')
			m_dBias[index%10000] = temp;
		index ++; 
	}
	// m_dBias[nIndex%300] = dBias;
}

double CEstimateCollisionNN::GetNNWeight(unsigned int nIndex)
{
	return m_dWeight[nIndex%10000];
}
double CEstimateCollisionNN::GetNNBias(unsigned int nIndex)
{
	return m_dBias[nIndex%300];
}