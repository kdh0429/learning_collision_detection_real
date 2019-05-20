
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
	file[4].open('Mean.txt', ios::in);
	file[5].open('Variance.txt', ios::in);
	file[6].open('InputMinMax.txt', ios::in);

	SetNNWeight(); ///< ����� �ۼ�
	SetNNBias(); ///< ����� �ۼ�
	SetBNGamma(); ///< ����� �ۼ�
	SetBNBeta(); ///< ����� �ۼ�
	SetBNMean(); ///< ����� �ۼ�
	SetBNVariance(); ///< ����� �ۼ�
	SetInputNormalMinMax(); ///< ����� �ۼ�
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
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[nJointDOF * 46 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 30; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[nJointDOF * 915 + nOutputNode * 30 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputBN[15*nJointDOF+nOutputNode] = m_dGamma[46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[46*nJointDOF + nOutputNode])/sqrt(m_dVariance[46*nJointDOF + nOutputNode]) - m_dBeta[46*nJointDOF + nOutputNode];
		}
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
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[15 + nJointDOF * 46 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[450 + nJointDOF * 915 + nOutputNode * 15 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputBN[15*nJointDOF+nOutputNode] = m_dGamma[15 + 46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[15+ 46*nJointDOF + nOutputNode])/sqrt(m_dVariance[15+46*nJointDOF + nOutputNode]) - m_dBeta[15+46*nJointDOF + nOutputNode];
		}
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
	double dOutputNN[90] = {0.0};
	double dOutputBN[90] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[30 + nJointDOF * 46 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[675 + nJointDOF * 915 + nOutputNode * 15 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputBN[15*nJointDOF+nOutputNode] = m_dGamma[30 + 46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[30+ 46*nJointDOF + nOutputNode])/sqrt(m_dVariance[30+46*nJointDOF + nOutputNode]) - m_dBeta[30+46*nJointDOF + nOutputNode];
		}
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


void CEstimateCollisionNN::CalLayer4(double dInput[], double dOutput[])
{
	double dOutputNN[6] = {0.0};
	double dOutputBN[6] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		dOutputNN[nJointDOF] = m_dBias[45 + nJointDOF * 46];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nJointDOF] += m_dWeight[900 + nJointDOF * 915 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		dOutputBN[nJointDOF] = m_dGamma[45 + 46*nJointDOF] * (dOutputNN[nJointDOF] - m_dMean[45+ 46*nJointDOF])/sqrt(m_dVariance[45+46*nJointDOF]) - m_dBeta[45+46*nJointDOF];
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


void CEstimateCollisionNN::CalLayer5(double dInput[], double dOutput[])
{
	double dOutputNN[15] = {0.0};
	double dOutputBN[15] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[276 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 6; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[5490 + nOutputNode * 15 + nInputNode] * dInput[nInputNode];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutputBN[nOutputNode] = m_dGamma[276 + nOutputNode] * (dOutputNN[nOutputNode] - m_dMean[276 + nOutputNode])/sqrt(m_dVariance[276 + nOutputNode]) - m_dBeta[276 + nOutputNode];
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
		dOutputNN[nOutputNode] = m_dBias[291 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[5580 + nOutputNode * 15 + nInputNode] * dInput[nInputNode];
		}
	}

	dOutput[0] = exp(dOutputNN[0])/(exp(dOutputNN[0]) + exp(dOutputNN[1]));
	dOutput[1] = exp(dOutputNN[1])/(exp(dOutputNN[0]) + exp(dOutputNN[1]));
}


void CEstimateCollisionNN::SetNNWeight()
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

void CEstimateCollisionNN::SetNNBias()
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

void CEstimateCollisionNN::SetBNGamma()
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
			m_dGamma[index%300] = temp;
		index ++; 
	}
	// m_dWeight[nIndex%10000] = dWeight;
}

void CEstimateCollisionNN::SetBNBeta()
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
			m_dBeta[index%300] = temp;
		index ++; 
	}
	// m_dBias[nIndex%300] = dBias;
}

void CEstimateCollisionNN::SetBNMean()
{
	if(!file[4].is_open())
	{
		std::cout<<"can not found the Mean file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[4].eof())
	{
		file[4] >> temp;
		if(temp != '\n')
			m_dMean[index%300] = temp;
		index ++; 
	}
	// m_dBias[nIndex%300] = dBias;
}

void CEstimateCollisionNN::SetBNVariance()
{
	if(!file[5].is_open())
	{
		std::cout<<"can not found the Variance file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[5].eof())
	{
		file[5] >> temp;
		if(temp != '\n')
			m_dBias[index%300] = temp;
		index ++; 
	}
	// m_dBias[nIndex%300] = dBias;
}

void CEstimateCollisionNN::SetInputNormalMinMax()
{
	if(!file[6].is_open())
	{
		std::cout<<"can not found the Input Normalize file"<<std::endl;
	}
	int index = 0;
	double temp;
	while(!file[6].eof())
	{
		file[6] >> temp;
		if(temp != '\n')
			if (index < 36)
				m_dInputNormalMin[index%36] = temp;
			else
				m_dInputNormalMax[index%36] = temp;
			index ++; 
	}
	for (int i = 36; i<180; i++)
	{
		m_dInputNormalMin[i] = m_dInputNormalMin[i%36];
		m_dInputNormalMax[i] = m_dInputNormalMax[i%36];
	}
}