
#include "UtilFunction.h"

CEstimateCollisionNN::CEstimateCollisionNN()
{
	for (unsigned int i=0;i<10000;i++){
		m_dWeight[i] = 0.0;
	}
	for (unsigned int i=0;i<300;i++){
		m_dBias[i] = 0.0;
	}
	
	file[0].open("Weight.txt", ios::in);
	file[1].open("Bias.txt", ios::in);
	file[2].open("Gamma.txt", ios::in);
	file[3].open("Beta.txt", ios::in);
	file[4].open("Mean.txt", ios::in);
	file[5].open("Variance.txt", ios::in);
	file[6].open("InputMinMax.txt", ios::in);

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
	InputNormalization(dInput, m_dInputN);
	
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

	// for (int i=0; i<90; i++)
	//  	cout<< m_dLayer4[i] <<endl;

}

void CEstimateCollisionNN::InputNormalization(double dInput[], double dInputN[])
{
	for (unsigned int i=0;i<180;i++){
		dInputN[i] = (dInput[i] - m_dInputNormalMin[i])/(m_dInputNormalMax[i] - m_dInputNormalMin[i]);
	}
}

void CEstimateCollisionNN::CalLayer1(double dInput[], double dOutput[])
{
	double dOutputNN[90] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[nJointDOF * 46 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 30; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[nJointDOF * 915 + nOutputNode * 30 + nInputNode] * dInput[nJointDOF * 30 + nInputNode];
			}
		}
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 90; i++){
		if (dOutputNN[i] < 0.0){
			dOutputNN[i] = 0.0;
		}
		else{
			dOutputNN[i] = dOutputNN[i];
		}
	}
	

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutput[15*nJointDOF+nOutputNode] = m_dGamma[46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[46*nJointDOF + nOutputNode])/sqrt(m_dVariance[46*nJointDOF + nOutputNode]+0.001) + m_dBeta[46*nJointDOF + nOutputNode];
		}
	}
}


void CEstimateCollisionNN::CalLayer2(double dInput[], double dOutput[])
{
	double dOutputNN[90] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[15 + nJointDOF * 46 + nOutputNode];
			for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[450 + nJointDOF * 915 + nOutputNode * 15 + nInputNode] * dInput[nJointDOF * 15 + nInputNode];
			}
		}
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 90; i++){
		if (dOutputNN[i] < 0.0){
			dOutputNN[i] = 0.0;
		}
		else{
			dOutputNN[i] = dOutputNN[i];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutput[15*nJointDOF+nOutputNode] = m_dGamma[15 + 46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[15+ 46*nJointDOF + nOutputNode])/sqrt(m_dVariance[15+46*nJointDOF + nOutputNode]+0.001) + m_dBeta[15+46*nJointDOF + nOutputNode];
		}
	}
}


void CEstimateCollisionNN::CalLayer3(double dInput[], double dOutput[])
{
	double dOutputNN[90] = {0.0};

	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutputNN[nJointDOF * 15 + nOutputNode] = m_dBias[30 + nJointDOF * 46 + nOutputNode];

			for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
				dOutputNN[nJointDOF * 15 + nOutputNode] += m_dWeight[675 + nJointDOF * 915 + nOutputNode * 15 + nInputNode] * dInput[nJointDOF * 15 + nInputNode];
			}
			
		}
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 90; i++){
		if (dOutputNN[i] < 0.0){
			dOutputNN[i] = 0.0;
		}
		else{
			dOutputNN[i] = dOutputNN[i];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
			dOutput[15*nJointDOF+nOutputNode] = m_dGamma[30 + 46*nJointDOF + nOutputNode] * (dOutputNN[nJointDOF * 15 + nOutputNode] - m_dMean[30+ 46*nJointDOF + nOutputNode])/sqrt(m_dVariance[30+46*nJointDOF + nOutputNode]+0.001) + m_dBeta[30+46*nJointDOF + nOutputNode];
		}
	}
}


void CEstimateCollisionNN::CalLayer4(double dInput[], double dOutput[])
{
	double dOutputNN[6] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		dOutputNN[nJointDOF] = m_dBias[45 + nJointDOF * 46];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nJointDOF] += m_dWeight[900 + nJointDOF * 915 + nInputNode] * dInput[nJointDOF * 15 + nInputNode];
		}
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 6; i++){
		if (dOutputNN[i] < 0.0){
			dOutputNN[i] = 0.0;
		}
		else{
			dOutputNN[i] = dOutputNN[i];
		}
	}
	
	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nJointDOF = 0;nJointDOF < 6; nJointDOF++){
		dOutput[nJointDOF] = m_dGamma[45 + 46*nJointDOF] * (dOutputNN[nJointDOF] - m_dMean[45+ 46*nJointDOF])/sqrt(m_dVariance[45+46*nJointDOF]+0.001) + m_dBeta[45+46*nJointDOF];
	}
}


void CEstimateCollisionNN::CalLayer5(double dInput[], double dOutput[])
{
	double dOutputNN[15] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[276 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 6; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[5490 + nOutputNode * 6 + nInputNode] * dInput[nInputNode];
		}
	}

	///< Calculation Activation Function (RELU)
	for (unsigned int i = 0;i < 15; i++){
		if (dOutputNN[i] < 0.0){
			dOutputNN[i] = 0.0;
		}
		else{
			dOutputNN[i] = dOutputNN[i];
		}
	}

	///< Calculation Batch Normalization (����� �ۼ�)
	for (unsigned int nOutputNode = 0; nOutputNode < 15; nOutputNode++){
		dOutput[nOutputNode] = m_dGamma[276 + nOutputNode] * (dOutputNN[nOutputNode] - m_dMean[276 + nOutputNode])/sqrt(m_dVariance[276 + nOutputNode]+0.001) + m_dBeta[276 + nOutputNode];
	}
}


void CEstimateCollisionNN::CalOutput(double dInput[], double dOutput[])
{
	double dOutputNN[2] = {0.0};
	///< Calculation Neural Network
	for (unsigned int nOutputNode = 0; nOutputNode < 2; nOutputNode++){
		dOutputNN[nOutputNode] = m_dBias[291 + nOutputNode];
		for (unsigned int nInputNode = 0; nInputNode < 15; nInputNode++){
			dOutputNN[nOutputNode] += m_dWeight[5580 + nOutputNode * 15 + nInputNode] * dInput[nInputNode];
		}
	}
	double Max = 0.0;//max(dOutputNN[0],dOutputNN[1]);
	dOutput[0] = exp(dOutputNN[0]-Max)/(exp(dOutputNN[0]-Max) + exp(dOutputNN[1]-Max));
	dOutput[1] = exp(dOutputNN[1]-Max)/(exp(dOutputNN[0]-Max) + exp(dOutputNN[1]-Max));
	//cout<<dOutput[0]<<endl;
	//cout<<(exp(dOutputNN[0]-Max) + exp(dOutputNN[1]-Max))<<endl;
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
			m_dVariance[index%300] = temp;
			index ++; 
	}
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
				m_dInputNormalMax[30*int(index/6)+index%6] = temp;
			else if (index < 72)
				m_dInputNormalMin[30*int((index-36)/6)+index%6] = temp;
			index ++; 
	}
	for (int j=0; j<6; j++)
		for (int i = 1; i<5; i++)
			for(int data=0; data<6; data++)
			{
				m_dInputNormalMin[30*j+6*i+data] = m_dInputNormalMin[30*j+data];
				m_dInputNormalMax[30*j+6*i+data] = m_dInputNormalMax[30*j+data];
			}
}