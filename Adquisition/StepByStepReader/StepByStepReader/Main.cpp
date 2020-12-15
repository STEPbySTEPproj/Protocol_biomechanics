#include <Windows.h>
#include <iostream>
#include <string>
#include <ctime>

//NEURON
// Include the NeuronDataReader head file 
#include "NeuronDataReader.h"
#define WM_UPDATE_MESSAGE (WM_USER+200)
//------

#include "Vector3.h"
#include "BoneData.h"
#include "BVHFrame.h"
#include <fstream>
using namespace std;
//NEURON
// Load SDK library
#pragma comment(lib, "NeuronDataReader.lib") //Add Lib
//------

//NEURON
// Receive BVH data
static void __stdcall bvhFrameDataReceived(void* customedObj, SOCKET_REF sender, BvhDataHeader* header, float* data);

time_t timer, now;
int secsPrev = 0, secsCur = 0;
int fps = 0;
ofstream out;


int main(int argc, char *argv[])
{
	timer = time(NULL);
	BoneData bones[(int)Bone::NumOfBones];
	BRRegisterFrameDataCallback(nullptr, bvhFrameDataReceived);
	SOCKET_REF sockTCPRef = nullptr;
	char ip[16] = "127.0.0.1\0";
	int port = 7001;
	BVHFrame::SetRotationConfig(EulerRotation::XYZ);
	if (argc > 1)
	{
		strcpy_s(ip, argv[1]);
		if (argc > 2)
		{
			port = atoi(argv[2]);
		}
	}
	std::cout << "Connecting to: " << ip << ":" << port << std::endl;
	sockTCPRef = BRConnectTo(ip, port);
	if (sockTCPRef != nullptr)
	{
		std::cout << "Connected." << std::endl;
		out.open("../../Data/Test02/WalkF_03.csv", ios::out);
		out << "RUL-X;RUL-Y;RUL-Z;RL-X;RL-Y;RL-Z;F-X;F-Y;F-Z" << endl;
	}
	else
	{
		std::cout << "Error: " << BRGetLastErrorMessage() << std::endl;
		std::cout << "       Make sure:" << std::endl;
		std::cout << "       - The Axis Neuron application is running." << std::endl;
		std::cout << "       - The IP is correct." << std::endl;
		std::cout << "       - The port is correct and open." << std::endl;
		exit(EXIT_FAILURE);
	}

	while (true)
	{
		if (GetKeyState(115) & 0x8000)
		{

		}
	}
	out.close();
	return 0;
}

void __stdcall bvhFrameDataReceived(void* customedObj, SOCKET_REF sender, BvhDataHeader* header, float* data)
{
	//now = time(NULL);
	
	//secsCur = (int)difftime(now, timer);
	//if (secsCur > secsPrev)
	//{
	//	std::cout << fps << std::endl;
	//	secsPrev = secsCur;
	//	fps = 0;
	//}
	//else
	//{
	//	fps++;
	//}
	BVHFrame frame(header, data);
	std::cout << sizeof(DATA_VER);
	//Bone boneIndex = Bone::Hips;
	//BoneData bone = frame.GetBoneData(boneIndex);
	//std::cout << bone.GetRotation() << std::endl;
	//std::cout << frame.GetRotation(header, data, boneIndex) << std::endl;
	std::cout << frame.GetRotation(header, data, Bone::RightUpLeg) << ";" << frame.GetRotation(header, data, Bone::RightLeg)
		<< ";" << frame.GetRotation(header, data, Bone::RightFoot) << ";" << std::endl;
	//showBvhBoneInfo(sender, header, data);
}