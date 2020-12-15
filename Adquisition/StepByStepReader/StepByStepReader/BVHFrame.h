#pragma once
#ifndef BVHFRAME_H
#define BVHFRAME_H

#include <string>
#include "Vector3.h"
#include "DataType.h"
#include "BoneData.h"

enum class EulerRotation
{
	XYZ = 0,
	XZY = 1,
	YXZ = 2, // Default Perception Neuron config
	YZX = 3,
	ZXY = 4,
	ZYX = 5
};

class BVHFrame
{
private:
	int frameIndex;
	bool withDisplacement;
	bool withReference;
	BoneData reference;
	BoneData bones[(int)Bone::NumOfBones];
	static EulerRotation rotation;
	static int rotationIndex[3];
public:
	BVHFrame(BvhDataHeader *header, float *data);
	BoneData GetBoneData(Bone bone);
	Vector3 GetPosition(BvhDataHeader *header, float *data, Bone bone) const;
	Vector3 GetRotation(BvhDataHeader *header, float *data, Bone bone) const;
	static EulerRotation GetRotationConfig();
	static void SetRotationConfig(const EulerRotation rotation = EulerRotation::YXZ);
	static EulerRotation GetEulerFromString(std::string rotationString);
};

#endif // !BVHFRAME_H
