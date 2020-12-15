#include "BVHFrame.h"
#include "Vector3.h"

// Default config
EulerRotation BVHFrame::rotation = EulerRotation::YXZ;
int BVHFrame::rotationIndex[3] = { 1, 0, 2 };

BVHFrame::BVHFrame(BvhDataHeader * header, float * data) : frameIndex(header->FrameIndex), withReference((bool)header->WithReference), withDisplacement((bool)header->WithDisp)
{
	// Init offset in 0
	int offset = 0;
	// Read reference
	if (withReference)
	{
		//reference = BoneData(Vector3(data[0], data[1], data[2]), Vector3(data[4], data[3], data[5]));
		reference = BoneData(Vector3(data[0], data[1], data[2]), Vector3(data[3 + rotationIndex[0]], data[3 + rotationIndex[1]], data[3 + rotationIndex[2]]));
		offset += 6;
	}
	// Read hips data. With or without displacement, the hips data has displacement and rotation.
	//bones[0] = BoneData(Vector3(data[offset], data[offset + 1], data[offset + 2]), Vector3(data[offset + 4], data[offset + 3], data[offset + 5]));
	bones[0] = BoneData(Vector3(data[offset], data[offset + 1], data[offset + 2]), Vector3(data[offset + 3 + rotationIndex[0]], data[offset + 3 + rotationIndex[1]], data[offset + 3 + rotationIndex[2]]));
	offset += 6;
	// Read rest of bone data
	for (int i = 1; i < (int)Bone::NumOfBones; i++)
	{
		// With displacement every bone has displacement (x, y, z) and rotation (alpha, beta, gamma)
		// Without displacement every bone has only the rotation.
		if (withDisplacement)
		{
			//bones[i] = BoneData(Vector3(data[offset], data[offset + 1], data[offset + 2]), Vector3(data[offset + 4], data[offset + 3], data[offset + 5]));
			bones[i] = BoneData(Vector3(data[offset], data[offset + 1], data[offset + 2]), Vector3(data[offset + 3 + rotationIndex[0]], data[offset + 3 + rotationIndex[1]], data[offset + 3 + rotationIndex[2]]));
			offset += 6;
		}
		else
		{
			//bones[i] = BoneData(Vector3(), Vector3(data[offset + 1], data[offset], data[offset + 2]));
			bones[i] = BoneData(Vector3(), Vector3(data[offset + rotationIndex[0]], data[offset + rotationIndex[1]], data[offset + rotationIndex[2]]));
			offset += 3;
		}
	}
}

BoneData BVHFrame::GetBoneData(Bone bone)
{
	return bones[(int)bone];
}

Vector3 BVHFrame::GetPosition(BvhDataHeader * header, float * data, Bone bone) const
{
	int offset = 0;
	if ((int)header->WithReference != 0)
	{
		offset += 6;
	}
	if ((int)header->WithDisp != 0 || bone == Bone::Hips)
	{
		offset += (int)bone * 6;
		return Vector3(data[offset], data[offset + 1], data[offset + 2]);
	}
	return Vector3();
}

Vector3 BVHFrame::GetRotation(BvhDataHeader * header, float * data, Bone bone) const
{
	int offset = 0;
	if ((int)header->WithReference != 0)
	{
		offset += 6;
	}
	if ((int)header->WithDisp != 0)
	{
		offset += 3 + (int)bone * 6;
	}
	else
	{
		offset += 3 + (int)bone * 3;
	}
	//return Vector3(data[offset + 1], data[offset], data[offset + 2]);
	return Vector3(data[offset + rotationIndex[0]], data[offset + rotationIndex[1]], data[offset + rotationIndex[2]]);
}

EulerRotation BVHFrame::GetRotationConfig()
{
	return BVHFrame::rotation;
}

void BVHFrame::SetRotationConfig(const EulerRotation rotation)
{
	BVHFrame::rotation = rotation;
	switch (BVHFrame::rotation)
	{
	case EulerRotation::XYZ:
		rotationIndex[0] = 0;
		rotationIndex[1] = 1;
		rotationIndex[2] = 2;
		break;
	case EulerRotation::XZY:
		rotationIndex[0] = 0;
		rotationIndex[1] = 2;
		rotationIndex[2] = 1;
		break;
	case EulerRotation::YXZ:
		rotationIndex[0] = 1;
		rotationIndex[1] = 0;
		rotationIndex[2] = 2;
		break;
	case EulerRotation::YZX:
		rotationIndex[0] = 1;
		rotationIndex[1] = 2;
		rotationIndex[2] = 0;
		break;
	case EulerRotation::ZXY:
		rotationIndex[0] = 2;
		rotationIndex[1] = 0;
		rotationIndex[2] = 1;
		break;
	case EulerRotation::ZYX:
		rotationIndex[0] = 2;
		rotationIndex[1] = 1;
		rotationIndex[2] = 0;
		break;
	default:
		break;
	}
}

EulerRotation BVHFrame::GetEulerFromString(std::string rotationString)
{
	if (rotationString == "XYZ")
	{

	}
	return EulerRotation();
}
