#pragma once
#ifndef BONEDATA_H
#define BONEDATA_H

#include "Vector3.h"

enum class Bone
{
	Hips              = 0,
	RightUpLeg        = 1,
	RightLeg          = 2,
	RightFoot         = 3,
	LeftUpLeg         = 4,
	LeftLeg           = 5,
	LeftFoot          = 6,
	Spine             = 7,
	Spine1            = 8,
	Spine2            = 9,
	Spine3            = 10,
	Neck              = 11,
	Head              = 12,
	RightShoulder     = 13,
	RightArm          = 14,
	RightForeArm      = 15,
	RightHand         = 16,
	RightHandThumb1   = 17,
	RightHandThumb2   = 18,
	RightHandThumb3   = 19,
	RightInHandIndex  = 20,
	RightHandIndex1   = 21,
	RightHandIndex2   = 22,
	RightHandIndex3   = 23,
	RightInHandMiddle = 24,
	RightHandMiddle1  = 25,
	RightHandMiddle2  = 26,
	RightHandMiddle3  = 27,
	RightInHandRing   = 28,
	RightHandRing1    = 29,
	RightHandRing2    = 30,
	RightHandRing3    = 31,
	RightInHandPinky  = 32,
	RightHandPinky1   = 33,
	RightHandPinky2   = 34,
	RightHandPinky3   = 35,
	LeftShoulder      = 36,
	LeftArm           = 37,
	LeftForeArm       = 38,
	LeftHand          = 39,
	LeftHandThumb1    = 40,
	LeftHandThumb2    = 41,
	LeftHandThumb3    = 42,
	LeftInHandIndex   = 43,
	LeftHandIndex1    = 44,
	LeftHandIndex2    = 45,
	LeftHandIndex3    = 46,
	LeftInHandMiddle  = 47,
	LeftHandMiddle1   = 48,
	LeftHandMiddle2   = 49,
	LeftHandMiddle3   = 50,
	LeftInHandRing    = 51,
	LeftHandRing1     = 52,
	LeftHandRing2     = 53,
	LeftHandRing3     = 54,
	LeftInHandPinky   = 55,
	LeftHandPinky1    = 56,
	LeftHandPinky2    = 57,
	LeftHandPinky3    = 58,

	NumOfBones        = 59
};

class BoneData
{
private:
	Vector3 displacement;
	Vector3 eulerAngles;
public:
	BoneData() : displacement(), eulerAngles() {}
	BoneData(const Vector3 displacement, const Vector3 eulerAngles) : displacement(displacement), eulerAngles(eulerAngles) {}
	BoneData(const float x, const float y, const float z, const float alpha, const float beta, const float gamma) : displacement(x, y, z), eulerAngles(alpha, beta, gamma) {}
	BoneData(const BoneData &bone): displacement(bone.displacement), eulerAngles(bone.eulerAngles) {}
	BoneData& operator=(const BoneData &bone);
	Vector3 GetDisplacement() const { return displacement; }
	Vector3 GetRotation() const { return eulerAngles; }
	void SetDisplacement(Vector3 &displacement) {}
};

#endif // !BONEDATA_H
