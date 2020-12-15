#include "BoneData.h"

BoneData & BoneData::operator=(const BoneData & bone)
{
	if (this != &bone)
	{
		displacement = bone.displacement;
		eulerAngles = bone.eulerAngles;
	}
	return *this;
}
