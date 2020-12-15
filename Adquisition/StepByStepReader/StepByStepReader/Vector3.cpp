#include "Vector3.h"
#include <iostream>

Vector3 & Vector3::operator=(const Vector3 & v)
{
	if (this != &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}
	return *this;
}

std::ostream & operator<<(std::ostream & os, const Vector3 & v)
{
	os << v.X() << ";" << v.Y() << ";" << v.Z();
	return os;
}
