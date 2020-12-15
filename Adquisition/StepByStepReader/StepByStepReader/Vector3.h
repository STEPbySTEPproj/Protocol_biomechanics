#pragma once
#ifndef VECTOR3_H
#define VECTOR3_H

// Forward declaration of std::ostream
#include <iosfwd>

class Vector3
{
private:
	float x, y, z;
public:
	Vector3() : x(0), y(0), z(0) {}
	Vector3(const float x, const float y, const float z): x(x), y(y), z(z) {}
	Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}
	Vector3& operator=(const Vector3 &v);
	friend std::ostream& operator<<(std::ostream& os, const Vector3 &v);

	float X() const { return x; }
	void X(const float x) { this->x = x; }
	float Y() const { return y; }
	void Y(const float y) { this->y = y; }
	float Z() const { return z; }
	void Z(const float z) { this->z = z; }
};

#endif // !VECTOR3_H