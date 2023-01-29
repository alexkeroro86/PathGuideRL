#pragma once

#include "cuda_helper.cuh"

// vec3 for vector and color
class vec3
{
public:
	__host__ __device__ vec3() : e{0.f, 0.f, 0.f} {}
	__host__ __device__ vec3(float c) : e{c, c, c} {}
	__host__ __device__ vec3(float e1, float e2, float e3) : e{e1, e2, e3} {}
	__host__ __device__ vec3(const vec3& v) : e{v.e[0], v.e[1], v.e[2]} {}

	// operator
	__host__ __device__ const vec3& operator+() const { return *this; }
	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator[](int i) const { return e[i]; }
	__host__ __device__ float& operator[](int i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3& rhs);
	__host__ __device__ vec3& operator-=(const vec3& rhs);
	__host__ __device__ vec3& operator*=(const vec3& rhs);
	__host__ __device__ vec3& operator/=(const vec3& rhs);
	__host__ __device__ vec3& operator*=(const float c);
	__host__ __device__ vec3& operator/=(const float c);

	__host__ __device__ friend vec3 operator+(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 operator-(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 operator*(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 operator/(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 operator*(const float c, const vec3& rhs);
	__host__ __device__ friend vec3 operator*(const vec3& lhs, const float c);
	__host__ __device__ friend vec3 operator/(const vec3& lhs, const float rhs);

	// vector math function
	__host__ __device__ friend float dot(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 cross(const vec3& lhs, const vec3& rhs);
	__host__ __device__ friend vec3 unit_vector(const vec3& v);
	__host__ __device__ friend vec3 power(const vec3& v, float exp);

	// access function
	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	// member function
	__host__ __device__ inline float length() const { return sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
	__host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ void make_unit_vector();
	__host__ __device__ float max() const { return fmaxf(e[0], fmaxf(e[1], e[2])); }

private:
	float e[3];
};

// operator
__host__ __device__ inline vec3& vec3::operator+=(const vec3 &rhs)
{
	e[0] += rhs.e[0];
	e[1] += rhs.e[1];
	e[2] += rhs.e[2];
	return *this;
}
__host__ __device__ inline vec3& vec3::operator-=(const vec3 &rhs)
{
	e[0] -= rhs.e[0];
	e[1] -= rhs.e[1];
	e[2] -= rhs.e[2];
	return *this;
}
__host__ __device__ inline vec3& vec3::operator*=(const vec3 &rhs)
{
	e[0] *= rhs.e[0];
	e[1] *= rhs.e[1];
	e[2] *= rhs.e[2];
	return *this;
}
__host__ __device__ inline vec3& vec3::operator/=(const vec3 &rhs)
{
	e[0] /= rhs.e[0];
	e[1] /= rhs.e[1];
	e[2] /= rhs.e[2];
	return *this;
}
__host__ __device__ inline vec3& vec3::operator*=(const float c)
{
	e[0] *= c;
	e[1] *= c;
	e[2] *= c;
	return *this;
}
__host__ __device__ inline vec3& vec3::operator/=(const float c)
{
	float k = 1.f / c;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

__host__ __device__ inline vec3 operator+(const vec3 &lhs, const vec3 &rhs)
{
	return vec3(lhs.e[0] + rhs.e[0], lhs.e[1] + rhs.e[1], lhs.e[2] + rhs.e[2]);
}
__host__ __device__ inline vec3 operator-(const vec3 &lhs, const vec3 &rhs)
{
	return vec3(lhs.e[0] - rhs.e[0], lhs.e[1] - rhs.e[1], lhs.e[2] - rhs.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3 &lhs, const vec3 &rhs)
{
	return vec3(lhs.e[0] * rhs.e[0], lhs.e[1] * rhs.e[1], lhs.e[2] * rhs.e[2]);
}
__host__ __device__ inline vec3 operator/(const vec3 &lhs, const vec3 &rhs)
{
	return vec3(lhs.e[0] / rhs.e[0], lhs.e[1] / rhs.e[1], lhs.e[2] / rhs.e[2]);
}
__host__ __device__ inline vec3 operator*(float c, const vec3 &rhs)
{
	return vec3(c * rhs.e[0], c * rhs.e[1], c * rhs.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3 &lhs, float c)
{
	return vec3(c * lhs.e[0], c * lhs.e[1], c * lhs.e[2]);
}
__host__ __device__ inline vec3 operator/(const vec3 &lhs, float c)
{
	float k = 1.f / c;
	return vec3(lhs.e[0] * k, lhs.e[1] * k, lhs.e[2] * k);
}

// vector math function
__host__ __device__ inline float dot(const vec3& lhs, const vec3& rhs)
{
	return lhs.e[0] * rhs.e[0] + lhs.e[1] * rhs.e[1] + lhs.e[2] * rhs.e[2];
}
__host__ __device__ inline vec3 cross(const vec3 &lhs, const vec3 &rhs)
{
	return vec3(
		lhs.e[1] * rhs.e[2] - lhs.e[2] * rhs.e[1],
		lhs.e[2] * rhs.e[0] - lhs.e[0] * rhs.e[2],
		lhs.e[0] * rhs.e[1] - lhs.e[1] * rhs.e[0]
	);
}
__host__ __device__ inline vec3 unit_vector(const vec3& v)
{
	return v / v.length();
}
__host__ __device__ inline vec3 power(const vec3& v, float exp)
{
	return vec3(powf(v.e[0], exp), powf(v.e[1], exp), powf(v.e[2], exp));
}

// member function
__host__ __device__ inline void vec3::make_unit_vector()
{
	float k = 1.f / sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

