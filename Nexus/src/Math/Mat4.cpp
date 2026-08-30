#include "Mat4.h"

namespace Nexus {

	Mat4 operator*(const Mat4& a, const Mat4& b)
	{
		Mat4 r;
		for (uint i = 0; i < 16; i += 4)
			for (uint j = 0; j < 4; ++j)
			{
				r[i + j] =
					(a.cell[i + 0] * b.cell[j + 0]) +
					(a.cell[i + 1] * b.cell[j + 4]) +
					(a.cell[i + 2] * b.cell[j + 8]) +
					(a.cell[i + 3] * b.cell[j + 12]);
			}
		return r;
	}

	Mat4 operator*(const Mat4& a, const float s)
	{
		Mat4 r;
		for (uint i = 0; i < 16; i += 4) r.cell[i] = a.cell[i] * s;
		return r;
	}

	Mat4 operator*(const float s, const Mat4& a)
	{
		Mat4 r;
		for (uint i = 0; i < 16; i++) r.cell[i] = a.cell[i] * s;
		return r;
	}

	Mat4 operator+(const Mat4& a, const Mat4& b)
	{
		Mat4 r;
		for (uint i = 0; i < 16; i += 4) r.cell[i] = a.cell[i] + b.cell[i];
		return r;
	}

	bool operator==(const Mat4& a, const Mat4& b)
	{
		for (uint i = 0; i < 16; i++)
			if (a.cell[i] != b.cell[i]) return false;
		return true;
	}

	bool operator!=(const Mat4& a, const Mat4& b) { return !(a == b); }

	float4 operator*(const Mat4& a, const float4& b)
	{
		return make_float4(a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3] * b.w,
			a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7] * b.w,
			a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11] * b.w,
			a.cell[12] * b.x + a.cell[13] * b.y + a.cell[14] * b.z + a.cell[15] * b.w);
	}

	float4 operator*(const float4& b, const Mat4& a)
	{
		return make_float4(a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3] * b.w,
			a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7] * b.w,
			a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11] * b.w,
			a.cell[12] * b.x + a.cell[13] * b.y + a.cell[14] * b.z + a.cell[15] * b.w);
	}

	float3 TransformPosition(const float3& a, const Mat4& M)
	{
		return make_float3(make_float4(a, 1) * M);
	}
	float3 TransformVector(const float3& a, const Mat4& M)
	{
		return make_float3(make_float4(a, 0) * M);
	}

}