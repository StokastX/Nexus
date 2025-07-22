#pragma once

#include <cuda_runtime_api.h>
#include <assimp/BaseImporter.h>
#include "Utils/cuda_math.h"

// 4 x 4 row major matrix class adapted from Jacco Bikker's guide. See https://github.com/jbikker/bvh_article/blob/main/template/precomp.h

class Mat4
{
public:
	Mat4() = default;

	// From aiMatrix4x4
	__device__ __host__ Mat4(const aiMatrix4x4& mat)
	{
		for (int i = 0; i < 16; i++)
			cell[i] = mat[i / 4][i % 4];
	}

	float cell[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

	inline __host__ __device__ float& operator [] (const int idx) { return cell[idx]; }

	inline __host__ __device__ float operator()(const int i, const int j) const { return cell[i * 4 + j]; }

	inline __host__ __device__ float& operator()(const int i, const int j) { return cell[i * 4 + j]; }

	inline __host__ __device__ Mat4& operator += (const Mat4& a)
	{
		for (int i = 0; i < 16; i++) cell[i] += a.cell[i];
		return *this;
	}

	inline __host__ __device__ bool operator==(const Mat4& m)
	{
		for (int i = 0; i < 16; i++) if (m.cell[i] != cell[i]) return false; return true;
	}

	inline __host__ __device__ float3 GetTranslation() const { return make_float3(cell[3], cell[7], cell[11]); }

	inline __host__ __device__ float3 GetScale() const
	{
		return make_float3(
			sqrtf(cell[0] * cell[0] + cell[4] * cell[4] + cell[8] * cell[8]),
			sqrtf(cell[1] * cell[1] + cell[5] * cell[5] + cell[9] * cell[9]),
			sqrtf(cell[2] * cell[2] + cell[6] * cell[6] + cell[10] * cell[10])
		);
	}

	inline __host__ __device__ static Mat4 FromColumnMajor(const Mat4& T)
	{
		Mat4 M;
		M.cell[0] = T.cell[0], M.cell[1] = T.cell[4], M.cell[2] = T.cell[8], M.cell[3] = T.cell[12];
		M.cell[4] = T.cell[1], M.cell[5] = T.cell[5], M.cell[6] = T.cell[9], M.cell[7] = T.cell[13];
		M.cell[8] = T.cell[2], M.cell[9] = T.cell[6], M.cell[10] = T.cell[10], M.cell[11] = T.cell[14];
		M.cell[12] = T.cell[3], M.cell[13] = T.cell[7], M.cell[14] = T.cell[11], M.cell[15] = T.cell[15];
		return M;
	}

	inline __host__ __device__ constexpr static Mat4 Identity() { return Mat4{}; }

	inline __host__ __device__ static Mat4 ZeroMatrix() { Mat4 r; memset(r.cell, 0, 64); return r; }

	inline __host__ __device__ static Mat4 RotateX(const float a) { Mat4 r; r.cell[5] = cosf(a); r.cell[6] = -sinf(a); r.cell[9] = sinf(a); r.cell[10] = cosf(a); return r; };

	inline __host__ __device__ static Mat4 RotateY(const float a) { Mat4 r; r.cell[0] = cosf(a); r.cell[2] = sinf(a); r.cell[8] = -sinf(a); r.cell[10] = cosf(a); return r; };

	inline __host__ __device__ static Mat4 RotateZ(const float a) { Mat4 r; r.cell[0] = cosf(a); r.cell[1] = -sinf(a); r.cell[4] = sinf(a); r.cell[5] = cosf(a); return r; };

	inline __host__ __device__ static Mat4 Scale(const float s) { Mat4 r; r.cell[0] = r.cell[5] = r.cell[10] = s; return r; }

	inline __host__ __device__ static Mat4 Scale(const float3 s) { Mat4 r; r.cell[0] = s.x, r.cell[5] = s.y, r.cell[10] = s.z; return r; }

	inline __host__ __device__ static Mat4 Scale(const float4 s) { Mat4 r; r.cell[0] = s.x, r.cell[5] = s.y, r.cell[10] = s.z, r.cell[15] = s.w; return r; }

	inline __host__ __device__ static Mat4 Rotate(const float3& u, const float a) { return Rotate(u.x, u.y, u.z, a); }

	inline __host__ __device__ static Mat4 Rotate(const float x, const float y, const float z, const float a)
	{
		const float c = cosf( a ), l_c = 1 - c, s = sinf( a );
		// row major
		Mat4 m;
		m[0] = x * x + (1 - x * x) * c, m[1] = x * y * l_c + z * s, m[2] = x * z * l_c - y * s, m[3] = 0;
		m[4] = x * y * l_c - z * s, m[5] = y * y + (1 - y * y) * c, m[6] = y * z * l_c + x * s, m[7] = 0;
		m[8] = x * z * l_c + y * s, m[9] = y * z * l_c - x * s, m[10] = z * z + (1 - z * z) * c, m[11] = 0;
		m[12] = m[13] = m[14] = 0, m[15] = 1;
		return m;
	}

	inline __host__ __device__ static Mat4 LookAt(const float3 P, const float3 T)
	{
		const float3 z = normalize( T - P );
		const float3 x = normalize( cross( z, make_float3( 0, 1, 0 ) ) );
		const float3 y = cross( x, z );
		Mat4 M = Translate( P );
		M[0] = x.x, M[4] = x.y, M[8] = x.z;
		M[1] = y.x, M[5] = y.y, M[9] = y.z;
		M[2] = z.x, M[6] = z.y, M[10] = z.z;
		return M;
	}

	inline __host__ __device__ static Mat4 LookAt(const float3& pos, const float3& look, const float3& up)
	{
		// PBRT's lookat
		Mat4 cameraToWorld;
		// initialize fourth column of viewing matrix
		cameraToWorld( 0, 3 ) = pos.x;
		cameraToWorld( 1, 3 ) = pos.y;
		cameraToWorld( 2, 3 ) = pos.z;
		cameraToWorld( 3, 3 ) = 1;

		// initialize first three columns of viewing matrix
		float3 dir = normalize( look - pos );
		float3 right = cross( normalize( up ), dir );
		if (dot( right, right ) == 0)
		{
			printf(
				"\"up\" vector (%f, %f, %f) and viewing direction (%f, %f, %f) "
				"passed to LookAt are pointing in the same direction.  Using "
				"the identity transformation.\n",
				up.x, up.y, up.z, dir.x, dir.y, dir.z );
			return Mat4();
		}
		right = normalize( right );
		float3 newUp = cross( dir, right );
		cameraToWorld( 0, 0 ) = right.x, cameraToWorld( 1, 0 ) = right.y;
		cameraToWorld( 2, 0 ) = right.z, cameraToWorld( 3, 0 ) = 0.;
		cameraToWorld( 0, 1 ) = newUp.x, cameraToWorld( 1, 1 ) = newUp.y;
		cameraToWorld( 2, 1 ) = newUp.z, cameraToWorld( 3, 1 ) = 0.;
		cameraToWorld( 0, 2 ) = dir.x, cameraToWorld( 1, 2 ) = dir.y;
		cameraToWorld( 2, 2 ) = dir.z, cameraToWorld( 3, 2 ) = 0.;
		return cameraToWorld.Inverted();
	}

	inline __host__ __device__ static Mat4 Translate(const float x, const float y, const float z) { Mat4 r; r.cell[3] = x; r.cell[7] = y; r.cell[11] = z; return r; };

	inline __host__ __device__ static Mat4 Translate(const float3 P) { Mat4 r; r.cell[3] = P.x; r.cell[7] = P.y; r.cell[11] = P.z; return r; };

	inline __host__ __device__ float Trace3() const { return cell[0] + cell[5] + cell[10]; }

	inline __host__ __device__ Mat4 Transposed() const
	{
		Mat4 M;
		M[0] = cell[0], M[1] = cell[4], M[2] = cell[8];
		M[4] = cell[1], M[5] = cell[5], M[6] = cell[9];
		M[8] = cell[2], M[9] = cell[6], M[10] = cell[10];
		return M;
	}

	inline __host__ __device__ Mat4 Inverted() const
	{
		// from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
		const float inv[16] = {
			cell[5] * cell[10] * cell[15] - cell[5] * cell[11] * cell[14] - cell[9] * cell[6] * cell[15] +
			cell[9] * cell[7] * cell[14] + cell[13] * cell[6] * cell[11] - cell[13] * cell[7] * cell[10],
			-cell[1] * cell[10] * cell[15] + cell[1] * cell[11] * cell[14] + cell[9] * cell[2] * cell[15] -
			cell[9] * cell[3] * cell[14] - cell[13] * cell[2] * cell[11] + cell[13] * cell[3] * cell[10],
			cell[1] * cell[6] * cell[15] - cell[1] * cell[7] * cell[14] - cell[5] * cell[2] * cell[15] +
			cell[5] * cell[3] * cell[14] + cell[13] * cell[2] * cell[7] - cell[13] * cell[3] * cell[6],
			-cell[1] * cell[6] * cell[11] + cell[1] * cell[7] * cell[10] + cell[5] * cell[2] * cell[11] -
			cell[5] * cell[3] * cell[10] - cell[9] * cell[2] * cell[7] + cell[9] * cell[3] * cell[6],
			-cell[4] * cell[10] * cell[15] + cell[4] * cell[11] * cell[14] + cell[8] * cell[6] * cell[15] -
			cell[8] * cell[7] * cell[14] - cell[12] * cell[6] * cell[11] + cell[12] * cell[7] * cell[10],
			cell[0] * cell[10] * cell[15] - cell[0] * cell[11] * cell[14] - cell[8] * cell[2] * cell[15] +
			cell[8] * cell[3] * cell[14] + cell[12] * cell[2] * cell[11] - cell[12] * cell[3] * cell[10],
			-cell[0] * cell[6] * cell[15] + cell[0] * cell[7] * cell[14] + cell[4] * cell[2] * cell[15] -
			cell[4] * cell[3] * cell[14] - cell[12] * cell[2] * cell[7] + cell[12] * cell[3] * cell[6],
			cell[0] * cell[6] * cell[11] - cell[0] * cell[7] * cell[10] - cell[4] * cell[2] * cell[11] +
			cell[4] * cell[3] * cell[10] + cell[8] * cell[2] * cell[7] - cell[8] * cell[3] * cell[6],
			cell[4] * cell[9] * cell[15] - cell[4] * cell[11] * cell[13] - cell[8] * cell[5] * cell[15] +
			cell[8] * cell[7] * cell[13] + cell[12] * cell[5] * cell[11] - cell[12] * cell[7] * cell[9],
			-cell[0] * cell[9] * cell[15] + cell[0] * cell[11] * cell[13] + cell[8] * cell[1] * cell[15] -
			cell[8] * cell[3] * cell[13] - cell[12] * cell[1] * cell[11] + cell[12] * cell[3] * cell[9],
			cell[0] * cell[5] * cell[15] - cell[0] * cell[7] * cell[13] - cell[4] * cell[1] * cell[15] +
			cell[4] * cell[3] * cell[13] + cell[12] * cell[1] * cell[7] - cell[12] * cell[3] * cell[5],
			-cell[0] * cell[5] * cell[11] + cell[0] * cell[7] * cell[9] + cell[4] * cell[1] * cell[11] -
			cell[4] * cell[3] * cell[9] - cell[8] * cell[1] * cell[7] + cell[8] * cell[3] * cell[5],
			-cell[4] * cell[9] * cell[14] + cell[4] * cell[10] * cell[13] + cell[8] * cell[5] * cell[14] -
			cell[8] * cell[6] * cell[13] - cell[12] * cell[5] * cell[10] + cell[12] * cell[6] * cell[9],
			cell[0] * cell[9] * cell[14] - cell[0] * cell[10] * cell[13] - cell[8] * cell[1] * cell[14] +
			cell[8] * cell[2] * cell[13] + cell[12] * cell[1] * cell[10] - cell[12] * cell[2] * cell[9],
			-cell[0] * cell[5] * cell[14] + cell[0] * cell[6] * cell[13] + cell[4] * cell[1] * cell[14] -
			cell[4] * cell[2] * cell[13] - cell[12] * cell[1] * cell[6] + cell[12] * cell[2] * cell[5],
			cell[0] * cell[5] * cell[10] - cell[0] * cell[6] * cell[9] - cell[4] * cell[1] * cell[10] +
			cell[4] * cell[2] * cell[9] + cell[8] * cell[1] * cell[6] - cell[8] * cell[2] * cell[5]
		};
		const float det = cell[0] * inv[0] + cell[1] * inv[4] + cell[2] * inv[8] + cell[3] * inv[12];
		Mat4 retVal;
		if (det != 0)
		{
			const float invdet = 1.0f / det;
			for (int i = 0; i < 16; i++) retVal.cell[i] = inv[i] * invdet;
		}
		return retVal;
	}

	inline __host__ __device__ Mat4 Inverted3x3() const
	{
		// via https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c
		const float invdet = 1.0f / (cell[0] * (cell[5] * cell[10] - cell[6] * cell[9]) -
		 cell[4] * (cell[1] * cell[10] - cell[9] * cell[2]) +
		 cell[8] * (cell[1] * cell[6] - cell[5] * cell[2]));
		Mat4 R;
		R.cell[0] = (cell[5] * cell[10] - cell[6] * cell[9]) * invdet;
		R.cell[4] = (cell[8] * cell[6] - cell[4] * cell[10]) * invdet;
		R.cell[8] = (cell[4] * cell[9] - cell[8] * cell[5]) * invdet;
		R.cell[1] = (cell[9] * cell[2] - cell[1] * cell[10]) * invdet;
		R.cell[5] = (cell[0] * cell[10] - cell[8] * cell[2]) * invdet;
		R.cell[9] = (cell[1] * cell[8] - cell[0] * cell[9]) * invdet;
		R.cell[2] = (cell[1] * cell[6] - cell[2] * cell[5]) * invdet;
		R.cell[6] = (cell[2] * cell[4] - cell[0] * cell[6]) * invdet;
		R.cell[10] = (cell[0] * cell[5] - cell[1] * cell[4]) * invdet;
		return R;
	}

	inline __host__ __device__ float3 TransformVector(const float3& v) const
	{
		return make_float3( cell[0] * v.x + cell[1] * v.y + cell[2] * v.z,
			cell[4] * v.x + cell[5] * v.y + cell[6] * v.z,
			cell[8] * v.x + cell[9] * v.y + cell[10] * v.z );
	}

	inline __host__ __device__ float3 TransformPoint(const float3& v) const
	{
		return make_float3(
			cell[0] * v.x + cell[1] * v.y + cell[2] * v.z + cell[3],
			cell[4] * v.x + cell[5] * v.y + cell[6] * v.z + cell[7],
			cell[8] * v.x + cell[9] * v.y + cell[10] * v.z + cell[11] );
	}
};

Mat4 operator * (const Mat4& a, const Mat4& b);
Mat4 operator + (const Mat4& a, const Mat4& b);
Mat4 operator * (const Mat4& a, const float s);
Mat4 operator * (const float s, const Mat4& a);
bool operator == (const Mat4& a, const Mat4& b);
bool operator != (const Mat4& a, const Mat4& b);
float4 operator * (const Mat4& a, const float4& b);
float4 operator * (const float4& a, const Mat4& b);

float3 TransformPosition(const float3& a, const Mat4& M);
float3 TransformVector(const float3& a, const Mat4& M);
