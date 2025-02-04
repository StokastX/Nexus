#pragma once
#include "Utils/cuda_math.h"
#include "Assets/Material.h"
#include "Cuda/Geometry/Triangle.cuh"


struct Triangle
{
	// Positions
	float3 pos0;
	float3 pos1;
	float3 pos2;

	Triangle() = default;

	Triangle(
		float3 p0, float3 p1, float3 p2,
		float3 n0 = make_float3(0.0f), float3 n1 = make_float3(0.0f), float3 n2 = make_float3(0.0f),
		float2 t0 = make_float2(0.0f), float2 t1 = make_float2(0.0f), float2 t2 = make_float2(0.0f)
	): pos0(p0), pos1(p1), pos2(p2) { }

	static D_Triangle ToDevice(const Triangle& triangle)
	{
		return D_Triangle{ triangle.pos0, triangle.pos1, triangle.pos2 };
	}
};

struct TriangleData
{
	TriangleData() = default;
	TriangleData(float3 n0, float3 n1, float3 n2, float2 t0, float2 t1, float2 t2)
		: normal0(n0), normal1(n1), normal2(n2), texCoord0(t0), texCoord1(t1), texCoord2(t2) { }

	// Normals
	float3 normal0;
	float3 normal1;
	float3 normal2;

	// Texture coordinates
	float2 texCoord0;
	float2 texCoord1;
	float2 texCoord2;
};
