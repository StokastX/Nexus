#pragma once
#include "Utils/cuda_math.h"
#include "Assets/Material.h"
#include "Cuda/Geometry/Triangle.cuh"


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
