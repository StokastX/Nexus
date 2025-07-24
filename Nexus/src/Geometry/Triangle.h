#pragma once
#include "Utils/cuda_math.h"
#include "Assets/Material.h"
#include "Cuda/Geometry/Triangle.cuh"


struct TriangleData
{
	TriangleData() = default;
	TriangleData(float3 n0, float3 n1, float3 n2, float3 t0, float3 t1, float3 t2, float2 uv0, float2 uv1, float2 uv2)
		: normal0(n0), normal1(n1), normal2(n2), tangent0(t0), tangent1(t1), tangent2(t2), texCoord0(uv0), texCoord1(uv1), texCoord2(uv2) { }

	// Normals
	float3 normal0;
	float3 normal1;
	float3 normal2;

	// Tangents
	float3 tangent0;
	float3 tangent1;
	float3 tangent2;

	// Texture coordinates
	float2 texCoord0;
	float2 texCoord1;
	float2 texCoord2;
};
