#pragma once
#include "Cuda/BVH/BVH.cuh"
#include "Cuda/Geometry/Triangle.cuh"

struct D_Mesh
{
	uint32_t bvhIdx;
	D_Triangle* triangles;
	D_TriangleData* triangleData;
};