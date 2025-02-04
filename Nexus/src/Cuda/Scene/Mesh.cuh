#pragma once
#include "Cuda/BVH/BVH.cuh"
#include "Cuda/Geometry/Triangle.cuh"
#include "BVH/BVH.h"

struct D_Mesh
{
	NXB::BVH bvh;
	D_Triangle* triangles;
	D_TriangleData* triangleData;
};