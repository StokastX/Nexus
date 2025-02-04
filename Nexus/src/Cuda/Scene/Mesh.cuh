#pragma once
#include "Cuda/BVH/BVH.cuh"
#include "Cuda/Geometry/Triangle.cuh"
#include "BVH/BVH.h"

struct D_Mesh
{
	NXB::BVH bvh;
	NXB::Triangle* triangles;
	D_TriangleData* triangleData;
};