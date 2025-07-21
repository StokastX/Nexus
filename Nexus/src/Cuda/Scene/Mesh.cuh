#pragma once
#include "Cuda/Geometry/Triangle.cuh"
#include "Geometry/BVH/BVH.h"

struct D_Mesh
{
	NXB::BVH bvh;
	NXB::Triangle* triangles;
	D_TriangleData* triangleData;
};