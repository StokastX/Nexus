#pragma once
#include "Cuda/Geometry/Triangle.cuh"
#include "Geometry/BVH/BVH.h"

namespace Nexus {

	struct D_Mesh
	{
		NXB::D_BVH bvh;
		NXB::Triangle* triangles;
		TriangleData* triangleData;
	};

}