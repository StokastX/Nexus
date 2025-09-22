#pragma once

#include <iostream>
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/AABB.cuh"
#include "Math/Mat4.h"


namespace Nexus {

	struct D_MeshInstance
	{
		uint32_t meshIdx;
		uint32_t materialIdx;

		Mat4 transform;
		Mat4 invTransform;
		NXB::AABB bounds;
	};

}