#pragma once

#include <iostream>
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/AABB.cuh"
#include "Math/Mat4.h"


struct D_MeshInstance
{
	uint32_t meshIdx;
	uint32_t materialIdx;

	Mat4 transform;
	Mat4 invTransform;
	D_AABB bounds;
};