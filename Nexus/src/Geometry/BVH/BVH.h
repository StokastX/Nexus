#pragma once
#include "NXB/BVHBuilder.h"

#define USE_BVH8

namespace NXB
{
#ifdef USE_BVH8
	using BVH = BVH8;
#else
	using BVH = BVH2;
#endif
}