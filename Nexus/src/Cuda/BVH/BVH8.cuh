#pragma once

#include <stdint.h>
#include "Cuda/Geometry/Triangle.cuh"

namespace Nexus {

	// Compressed stack entry (we don't use it in the traversal algorithm, instead we use a uint2 for performance)
	struct D_StackEntry
	{
		struct D_Internal
		{
			// Base node index in this entry
			uint32_t childBaseIdx;

			// Field indicating for each node if it has not already been traversed (1 if not traversed)
			uint8_t hits;

			// Dummy
			uint8_t pad;

			// imask of the parent node
			uint8_t imask;
		};

		struct D_Triangle
		{
			// Base triangle index
			uint32_t triangleBaseIdx;

			// Dummy
			unsigned pad : 8;

			// Field indicating for each triangle if it has not already been traversed (1 if not traversed)
			unsigned triangleHits : 24;
		};

		union
		{
			D_Internal internal;
			D_Triangle triangle;
		};
	};

}
