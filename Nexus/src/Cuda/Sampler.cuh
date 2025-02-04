#pragma once
#include <cuda_runtime_api.h>
#include "Cuda/BVH/BVH8.cuh"
#include "Cuda/Scene/Light.cuh"
#include "Random.cuh"

class Sampler
{
public:

	// Balance heuristic for multiple importance sampling with two
	// samples from two different sampling strategies. (We use BSDF and direct light sampling)
	// See Eric Veach's thesis
	inline __device__ static float BalanceHeuristic(const float pdf1, const float pdf2)
	{
		return pdf1 / (pdf1 + pdf2);
	}

	// Power heuristic with beta = 2 for multiple importance sampling with two
	// samples from two different sampling strategies. (We use BSDF and direct light sampling)
	// See Eric Veach's thesis
	inline __device__ static float PowerHeuristic(const float pdf1, const float pdf2)
	{
		return pdf1 * pdf1 / (pdf1 * pdf1 + pdf2 * pdf2);
	}

	// Uniform sampling between 0 and max - 1
	inline __device__ static uint32_t Uniform(uint32_t max, uint32_t& rngState)
	{
		return floor(Random::Rand(rngState) * max);
	}

	// Sample a scene light uniformly (lights intensity or area are not considered)
	inline __device__ static D_Light UniformSampleLights(D_Light* lights, uint32_t lightCount, uint32_t& rngState)
	{
		const uint32_t lightIdx = Uniform(lightCount, rngState);
		return lights[lightIdx];
	}

	// Uniform triangle sampling.
	// See https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformSampleTriangle
	inline __device__ static float2 UniformSampleTriangle(uint32_t& rngState)
	{
		const float a = Random::Rand(rngState);
		const float b = Random::Rand(rngState);
		const float su0 = sqrtf(a);

		return make_float2(1 - su0, b * su0);
	}

	// Sample a triangle uniformly based on the triangle count, then sample a point uniformly on the surface of this triangle
	inline __device__ static void UniformSampleMesh(uint32_t primCount, uint32_t& rngState, uint32_t& triangleIdx, float2& uv)
	{
		triangleIdx = Uniform(primCount, rngState);
		uv = UniformSampleTriangle(rngState);
	}

	inline __device__ static bool IsPdfValid(const float pdf)
	{
		return isfinite(pdf) && pdf > 1.0e-4f;
	}
};