#pragma once

#include <cuda_runtime_api.h>
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"

namespace Nexus {

	struct D_BSDF
	{

		template<typename T>
		inline __device__ static bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
		{
			T bsdf;
			bsdf.PrepareBSDFData(wi, material);
			return bsdf.Sample(material, wi, wo, throughput, pdf, rngState);
		}

		template<typename T>
		inline __device__ static bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
		{
			T bsdf;
			bsdf.PrepareBSDFData(wi, material);
			return bsdf.Eval(material, wi, wo, throughput, pdf);
		}
	};

}