#pragma once

#include "PlasticBSDF.cuh"
#include "DielectricBSDF.cuh"
#include "ConductorBSDF.cuh"
#include "Cuda/Random.cuh"
#include "Cuda/Sampler.cuh"

class D_PrincipledBSDF
{
public:
	static inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		throughput = make_float3(0.0f);
		pdf = 0.0f;

		if (material.metalness > 0.0f)
		{
			D_ConductorBSDF bsdf;
			bsdf.PrepareBSDFData(wi, material);
			float3 metalThroughput;
			float metalPdf;
			if (bsdf.Eval(material, wi, wo, metalThroughput, metalPdf))
			{
				throughput += material.metalness * metalThroughput;
				pdf += material.metalness * metalPdf;
			}
		}
		if (material.transmission > 0.0f)
		{
			D_DielectricBSDF bsdf;
			bsdf.PrepareBSDFData(wi, material);
			float3 dielectricThroughput = make_float3(0.0f);
			float dielectricPdf = 0.0f;
			if (bsdf.Eval(material, wi, wo, dielectricThroughput, dielectricPdf))
			{
				throughput += (1.0f - material.metalness) * material.transmission * dielectricThroughput;
				pdf += (1.0f - material.metalness) * material.transmission * dielectricPdf;
			}
		}
		D_PlasticBSDF bsdf;
		bsdf.PrepareBSDFData(wi, material);
		float3 plasticThroughput = make_float3(0.0f);
		float plasticPdf = 0.0f;
		if (bsdf.Eval(material, wi, wo, plasticThroughput, plasticPdf))
		{
			throughput += (1.0f - material.metalness) * (1.0f - material.transmission) * plasticThroughput;
			pdf += (1.0f - material.metalness) * (1.0f - material.transmission) * plasticPdf;
		}

		return Sampler::IsPdfValid(pdf);
	}

	static inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		if (Random::Rand(rngState) < material.metalness)
		{
			D_ConductorBSDF bsdf;
			bsdf.PrepareBSDFData(wi, material);
			bool scattered = bsdf.Sample(material, wi, wo, throughput, pdf, rngState);
			pdf *= material.metalness;
			return scattered && Sampler::IsPdfValid(pdf);
		}
		else
		{
			if (Random::Rand(rngState) < material.transmission)
			{

				D_DielectricBSDF bsdf;
				bsdf.PrepareBSDFData(wi, material);
				bool scattered = bsdf.Sample(material, wi, wo, throughput, pdf, rngState);
				pdf *= (1.0f - material.metalness) * material.transmission;
				return scattered && Sampler::IsPdfValid(pdf);
			}
			else
			{
				D_PlasticBSDF bsdf;
				bsdf.PrepareBSDFData(wi, material);
				bool scattered = bsdf.Sample(material, wi, wo, throughput, pdf, rngState);
				pdf *= (1.0f - material.metalness) * (1.0f - material.transmission);
				return scattered && Sampler::IsPdfValid(pdf);
			}
		}
	}
};