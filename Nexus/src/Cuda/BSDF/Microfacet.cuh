#pragma once

#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "Utils/cuda_math.h"
#include "Cuda/Random.cuh"
#include "Cuda/Utils.cuh"

class Microfacet 
{
public:

	inline static __device__ float BeckmannD(const float alpha, const float mDotN)
	{
		const float alphaSq = alpha * alpha;
		const float cosThetaSq = mDotN * mDotN;
		const float numerator = exp((cosThetaSq - 1.0f) / (alphaSq * cosThetaSq));
		const float denominator = PI * alphaSq * cosThetaSq * cosThetaSq;
		return numerator / denominator;
	}

	// Heitz 2018, "Sampling the GGX Distribution of Visible Normals"
	inline static __device__ float D_GGXAnisotropic(const float3& m, const float2 alpha)
	{
		float denom = Square(m.x / alpha.x) + Square(m.y / alpha.y) + Square(m.z);
		return 1.0f / (PI * alpha.x * alpha.y * Square(denom));
	}

	inline static __device__ float Lambda_GGX(const float3 wi, const float2 alpha)
	{
		return 0.5f * sqrtf(1.0f + (Square(alpha.x * wi.x) + Square(alpha.y * wi.y)) / Square(wi.z)) - 0.5f;
	}

	inline static __device__ float G1_GGX(const float3& wi, const float2 alpha)
	{
		return 1.0f / (1.0f + Lambda_GGX(wi, alpha));
	}

	// Height correlated masking-shadowing function
	// Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
	inline static __device__ float G2_GGX(const float3& wi, const float3& wo, const float2 alpha)
	{
		return 1.0f / (1.0f + Lambda_GGX(wi, alpha) + Lambda_GGX(wo, alpha));
	}

	// The VNDF is exactly the pdf of our sampling (assuming wi and m lie in the same hemisphere)
	inline static __device__ float VNDF_GGX(float D, float G1, const float3& wi, const float3& m)
	{
		return G1 * dot(wi, m) * D / wi.z;
	}

	inline static __device__ float ReflectionPdf_GGX(float D, float G1, float wiDotN)
	{
		// VNDF * Jacobian of the reflection operator
		return G1 * D / (4.0f * wiDotN);
	}

	inline static __device__ float RefractionPdf_GGX(float D, float G1, float eta, float wiDotN, float wiDotM, float woDotM)
	{
		// VNDF * Jacobian of the refraction operator
		return G1 * D * fabs(wiDotM * woDotM) / (fabs(wiDotN) * Square(eta * wiDotM + woDotM));
	}

	inline static __device__ float Smith_G_a(const float alpha, const float sDotN) {
		return sDotN / (alpha * sqrt(1.0f - min(0.99999f, sDotN * sDotN)));
	}

	inline static __device__ float Smith_G1_Beckmann_Walter(const float a) {
		if (a < 1.6f) {
			return ((3.535f + 2.181f * a) * a) / (1.0f + (2.276f + 2.577f * a) * a);
		}
		else {
			return 1.0f;
		}
	}

	inline static __device__ float Smith_G1_Beckmann_Walter(const float alpha, const float sDotN, const float alphaSquared, const float NdotSSquared)
	{
		return Smith_G1_Beckmann_Walter(Smith_G_a(alpha, sDotN));
	}

	inline static __device__ float Smith_G2(const float alpha, const float woDotN, const float wiDotN)
	{
		float aL = Smith_G_a(alpha, woDotN);
		float aV = Smith_G_a(alpha, wiDotN);
		return Smith_G1_Beckmann_Walter(aL) * Smith_G1_Beckmann_Walter(aV);
	}

	// Weight of the sample for a Walter-Beckmann sampling
	// See https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
	inline static __device__ float WeightBeckmannWalter(
		const float alpha, const float wiDotM, const float woDotN,
		const float wiDotN, const float mDotN
	) {
		return (wiDotM * Smith_G2(alpha, woDotN, wiDotN)) / (wiDotN * mDotN);
	}

	inline static __device__ float SampleWalterReflectionPdf(const float alpha, const float mDotN, const float wiDotM)
	{
		return BeckmannD(alpha, mDotN) * mDotN / (4.0f * wiDotM);
	}

	inline static __device__ float SampleWalterRefractionPdf(const float alpha, const float mDotN, const float wiDotM, const float woDotM, const float eta)
	{
		return BeckmannD(alpha, mDotN) * mDotN * woDotM / Square(eta * wiDotM + woDotM);
	}

	inline static __device__ float3 SampleSpecularHalfBeckWalt(const float alpha, unsigned int& rngState)
	{
		const float a = dot(make_float2(alpha), make_float2(0.5f, 0.5f));

		const float2 u = make_float2(Random::Rand(rngState), Random::Rand(rngState));
		const float tanThetaSquared = -(a * a) * log(1.0f - u.x);
		const float phi = TWO_TIMES_PI * u.y;

		// Calculate cosTheta and sinTheta needed for conversion to m vector
		const float cosTheta = 1.0 / sqrt(1.0f + tanThetaSquared);
		const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

		// Convert sampled spherical coordinates to m vector
		return normalize(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
	}

	// Sampling the visible hemisphere using its cross section
	// Heitz 2018, "Sampling the GGX Distribution of Visible Normals"
	inline static __device__ float3 SampleVndf_Heitz(float3 wi, uint32_t& rngState)
	{
		float2 u = make_float2(Random::Rand(rngState), Random::Rand(rngState));
		// orthonormal basis (with special case if cross product is 0)
		float tmp = wi.x * wi.x + wi.y * wi.y;
		float3 w1 = tmp > 0.0f ? make_float3(-wi.y, wi.x, 0) * rsqrtf(tmp)
			: make_float3(1, 0, 0);
		float3 w2 = cross(wi, w1);
		// parameterization of the cross section
		float phi = 2.0f * PI * u.x;
		float r = sqrt(u.y);
		float t1 = r * cos(phi);
		float t2 = r * sin(phi);
		float s = (1.0f + wi.z) / 2.0f;
		t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
		float ti = sqrt(max(1.0f - t1 * t1 - t2 * t2, 0.0f));
		// reprojection onto hemisphere
		float3 wm = t1 * w1 + t2 * w2 + ti * wi;
		// return hemispherical sample
		return wm;
	}

	// Sampling the visible hemisphere as half vectors
	// Dupuy and Benyoub 2023, "Sampling Visible GGX Normals with Spherical Caps"
	inline static __device__ float3 SampleVndf_SphericalCaps(float3 wi, uint32_t& rngState)
	{
		float2 u = make_float2(Random::Rand(rngState), Random::Rand(rngState));
		// sample a spherical cap in (-wi.z, 1]
		float phi = 2.0f * PI * u.x;
		float z = fma((1.0f - u.y), (1.0f + wi.z), -wi.z);
		float sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
		float x = sinTheta * cos(phi);
		float y = sinTheta * sin(phi);
		float3 c = make_float3(x, y, z);
		// compute halfway direction;
		float3 h = c + wi;
		// return without normalization (as this is done later)
		return h;
	}

	inline static __device__ float3 SampleVNDF_GGX(float3 wi, float2 alpha, uint32_t& rngState)
	{
		// warp to the hemisphere configuration
		float3 wiStd = normalize(make_float3(wi.x * alpha.x, wi.y * alpha.y, wi.z));
		// sample the hemisphere (see implementation 2 or 3)
		float3 wmStd = SampleVndf_Heitz(wiStd, rngState);
		// warp back to the ellipsoid configuration
		float3 wm = normalize(make_float3(wmStd.x * alpha.x, wmStd.y * alpha.y, wmStd.z));
		// return final normal
		return wm;
	}
};