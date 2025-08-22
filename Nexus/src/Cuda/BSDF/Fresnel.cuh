#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Utils/ColorUtils.h"
#include "Cuda/Utils.cuh"

class Fresnel
{
public:
	// Fresnel reflectance for dieletric materials. See https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
	// cosThetaI should be positive
	inline static __device__ float DielectricReflectance(float eta, float cosThetaI)
	{
		const float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

		if (sinThetaTSq >= 1.0f)
			return 1.0f;

		const float cosThetaT = sqrtf(1.0f - sinThetaTSq);

		const float Rparl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
		const float Rperp = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);

		return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
	}

	// Portsmouth, Kutz, and Hill 2025, "OpenPBR: Novel Features and Implementation Details"
	// specular_weight (>= 0) modulates the Fresnel F0 linearly, without disturbing TIR/refraction
	inline static __device__ float DielectricReflectance(float eta, float cosThetaI, float specularWeight)
	{
		if (specularWeight == 1.0f)
			return DielectricReflectance(eta, cosThetaI);

		float F0 = Square((eta - 1.0f) / (eta + 1.0f));
		float eps = copysignf(fmin(1.0f, sqrtf(specularWeight * F0)), 1.0f - eta);
		float etaPrime = (1.0f - eps) / fmax(FLT_EPSILON, 1.0f + eps);

		// I changed etaPrime to eta because in case specularWeight is zero,
		// etaPrime is equal to 1 while eta can be greater than one which leads to false positives
		if (eta <= 1.0f) // (No TIR possible)
			return DielectricReflectance(etaPrime, cosThetaI);

		float cosThetaTSq = 1.0f - (1.0f - Square(cosThetaI)) * Square(eta);
		if (cosThetaTSq <= 0.0f)
			return 1.0f; // (TIR occurs)
		return DielectricReflectance(1.0f / eta, sqrtf(cosThetaTSq));
	}

	// Schlick reflectance approximation
	inline static __device__ float SchlickDielectricReflectance(const float F0, const float F90, const float cosThetaI)
	{
		return F0 + (F90 - F0) * fifthPower(1.0f - cosThetaI);
	}

	// Modified fresnel reflectance at grazing angles (90�) depending on R0. See https://boksajak.github.io/files/CrashCourseBRDF.pdf
	inline static __device__ float ShadowedF90(const float3& F0) {
		const float t = (1.0f / 0.04f);
		return min(1.0f, t * ColorUtils::Luminance(F0));
	}

	inline static __device__ float3 SchlickMetallicReflectance(const float3& F0, const float cosTheta)
	{
		return F0 + (1.0f - F0) * fifthPower(1.0f - cosTheta);
	}

	// From Kutz et al. 2021. "Novel aspects of the Adobe Standard Material"
	inline static __device__ float3 SchlickMetallicF82(const float3& F0, const float3& F82, const float cosTheta)
	{
		constexpr float cosThetaMax = 1.0f / 7.0f;
		constexpr float oneMinusCosThetaMax = 1.0f - cosThetaMax;
		constexpr float oneMinusCosThetaMaxToTheFifth = fifthPower(oneMinusCosThetaMax);
		constexpr float oneMinusCosThetaMaxToTheSixth = sixthPower(oneMinusCosThetaMax);
		const float3 r = F0;
		const float3 t = F82;
		const float3 whiteMinusR = make_float3(1.0f) - r;
		const float3 whiteMinusT = make_float3(1.0f) - t;
		const float3 bNumerator = (r + whiteMinusR * oneMinusCosThetaMaxToTheFifth) * whiteMinusT;
		constexpr float bDenominator = cosThetaMax * oneMinusCosThetaMaxToTheSixth;
		constexpr float bDenominatorReciprocal = 1.0f / bDenominator;
		const float3 b = bNumerator * bDenominatorReciprocal;
		const float oneMinusCosTheta = 1.0f - cosTheta;
		const float3 offsetFromR = (whiteMinusR - b * cosTheta * oneMinusCosTheta) * fifthPower(oneMinusCosTheta);
		const float3 FTheta = r + offsetFromR;
		return clamp(FTheta, 0.0f, 1.0f);
	}

	inline static __device__ float ComplexReflectance(float cosThetaI, const float eta, const float k)
	{
		cosThetaI = clamp(cosThetaI, 0.0f, 1.0f);

		float cosThetaISq = cosThetaI * cosThetaI;
		float sinThetaISq = max(1.0f - cosThetaISq, 0.0f);
		float sinThetaIQu = sinThetaISq * sinThetaISq;

		float innerTerm = eta * eta - k * k - sinThetaISq;
		float aSqPlusBSq = sqrtf(max(innerTerm * innerTerm + 4.0f * eta * eta * k * k, 0.0f));
		float a = sqrtf(max((aSqPlusBSq + innerTerm) * 0.5f, 0.0f));

		float Rs = ((aSqPlusBSq + cosThetaISq) - (2.0f * a * cosThetaI)) /
			((aSqPlusBSq + cosThetaISq) + (2.0f * a * cosThetaI));
		float Rp = ((cosThetaISq * aSqPlusBSq + sinThetaIQu) - (2.0f * a * cosThetaI * sinThetaISq)) /
			((cosThetaISq * aSqPlusBSq + sinThetaIQu) + (2.0f * a * cosThetaI * sinThetaISq));

		return 0.5f * (Rs + Rs * Rp);
	}

	inline static __device__ float3 ComplexReflectance(float cosThetaI, const float3 eta, const float3 k)
	{
		return make_float3(ComplexReflectance(cosThetaI, eta.x, k.x), ComplexReflectance(cosThetaI, eta.y, k.y), ComplexReflectance(cosThetaI, eta.z, k.z));
	}
};