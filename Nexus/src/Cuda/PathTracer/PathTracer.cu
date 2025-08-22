#include "PathTracer.cuh"
#include "Cuda/Random.cuh"
#include "Cuda/BSDF/LambertianBSDF.cuh"
#include "Cuda/BSDF/DielectricBSDF.cuh"
#include "Cuda/BSDF/PlasticBSDF.cuh"
#include "Cuda/BSDF/ConductorBSDF.cuh"
#include "Cuda/BSDF/PrincipledBSDF.cuh"
#include "Cuda/BSDF/BSDF.cuh"
#include "Utils/cuda_math.h"
#include "Math/TangentFrame.h"
#include "Utils/Utils.h"
#include "texture_indirect_functions.h"
#include "Cuda/Scene/Scene.cuh"
#include "Cuda/Scene/Camera.cuh"
#include "Cuda/Sampler.cuh"
#include "NXB/BVHBuilder.h"
#include "Cuda/BVH/BVH2Traversal.cuh"
#include "Cuda/BVH/BVH8Traversal.cuh"
#include "Utils/ColorUtils.h"

__device__ __constant__ uint32_t frameNumber;
__device__ __constant__ uint32_t bounce;

__device__ __constant__ float3* accumulationBuffer;
__device__ __constant__ uint32_t* renderBuffer;

__device__ __constant__ D_Scene scene;
__device__ __constant__ NXB::BVH tlas;
__device__ __constant__ D_Mesh* meshes;
__device__ __constant__ D_PathStateSOA pathState;
__device__ __constant__ D_TraceRequestSOA traceRequest;
__device__ __constant__ D_ShadowTraceRequestSOA shadowTraceRequest;

__device__ __constant__ D_MaterialRequestSOA materialRequest;

__device__ D_PixelQuery pixelQuery;
__device__ D_QueueSize queueSize;

// If necessary, sample the HDR map (from spherical to equirectangular projection)
inline __device__ float3 SampleBackground(const D_Scene& scene, float3 direction)
{
	float3 backgroundColor;
	if (scene.hasHdrMap)
	{
		// Theta goes from -PI to PI, phi from -PI/2 to PI/2
		const float theta = atan2(direction.z, direction.x);
		const float phi = asin(direction.y);

		// Equirectangular projection
		const float u = (theta + PI) * INV_PI * 0.5;
		const float v = 1.0f - (phi + PI * 0.5f) * INV_PI;

		backgroundColor = make_float3(tex2D<float4>(scene.hdrMap, u, v)) * scene.renderSettings.backgroundIntensity;
	}
	else
		backgroundColor = scene.renderSettings.backgroundColor * scene.renderSettings.backgroundIntensity;
	return backgroundColor;
}

__global__ void GenerateKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const D_Camera camera = scene.camera;
	uint2 resolution = camera.resolution;

	if (index >= resolution.x * resolution.y)
		return;

	const uint32_t j = index / resolution.x;
	const uint32_t i = index - j * resolution.x;

	const uint2 pixel = make_uint2(i, j);

	unsigned int rngState = Random::InitRNG(pixel, resolution, frameNumber);

	// Normalized jittered coordinates
	const float x = (pixel.x + Random::Rand(rngState)) / (float)resolution.x;
	const float y = (pixel.y + Random::Rand(rngState)) / (float)resolution.y;

	float2 rd = camera.lensRadius * Random::RandomInUnitDisk(rngState);
	float3 offset = camera.right * rd.x + camera.up * rd.y;

	D_Ray ray(
		camera.position + offset,
		normalize(camera.lowerLeftCorner + x * camera.viewportX + y * camera.viewportY - camera.position - offset)
	);

	if (index == 0)
		queueSize.traceSize[0] = resolution.x * resolution.y;

	traceRequest.ray.origin[index] = ray.origin;
	traceRequest.ray.direction[index] = ray.direction;
	traceRequest.pixelIdx[index] = index;
}


__global__ void TraceKernel()
{
#ifdef USE_BVH8
	if (!scene.renderSettings.visualizeBvh)
		BVH8Trace(tlas, meshes, scene.meshInstances, traceRequest, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
	else
		BVH8TraceVisualize(tlas, meshes, scene.meshInstances, traceRequest, pathState, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
#else
	if (!scene.renderSettings.visualizeBvh)
		BVH2Trace(tlas, meshes, scene.meshInstances, traceRequest, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
	else if (!scene.renderSettings.wireFrameBvh)
		BVH2TraceVisualize(tlas, meshes, scene.meshInstances, traceRequest, pathState, bounce, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
	else
		BVH2TraceVisualizeWireframe(tlas, meshes, scene.meshInstances, traceRequest, pathState, bounce, queueSize.traceSize[bounce], &queueSize.traceCount[bounce]);
#endif
}

__global__ void TraceShadowKernel()
{
#ifdef USE_BVH8
	BVH8TraceShadow(tlas, meshes, scene.meshInstances, shadowTraceRequest, queueSize.traceShadowSize[bounce], &queueSize.traceShadowCount[bounce], pathState.radiance);
#else
	BVH2TraceShadow(tlas, meshes, scene.meshInstances, shadowTraceRequest, queueSize.traceShadowSize[bounce], &queueSize.traceShadowCount[bounce], pathState.radiance);
#endif
}

__global__ void LogicKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= queueSize.traceSize[bounce - 1])
		return;

	uint32_t rngState = Random::InitRNG(index, scene.camera.resolution, frameNumber);

	const D_Intersection intersection = traceRequest.intersection.Get(index);
	const D_Ray ray(traceRequest.ray.origin[index], traceRequest.ray.direction[index]);
	const uint32_t pixelIdx = traceRequest.pixelIdx[index];

	const float3 throughput = bounce == 1 ? make_float3(1.0f) : pathState.throughput[pixelIdx];

	// If no intersection, sample background
	if (intersection.hitDistance == 1e30f)
	{
		if (!scene.renderSettings.visualizeBvh)
		{
			float3 backgroundColor = SampleBackground(scene, ray.direction);
			if (bounce == 1)
				pathState.radiance[pixelIdx] = throughput * backgroundColor;
			else
				pathState.radiance[pixelIdx] += throughput * backgroundColor;

			if (bounce == 1 && pixelQuery.pixelIdx == pixelIdx)
				pixelQuery.instanceIdx = -1;
		}

		return;
	}

	// Russian roulette
	float probability = fmaxf(throughput);// clamp(fmaxf(currentThroughput), 0.01f, 1.0f);
	if (Random::Rand(rngState) < probability)
	{
		// To get unbiased results, we need to increase the contribution of
		// the non-terminated rays with their probability of being terminated
		pathState.throughput[pixelIdx] = throughput / probability;
	}
	else
		return;

	int32_t requestIdx;
	requestIdx = atomicAdd(&queueSize.materialSize[bounce], 1);
	materialRequest.intersection.Set(requestIdx, intersection);
	materialRequest.rayDirection[requestIdx] = ray.direction;
	materialRequest.pixelIdx[requestIdx] = pixelIdx;
}


inline __device__ void NextEventEstimation(
	const float3 wi,
	const float3 rayDirection,
	const D_Material& material,
	const D_Intersection& intersection,
	const float3 hitPoint,
	const float3 normal,
	const float3 hitGNormal,
	const float3 throughput,
	const uint32_t pixelIdx,
	unsigned int& rngState
) {
	D_Light light = Sampler::UniformSampleLights(scene.lights, scene.lightCount, rngState);

	float weight = 1.0f;
	float lightPdf, hitDistance;
	float3 sampleThroughput;
	float3 emissive;
	D_Ray shadowRay;

	if (light.type == D_Light::Type::MESH)
	{
		D_MeshInstance instance = scene.meshInstances[light.mesh.meshId];

		uint32_t triangleIdx;
		float2 uv;
		Sampler::UniformSampleMesh(meshes[instance.meshIdx].bvh.primCount, rngState, triangleIdx, uv);

		NXB::Triangle triangle = meshes[instance.meshIdx].triangles[triangleIdx];
		D_TriangleData triangleData = meshes[instance.meshIdx].triangleData[triangleIdx];

		float3 p = Barycentric(triangle.v0, triangle.v1, triangle.v2, uv);
		p = instance.transform.TransformPoint(p);

		const float3 lightGNormal = normalize(instance.invTransform.Transposed().TransformVector(triangle.Normal()));

		float3 lightNormal = Barycentric(triangleData.normal0, triangleData.normal1, triangleData.normal2, uv);
		lightNormal = normalize(instance.invTransform.Transposed().TransformVector(lightNormal));

		float3 toLight = p - hitPoint;

		const bool reflect = dot(-rayDirection, hitGNormal) * dot(toLight, hitGNormal) > 0.0f;

		if (!reflect && material.transmission == 0.0f)
			return;

		float offsetDirection = Utils::SgnE(dot(toLight, normal));
		shadowRay.origin = OffsetRay(hitPoint, hitGNormal * offsetDirection);

		offsetDirection = Utils::SgnE(dot(-toLight, lightNormal));
		p = OffsetRay(p, lightGNormal * offsetDirection);

		float3 offsetRay = p - shadowRay.origin;
		hitDistance = length(offsetRay);
		shadowRay.direction = offsetRay / hitDistance;
		shadowRay.invDirection = 1.0f / shadowRay.direction;

		TangentFrame tangentFrame(normal);
		const float3 wo = tangentFrame.WorldToLocal(shadowRay.direction);

		const float cosThetaO = fabs(dot(lightNormal, shadowRay.direction));

		const float dSquared = dot(toLight, toLight);

		const NXB::Triangle triangleTransformed(
			instance.transform.TransformPoint(triangle.v0),
			instance.transform.TransformPoint(triangle.v1),
			instance.transform.TransformPoint(triangle.v2)
		);

		lightPdf = 1.0f / (scene.lightCount * meshes[instance.meshIdx].bvh.primCount * triangleTransformed.Area());
		// Transform pdf over an area to pdf over directions
		lightPdf *= dSquared / cosThetaO;

		if (!Sampler::IsPdfValid(lightPdf))
			return;

		const D_Material lightMaterial = scene.materials[instance.materialIdx];

		float bsdfPdf;
		bool sampleIsValid = D_PrincipledBSDF::Eval(material, wi, wo, sampleThroughput, bsdfPdf);

		if (!sampleIsValid)
			return;

		weight = Sampler::PowerHeuristic(lightPdf, bsdfPdf);

		if (lightMaterial.emissiveMapId != -1)
		{
			float2 texUv = Barycentric(triangleData.texCoord0, triangleData.texCoord1, triangleData.texCoord2, uv);
			emissive = make_float3(tex2D<float4>(scene.textures[lightMaterial.emissiveMapId], texUv.x, texUv.y));
		}
		else
			emissive = lightMaterial.emissionColor;

		emissive *= lightMaterial.intensity;
	}

	else if (light.type == D_Light::Type::POINT)
	{
		float3 toLight = light.point.position - hitPoint;
		float dSquared = dot(toLight, toLight);
		lightPdf = 1.0f / scene.lightCount;
		lightPdf *= dSquared;
		if (!Sampler::IsPdfValid(lightPdf))
			return;

		emissive = light.point.color * light.point.intensity;

		const bool reflect = dot(-rayDirection, hitGNormal) * dot(toLight, hitGNormal) > 0.0f;

		if (!reflect && material.transmission == 0.0f)
			return;

		float offsetDirection = Utils::SgnE(dot(toLight, normal));
		shadowRay.origin = OffsetRay(hitPoint, hitGNormal * offsetDirection);

		hitDistance = length(toLight);
		shadowRay.direction = toLight / hitDistance;
		shadowRay.invDirection = 1.0f / shadowRay.direction;

		TangentFrame tangentFrame(normal);
		const float3 wo = tangentFrame.WorldToLocal(shadowRay.direction);

		float bsdfPdf;
		bool sampleIsValid = D_PrincipledBSDF::Eval(material, wi, wo, sampleThroughput, bsdfPdf);

		if (!sampleIsValid)
			return;
	}
	else if (light.type == D_Light::Type::DIRECTIONAL)
	{
		float3 toLight = -light.directional.direction;
		lightPdf = 1.0f / scene.lightCount;
		emissive = light.directional.color * light.directional.intensity;

		const bool reflect = dot(-rayDirection, hitGNormal) * dot(toLight, hitGNormal) > 0.0f;

		if (!reflect && material.transmission == 0.0f)
			return;

		float offsetDirection = Utils::SgnE(dot(toLight, normal));
		shadowRay.origin = OffsetRay(hitPoint, hitGNormal * offsetDirection);
		shadowRay.direction = normalize(toLight);
		shadowRay.invDirection = 1.0f / shadowRay.direction;
		hitDistance = 1.0e30f;

		TangentFrame tangentFrame(normal);
		const float3 wo = tangentFrame.WorldToLocal(shadowRay.direction);

		float bsdfPdf;
		bool sampleIsValid = D_PrincipledBSDF::Eval(material, wi, wo, sampleThroughput, bsdfPdf);

		if (!sampleIsValid)
			return;

	}
	else
		return;

	const float3 radiance = weight * throughput * sampleThroughput * emissive / lightPdf;

	const int32_t index = atomicAdd(&queueSize.traceShadowSize[bounce], 1);
	shadowTraceRequest.hitDistance[index] = hitDistance;
	shadowTraceRequest.radiance[index] = radiance;
	shadowTraceRequest.ray.Set(index, shadowRay);
	shadowTraceRequest.pixelIdx[index] = pixelIdx;
}


__global__ void MaterialKernel()
{
	const int32_t requestIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (requestIdx >= queueSize.materialSize[bounce])
		return;

	const D_Intersection intersection = materialRequest.intersection.Get(requestIdx);
	const float3 rayDirection = materialRequest.rayDirection[requestIdx];
	const uint32_t pixelIdx = materialRequest.pixelIdx[requestIdx];

	float3 throughput = bounce == 1 ? make_float3(1.0f) : pathState.throughput[pixelIdx];

	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t rngState = Random::InitRNG(index, scene.camera.resolution, frameNumber);

	const D_MeshInstance instance = scene.meshInstances[intersection.instanceIdx];
	const NXB::Triangle triangle = meshes[instance.meshIdx].triangles[intersection.triIdx];
	const D_TriangleData triangleData = meshes[instance.meshIdx].triangleData[intersection.triIdx];

	D_Material material = scene.materials[instance.materialIdx];

	const float2 uv = make_float2(intersection.u, intersection.v);

	float3 p = Barycentric(triangle.v0, triangle.v1, triangle.v2, uv);
	p = instance.transform.TransformPoint(p);

	float2 texUv = Barycentric(triangleData.texCoord0, triangleData.texCoord1, triangleData.texCoord2, uv);
	float3 normal = normalize(Barycentric(triangleData.normal0, triangleData.normal1, triangleData.normal2, uv));

	if (material.normalMapId != -1)
	{
		float3 texNormal = make_float3(tex2D<float4>(scene.textures[material.normalMapId], texUv.x, texUv.y));
		texNormal = normalize(2.0f * texNormal - 1.0f);

		float3 tangent = Barycentric(triangleData.tangent0, triangleData.tangent1, triangleData.tangent2, uv);
		TangentFrame tangentFrame(normal, tangent);
		normal = tangentFrame.LocalToWorld(texNormal);
	}

	// We use the transposed of the inverse matrix to transform normals.
	// See https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html
	normal = normalize(instance.invTransform.Transposed().TransformVector(normal));

	float3 gNormal = normalize(instance.invTransform.Transposed().TransformVector(triangle.Normal()));

	if (material.baseColorMapId != -1)
	{
		float4 color = tex2D<float4>(scene.textures[material.baseColorMapId], texUv.x, texUv.y);
		material.baseColor *= make_float3(color);
		material.opacity *= color.w;
	}
	if (material.emissiveMapId != -1)
		material.emissionColor *= make_float3(tex2D<float4>(scene.textures[material.emissiveMapId], texUv.x, texUv.y));
	if (material.roughnessMapId != -1)
		material.roughness *= tex2D<float4>(scene.textures[material.roughnessMapId], texUv.x, texUv.y).x;
	if (material.metalnessMapId != -1)
		material.metalness *= tex2D<float4>(scene.textures[material.metalnessMapId], texUv.x, texUv.y).x;
	if (material.metallicRoughnessMapId != -1)
	{
		float4 c = tex2D<float4>(scene.textures[material.metallicRoughnessMapId], texUv.x, texUv.y);
		material.roughness *= c.y;
		material.metalness *= c.z;
	}

	//if (pixelQuery.pixelIdx == pixelIdx && bounce == 1)
	//	printf("roughness: %f, metalness: %f\n", material.roughness, material.metalness);

	bool allowMIS = bounce > 1 && scene.renderSettings.useMIS;

	float3 radiance = make_float3(0.0f);

	if (fmaxf(material.emissionColor * material.intensity) > 0.0f)
	{
		float weight = 1.0f;

		// Not using MIS for primary rays
		if (allowMIS)
		{
			const float lastPdf = pathState.lastPdf[pixelIdx];

			const float cosThetaO = fabs(dot(normal, rayDirection));

			const float dSquared = Square(intersection.hitDistance);

			const NXB::Triangle triangleTransformed(
				instance.transform.TransformPoint(triangle.v0),
				instance.transform.TransformPoint(triangle.v1),
				instance.transform.TransformPoint(triangle.v2)
			);

			float lightPdf = 1.0f / (scene.lightCount * meshes[instance.meshIdx].bvh.primCount * triangleTransformed.Area());
			// Transform pdf over an area to pdf over directions
			lightPdf *= dSquared / cosThetaO;

			if (!Sampler::IsPdfValid(lightPdf))
				weight = 0.0f;
			else
				weight = Sampler::PowerHeuristic(lastPdf, lightPdf);
		}
		radiance = weight * material.emissionColor * material.intensity * throughput;
	}

	if (bounce == 1)
		pathState.radiance[pixelIdx] = radiance;
	else
		pathState.radiance[pixelIdx] += radiance;

	if (bounce == scene.renderSettings.pathLength)
		return;

	if (bounce == 1 && pixelQuery.pixelIdx == pixelIdx)
		pixelQuery.instanceIdx = intersection.instanceIdx;

	TangentFrame tangentFrame(normal);
	float3 wi = tangentFrame.WorldToLocal(-rayDirection);

	float3 wo;

	// Handle texture transparency
	if (Random::Rand(rngState) > material.opacity)
	{
		wo = tangentFrame.LocalToWorld(-wi);
		const float offsetDirection = Utils::SgnE(dot(wo, normal));
		const float3 offsetOrigin = OffsetRay(p, gNormal * offsetDirection);
		const D_Ray scatteredRay = D_Ray(offsetOrigin, wo);

		const int32_t traceRequestIdx = atomicAdd(&queueSize.traceSize[bounce], 1);
		traceRequest.ray.Set(traceRequestIdx, scatteredRay);
		traceRequest.pixelIdx[traceRequestIdx] = pixelIdx;
	}
	else
	{
		if (scene.renderSettings.useMIS && scene.lightCount > 0)
			NextEventEstimation(wi, rayDirection, material, intersection, p, normal, gNormal, throughput, pixelIdx, rngState);

		float pdf;
		float3 sampleThroughput;
		const bool scattered = D_PrincipledBSDF::Sample(material, wi, wo, sampleThroughput, pdf, rngState);

		if (!scattered)
			return;

		wo = tangentFrame.LocalToWorld(wo);

		const bool reflect = dot(-rayDirection, gNormal) * dot(wo, gNormal) > 0.0f;

		if (!reflect && material.transmission == 0.0f)
			return;

		const float offsetDirection = Utils::SgnE(dot(wo, normal));
		const float3 offsetOrigin = OffsetRay(p, gNormal * offsetDirection);
		const D_Ray scatteredRay = D_Ray(offsetOrigin, wo);

		// If sample is valid, write trace request in the path state
		throughput *= sampleThroughput;

		const int32_t traceRequestIdx = atomicAdd(&queueSize.traceSize[bounce], 1);
		traceRequest.ray.Set(traceRequestIdx, scatteredRay);
		traceRequest.pixelIdx[traceRequestIdx] = pixelIdx;

		pathState.throughput[pixelIdx] = throughput;
		pathState.lastPdf[pixelIdx] = pdf;
	}
}

__global__ void AccumulateKernel()
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const uint2 resolution = scene.camera.resolution;

	if (index >= resolution.x * resolution.y)
		return;

	if (frameNumber == 1)
		accumulationBuffer[index] = pathState.radiance[index];
	else
		accumulationBuffer[index] += (pathState.radiance[index] - accumulationBuffer[index]) / frameNumber;

	float3 color = accumulationBuffer[index] * exp2f(scene.renderSettings.exposure);
	switch (scene.renderSettings.toneMapping)
	{
	case ColorUtils::ToneMapping::ACES:
		color = ColorUtils::ACESToneMapping(color);
		break;
	case ColorUtils::ToneMapping::UNCHARTED2:
		color = ColorUtils::Uncharted2ToneMapping(color);
		break;
	case ColorUtils::ToneMapping::AGX_DEFAULT:
	case ColorUtils::ToneMapping::AGX_GOLDEN:
	case ColorUtils::ToneMapping::AGX_PUNCHY:
		color = ColorUtils::AgxToneMapping(color, scene.renderSettings.toneMapping);
		break;
	default:
		break;
	}
	renderBuffer[index] = ColorUtils::ToColorUInt(ColorUtils::LinearToGamma(color));
}

D_Scene* GetDeviceSceneAddress()
{
	D_Scene* deviceScene;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&deviceScene, scene));
	return deviceScene;
}

float3** GetDeviceAccumulationBufferAddress()
{
	float3** buffer;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&buffer, accumulationBuffer));
	return buffer;
}

uint32_t** GetDeviceRenderBufferAddress()
{
	uint32_t** buffer;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&buffer, renderBuffer));
	return buffer;
}

uint32_t* GetDeviceFrameNumberAddress()
{
	uint32_t* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, frameNumber));
	return target;
}

uint32_t* GetDeviceBounceAddress()
{
	uint32_t* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, bounce));
	return target;
}

NXB::BVH* GetDeviceTLASAddress()
{
	NXB::BVH* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, tlas));
	return target;
}

D_Mesh** GetDeviceMeshesAdress()
{
	D_Mesh** target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, meshes));
	return target;
}

D_PathStateSOA* GetDevicePathStateAddress()
{
	D_PathStateSOA* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, pathState));
	return target;
}

D_ShadowTraceRequestSOA* GetDeviceShadowTraceRequestAddress()
{
	D_ShadowTraceRequestSOA* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, shadowTraceRequest));
	return target;
}

D_TraceRequestSOA* GetDeviceTraceRequestAddress()
{
	D_TraceRequestSOA* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, traceRequest));
	return target;
}

D_MaterialRequestSOA* GetDeviceMaterialRequestAddress()
{
	D_MaterialRequestSOA* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, materialRequest));
	return target;
}

D_QueueSize* GetDeviceQueueSizeAddress()
{
	D_QueueSize* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, queueSize));
	return target;
}

D_PixelQuery* GetDevicePixelQueryAddress()
{
	D_PixelQuery* target;
	CheckCudaErrors(cudaGetSymbolAddress((void**)&target, pixelQuery));
	return target;
}
