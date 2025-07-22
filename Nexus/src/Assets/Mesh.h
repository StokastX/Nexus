#pragma once

#include <vector>
#include "NXB/BVHBuilder.h"
#include "Math/Mat4.h"
#include "Cuda/Scene/Mesh.cuh"
#include "Geometry/Triangle.h"
#include "Device/CudaMemory.h"
#include "Device/DeviceVector.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, const std::vector<NXB::Triangle>& t, const std::vector<TriangleData>& td,
		 uint32_t mId = INVALID_IDX, float3 p = make_float3(0.0f),
		 float3 r = make_float3(0.0f), float3 s = make_float3(1.0f))
		: name(n), triangles(t), triangleData(td), materialIdx(mId), position(p), rotation(r), scale(s)
	{
		deviceTriangles = triangles;
		deviceTriangleData = triangleData;

		NXB::BuildConfig buildConfig;
		buildConfig.prioritizeSpeed = true;

//		NXB::BVHBuildMetrics buildMetrics = NXB::BenchmarkBuild(
//#ifdef USE_BVH8
//			NXB::BuildBVH8<NXB::Triangle>,
//#else
//			NXB::BuildBVH2<NXB::Triangle>,
//#endif
//			20, 20, deviceTriangles.Data(), deviceTriangles.Size(), buildConfig);

		std::cout << std::endl << "========== Building BVH for mesh " << name << " ==========" << std::endl << std::endl;

#ifdef USE_BVH8
		bvh = NXB::BuildBVH8<NXB::Triangle>(deviceTriangles.Data(), deviceTriangles.Size(), buildConfig);
#else
		bvh = NXB::BuildBVH2<NXB::Triangle>(deviceTriangles.Data(), deviceTriangles.Size(), buildConfig);
#endif

		std::cout << "Triangle count: " << bvh.primCount << std::endl;
		std::cout << "Node count: " << bvh.nodeCount << std::endl;

		std::cout << std::endl << "========== Building done ==========" << std::endl << std::endl;
	}

	Mesh(const Mesh& other) = default;

	Mesh(Mesh &&other) noexcept
		: name(other.name),
		  position(other.position),
		  rotation(other.rotation),
		  scale(other.scale),
		  materialIdx(other.materialIdx),
		  triangles(std::move(other.triangles)),
		  triangleData(std::move(other.triangleData)),
		  deviceTriangles(std::move(other.deviceTriangles)),
		  deviceTriangleData(std::move(other.deviceTriangleData))
	{
		bvh.bounds = other.bvh.bounds;
		bvh.nodeCount = other.bvh.nodeCount;
		bvh.nodes = other.bvh.nodes;
		bvh.primCount = other.bvh.primCount;
		other.bvh.nodes = nullptr;
#ifdef USE_BVH8
		bvh.primIdx = other.bvh.primIdx;
		other.bvh.primIdx = nullptr;
#endif
	}

	~Mesh()
	{
		NXB::FreeDeviceBVH(bvh);
	}

	static D_Mesh ToDevice(const Mesh& mesh)
	{
		D_Mesh deviceMesh;
		deviceMesh.triangles = mesh.deviceTriangles.Data();
		deviceMesh.triangleData = mesh.deviceTriangleData.Data();
		deviceMesh.bvh = mesh.bvh;
		return deviceMesh;
	}

	std::string name;

	// Transform component of the mesh at loading
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);

	uint32_t materialIdx = INVALID_IDX;

	// All pointers stored in bvh are device pointers
	NXB::BVH bvh;

	std::vector<NXB::Triangle> triangles;
	std::vector<TriangleData> triangleData;

	DeviceVector<NXB::Triangle> deviceTriangles;
	DeviceVector<TriangleData, D_TriangleData> deviceTriangleData;
};
