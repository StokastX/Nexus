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
		bounds.Clear();
	}

	Mesh(const Mesh& other) = default;
	Mesh(Mesh&& other) = default;

	void BuildBVH()
	{
		NXB::BuildConfig buildConfig;
		buildConfig.prioritizeSpeed = false;

		// Benchmarking BVH build
		NXB::BVHBuildMetrics buildMetrics = NXB::BenchmarkBuild(
			NXB::BuildBVH2<NXB::Triangle>,
			0, 1, (NXB::Triangle*)deviceTriangles.Data(), deviceTriangles.Size(), buildConfig);

		std::cout << std::endl << "========== Building BVH for mesh " << name << " ==========" << std::endl << std::endl;

		bvh = NXB::BuildBVH2((NXB::Triangle*)deviceTriangles.Data(), deviceTriangles.Size(), buildConfig);

		std::cout << "Triangle count: " << bvh.primCount << std::endl;
		std::cout << "Node count: " << bvh.nodeCount << std::endl;

		std::cout << std::endl << "========== Building done ==========" << std::endl << std::endl;
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
	NXB::AABB bounds;

	uint32_t materialIdx = INVALID_IDX;

	// All pointers stored in bvh are device pointers
	NXB::BVH bvh;

	std::vector<NXB::Triangle> triangles;
	std::vector<TriangleData> triangleData;

	DeviceVector<NXB::Triangle> deviceTriangles;
	DeviceVector<TriangleData, D_TriangleData> deviceTriangleData;
};
