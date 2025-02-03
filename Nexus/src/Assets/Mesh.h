#pragma once

#include <vector>
#include "BVHBuilder.h"
#include "Math/Mat4.h"
#include "Cuda/Scene/Mesh.cuh"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, const std::vector<NXB::Triangle>& t, const std::vector<TriangleData>& td,
		uint32_t bId = INVALID_IDX, uint32_t mId = INVALID_IDX, float3 p = make_float3(0.0f),
		float3 r = make_float3(0.0f), float3 s = make_float3(1.0f))
		: name(n), triangles(t), triangleData(td), materialIdx(mId), position(p), rotation(r), scale(s), bvhIdx(bId)
	{
		deviceTriangles = triangles;
		deviceTriangleData = triangleData;
	}

	Mesh(const Mesh& other) = default;
	Mesh(Mesh&& other) = default;

	static D_Mesh ToDevice(const Mesh& mesh)
	{
		D_Mesh deviceMesh;
		deviceMesh.triangles = mesh.deviceTriangles.Data();
		deviceMesh.triangleData = mesh.deviceTriangleData.Data();
		deviceMesh.bvhIdx = mesh.bvhIdx;
	}

	std::string name;

	// Transform component of the mesh at loading
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);

	uint32_t materialIdx;
	uint32_t bvhIdx;

	std::vector<NXB::Triangle> triangles;
	std::vector<TriangleData> triangleData;

	DeviceVector<NXB::Triangle, D_Triangle> deviceTriangles;
	DeviceVector<TriangleData, D_TriangleData> deviceTriangleData;
};
