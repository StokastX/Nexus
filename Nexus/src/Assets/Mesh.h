#pragma once

#include <vector>
#include "NXB/BVHBuilder.h"
#include "Math/Mat4.h"
#include "Cuda/Scene/Mesh.cuh"
#include "Geometry/Triangle.h"
#include "Device/CudaMemory.h"
#include "Device/DeviceVector.h"
#include <GL/glew.h>

namespace Nexus {

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

			// TODO: implement per-vertex data
			std::vector<float3> normals(triangleData.size() * 3);
			for (uint32_t i = 0; i < triangleData.size(); i++)
			{
				normals[i * 3] = triangleData[i].normal0;
				normals[i * 3 + 1] = triangleData[i].normal1;
				normals[i * 3 + 2] = triangleData[i].normal2;
			}

			// OpenGL buffers initialization
			glGenVertexArrays(1, &vao);
			glGenBuffers(1, &vbo);
			glGenBuffers(1, &vboNormals);
			glBindVertexArray(vao);

			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, triangles.size() * sizeof(NXB::Triangle), triangles.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, vboNormals);
			glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float3), normals.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
			glEnableVertexAttribArray(1);

			glBindVertexArray(0);
		}

		Mesh(const Mesh& other) = default;

		Mesh(Mesh&& other) noexcept
			: name(other.name),
			position(other.position),
			rotation(other.rotation),
			scale(other.scale),
			materialIdx(other.materialIdx),
			triangles(std::move(other.triangles)),
			triangleData(std::move(other.triangleData)),
			deviceTriangles(std::move(other.deviceTriangles)),
			deviceTriangleData(std::move(other.deviceTriangleData)),
			vao(other.vao),
			vbo(other.vbo),
			vboNormals(other.vboNormals)
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
			other.vao = 0;
			other.vbo = 0;
			other.vboNormals = 0;
		}

		~Mesh()
		{
			NXB::FreeDeviceBVH(bvh);
			// Free OpenGL buffers
			glDeleteVertexArrays(1, &vao);
			glDeleteBuffers(1, &vbo);
			glDeleteBuffers(1, &vboNormals);
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

		// OpenGL buffers
		uint32_t vbo = 0;
		uint32_t vboNormals = 0;
		uint32_t vao = 0;
	};

}
