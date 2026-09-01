#pragma once

#include <vector>
#include <type_traits>
#include "NXB/BVHBuilder.h"
#include "Math/Mat4.h"
#include "Cuda/Scene/Mesh.cuh"
#include "Geometry/Triangle.h"
#include "Device/CudaMemory.h"
#include "Device/DeviceVector.h"
#include "Device/DeviceTraits.h"
#include "Platform/OpenGL/GLVertexArray.h"

namespace Nexus {

	struct Mesh
	{
		Mesh() = default;
		// The arrays are taken by value and moved into place: a Mesh holds them for as long as it
		// lives, so a caller that has no further use for them should hand them over rather than
		// have them copied. Cheap either way for a caller that does still need its own.
		Mesh(std::string n, std::vector<NXB::Triangle> t, std::vector<TriangleData> td,
			uint32_t mId = INVALID_IDX, float3 p = make_float3(0.0f),
			float3 r = make_float3(0.0f), float3 s = make_float3(1.0f))
			: name(std::move(n)), triangles(std::move(t)), triangleData(std::move(td)), materialIdx(mId), position(p), rotation(r), scale(s)
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

			std::cout << "Triangle count: " << bvh.PrimCount() << std::endl;
			std::cout << "Node count: " << bvh.NodeCount() << std::endl;

			std::cout << std::endl << "========== Building done ==========" << std::endl << std::endl;

			// TODO: implement per-vertex data
			std::vector<float3> normals(triangleData.size() * 3);
			for (uint32_t i = 0; i < triangleData.size(); i++)
			{
				normals[i * 3] = triangleData[i].normal0;
				normals[i * 3 + 1] = triangleData[i].normal1;
				normals[i * 3 + 2] = triangleData[i].normal2;
			}

			// OpenGL buffers initialization. The triangle array is uploaded as a flat stream of
			// vertices rather than of triangles, which only works because NXB::Triangle is exactly
			// three float3 with no padding.
			static_assert(sizeof(NXB::Triangle) == 3 * sizeof(float3),
				"Triangles are uploaded as a flat vertex stream, which assumes no padding");

			// Attribute locations are handed out in the order buffers are added, so positions land
			// on location 0 and normals on location 1, matching layout.vert.
			GLVertexBuffer positionBuffer(triangles.data(), static_cast<uint32_t>(triangles.size() * sizeof(NXB::Triangle)));
			positionBuffer.SetLayout({ { ShaderDataType::Float3, "aPos" } });
			vertexArray.AddVertexBuffer(std::move(positionBuffer));

			GLVertexBuffer normalBuffer(normals.data(), static_cast<uint32_t>(normals.size() * sizeof(float3)));
			normalBuffer.SetLayout({ { ShaderDataType::Float3, "aNormal" } });
			vertexArray.AddVertexBuffer(std::move(normalBuffer));
		}

		/*
		 * No destructor and no copy or move members: every member owns its own storage, so the
		 * compiler-generated ones are correct. The copy constructor is implicitly deleted because
		 * GLVertexArray is move-only, which is the point.
		 */

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
		DeviceVector<TriangleData> deviceTriangleData;

		// OpenGL buffers: positions on attribute location 0, normals on location 1.
		GLVertexArray vertexArray;
	};

	// Mesh owns GL object names, so copying one used to duplicate them and hand two owners the
	// same buffers. Asserted rather than merely commented, so re-introducing a copy path fails
	// to compile instead of corrupting geometry at run time.
	static_assert(!std::is_copy_constructible_v<Mesh>, "Mesh must stay move-only");
	static_assert(!std::is_copy_assignable_v<Mesh>, "Mesh must stay move-only");
	static_assert(std::is_move_constructible_v<Mesh>, "Mesh is stored in a std::vector and must be movable");


	// Gathers pointers into memory the Mesh already owns -- nothing is allocated here, so there is
	// no matching teardown.
	template<>
	struct DeviceTraits<Mesh>
	{
		using DeviceType = D_Mesh;

		static D_Mesh ToDevice(const Mesh& mesh)
		{
			D_Mesh deviceMesh{};
			deviceMesh.triangles = mesh.deviceTriangles.Data();
			deviceMesh.triangleData = mesh.deviceTriangleData.Data();
			deviceMesh.bvh = mesh.bvh.View();
			return deviceMesh;
		}
	};

}
