#pragma once
#include <vector>
#include "Assets/Mesh.h"
#include "Cuda/Scene/MeshInstance.cuh"
#include "NXB/BVHBuilder.h"
#include "Geometry/BVH/BVH.h"

struct MeshInstance
{
	MeshInstance() = default;
	MeshInstance(const Mesh& mesh, uint32_t mIdx, uint32_t matIdx = INVALID_IDX)
	{
		name = mesh.name;
		rotation = mesh.rotation;
		scale = mesh.scale;
		position = mesh.position;
		meshBounds = mesh.bvh.bounds;
		meshIdx = mIdx;
		materialIdx = matIdx;
	}

	void SetPosition(float3 p) { position = p; }
	void SetRotationX(float r) { rotation.x = r; }
	void SetRotationY(float r) { position.y = r; }
	void SetRotationZ(float r) { position.z = r; }
	void SetScale(float s) { scale = make_float3(s); }
	void SetScale(float3 s) { scale = s; }
	void SetTransform(float3 p, float3 r, float3 s)
	{
		position = p;
		rotation = r;
		scale = s;
	}
	void AssignMaterial(int mIdx) { materialIdx = mIdx; }

	Mat4 GetTransfromationMatrix() const
	{
		return Mat4::Translate(position) * Mat4::RotateZ(Utils::ToRadians(rotation.z))
			* Mat4::RotateY(Utils::ToRadians(rotation.y)) * Mat4::RotateX(Utils::ToRadians(rotation.x)) * Mat4::Scale(scale);
	}

	NXB::AABB GetBounds() const
	{
		Mat4 transformationMatrix = GetTransfromationMatrix();
		NXB::AABB bounds;
		bounds.Clear();
		for (int i = 0; i < 8; i++)
		{
			bounds.Grow(transformationMatrix.TransformPoint(make_float3(i & 1 ? meshBounds.bMax.x : meshBounds.bMin.x,
				i & 2 ? meshBounds.bMax.y : meshBounds.bMin.y, i & 4 ? meshBounds.bMax.z : meshBounds.bMin.z)));
		}
		return bounds;
	}

	static D_MeshInstance ToDevice(const MeshInstance& meshInstance)
	{
		Mat4 transformationMatrix = meshInstance.GetTransfromationMatrix();
		D_MeshInstance deviceInstance;
		deviceInstance.meshIdx = meshInstance.meshIdx;
		deviceInstance.materialIdx = meshInstance.materialIdx;
		deviceInstance.transform = transformationMatrix;
		deviceInstance.invTransform = transformationMatrix.Inverted();
		deviceInstance.bounds = *(D_AABB*)&meshInstance.GetBounds();
		return deviceInstance;
	}

	std::string name;

	NXB::AABB meshBounds;
	uint32_t meshIdx = INVALID_IDX;
	uint32_t materialIdx = INVALID_IDX;

	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);
	float3 position = make_float3(0.0f);
};
