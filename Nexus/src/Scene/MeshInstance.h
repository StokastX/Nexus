#pragma once
#include <vector>
#include "Geometry/BVH/BVHInstance.h"
#include "Assets/Mesh.h"

struct MeshInstance
{
	MeshInstance() = default;
	MeshInstance(const Mesh& mesh, int bvhInstIdx, int mId = -1)
	{
		name = mesh.name;
		rotation = mesh.rotation;
		scale = mesh.scale;
		position = mesh.position;
		bvhInstanceIdx = bvhInstIdx;
		materialId = mId;
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
	void AssignMaterial(int mId) { materialId = mId; }

	std::string name;

	int bvhInstanceIdx;
	int materialId = -1;

	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);
	float3 position = make_float3(0.0f);
};
