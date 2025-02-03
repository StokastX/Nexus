#include "BVHInstance.h"
#include "Utils/Utils.h"

void BVHInstance::SetTransform(Mat4& t)
{
	m_Transform = t;
	m_InvTransform = t.Inverted();
	float3 bMin = m_Bvh->nodes[0].p;
	float3 bMax = bMin + make_float3(
		exp2f(m_Bvh->nodes[0].e[0] - 127),
		exp2f(m_Bvh->nodes[0].e[1] - 127),
		exp2f(m_Bvh->nodes[0].e[2] - 127)
	) * (exp2f(8) - 1);
	

	m_Bounds = AABB();
	for (int i = 0; i < 8; i++)
	{
		m_Bounds.Grow(TransformPosition(make_float3(i & 1 ? bMax.x : bMin.x,
			i & 2 ? bMax.y : bMin.y, i & 4 ? bMax.z : bMin.z), t));
	}
}

void BVHInstance::SetTransform(float3 pos, float3 r, float3 s)
{
	Mat4 t = Mat4::Translate(pos) * Mat4::RotateZ(Utils::ToRadians(r.z))
		* Mat4::RotateY(Utils::ToRadians(r.y)) * Mat4::RotateX(Utils::ToRadians(r.x)) * Mat4::Scale(s);
	SetTransform(t);
}

void BVHInstance::AssignMaterial(int mId)
{
	m_MaterialId = mId;
}

D_BVHInstance BVHInstance::ToDevice(const BVHInstance& bvhInstance)
{
	D_BVHInstance deviceInstance;
	deviceInstance.bounds = { bvhInstance.m_Bounds.bMin, bvhInstance.m_Bounds.bMax };
	deviceInstance.bvhIdx = bvhInstance.m_BvhIdx;
	deviceInstance.transform = bvhInstance.m_Transform;
	deviceInstance.invTransform = bvhInstance.m_InvTransform;
	deviceInstance.materialIdx = bvhInstance.m_MaterialId;
	return deviceInstance;
}
