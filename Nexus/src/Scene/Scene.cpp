#include "Scene.h"
#include "Cuda/PathTracer/Pathtracer.cuh"
#include "Utils/cuda_math.h"
#include "Assets/IMGLoader.h"
#include "Geometry/BVH/TLASBuilder.h"
#include "Assets/OBJLoader.h"


Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 60.0f,
		width, height, 5.0f, 0.0f)), m_DeviceTlas(GetDeviceTLASAddress()) { }

void Scene::Reset()
{
	m_Invalid = true;
	m_InvalidMeshInstances.clear();
	m_MeshInstances.clear();
	m_AssetManager.Reset();
	m_Camera->Invalidate();
}

void Scene::AddMaterial(Material& material)
{
	m_AssetManager.AddMaterial(material);
}

void Scene::Update()
{
	m_Camera->SetInvalid(false);

	m_AssetManager.SendDataToDevice();

	if (m_InvalidMeshInstances.size() != 0)
	{
		for (uint32_t i : m_InvalidMeshInstances)
		{
			MeshInstance& meshInstance = m_MeshInstances[i];
			m_DeviceMeshInstances[i] = meshInstance;

			if (meshInstance.materialIdx != -1)
				UpdateInstanceLighting(i);
		}

		BuildTLAS();

		m_InvalidMeshInstances.clear();
	}
	m_Invalid = false;
}

void Scene::BuildTLAS()
{
	std::vector<NXB::AABB> instancesBounds(m_MeshInstances.size());
	for (uint32_t i = 0; i < m_MeshInstances.size(); i++)
		instancesBounds[i] = m_MeshInstances[i].GetBounds();

	DeviceVector<NXB::AABB> deviceBounds = instancesBounds;

	m_DeviceTlas = NXB::BuildBVH2(deviceBounds.Data(), instancesBounds.size());
}

MeshInstance& Scene::CreateMeshInstance(uint32_t meshId)
{
	Mesh& mesh = m_AssetManager.GetMeshes()[meshId];

	MeshInstance meshInstance(mesh, meshId, mesh.materialIdx);
	m_MeshInstances.push_back(meshInstance);
	m_DeviceMeshInstances.PushBack(meshInstance);

	const size_t instanceId = m_MeshInstances.size() - 1;

	// Create light if needed
	UpdateInstanceLighting(instanceId);
	InvalidateMeshInstance(instanceId);

	return m_MeshInstances[instanceId];
}

void Scene::CreateMeshInstanceFromFile(const std::string& path, const std::string& fileName)
{
	OBJLoader::LoadOBJ(path, fileName, this, &m_AssetManager);
}

void Scene::AddHDRMap(const std::string& filePath, const std::string& fileName)
{
	m_HdrMap = IMGLoader::LoadIMG(filePath + fileName);
	m_DeviceHdrMap = Texture::ToDevice(m_HdrMap);
}

void Scene::InvalidateMeshInstance(uint32_t instanceId)
{
	m_InvalidMeshInstances.insert(instanceId);
}

size_t Scene::AddLight(const Light& light)
{
	m_Lights.push_back(light);
	return m_Lights.size() - 1;
}

void Scene::RemoveLight(const size_t index)
{
	m_Lights.erase(m_Lights.begin() + index);
}

D_Scene Scene::ToDevice(const Scene& scene)
{
	D_Scene deviceScene;

	const DeviceVector<Texture, cudaTextureObject_t>& deviceDiffuseMaps = scene.m_AssetManager.GetDeviceDiffuseMaps();
	const DeviceVector<Texture, cudaTextureObject_t>& deviceEmissiveMaps = scene.m_AssetManager.GetDeviceEmissiveMaps();
	const DeviceVector<Material, D_Material>& deviceMaterials = scene.m_AssetManager.GetDeviceMaterials();

	deviceScene.diffuseMaps = deviceDiffuseMaps.Data();
	deviceScene.emissiveMaps = deviceEmissiveMaps.Data();
	deviceScene.materials = deviceMaterials.Data();
	deviceScene.meshInstances = scene.m_DeviceMeshInstances.Data();
	deviceScene.lights = scene.m_DeviceLights.Data();
	deviceScene.lightCount = scene.m_DeviceLights.Size();

	deviceScene.renderSettings = *(D_RenderSettings*)&scene.m_RenderSettings;

	deviceScene.hasHdrMap = scene.m_HdrMap.pixels != nullptr;
	// TODO: clear m_DeviceHdrMap when reset
	deviceScene.hdrMap = scene.m_DeviceHdrMap;
	deviceScene.camera = Camera::ToDevice(*scene.m_Camera);

	return deviceScene;
}

void Scene::UpdateInstanceLighting(size_t index)
{
	const MeshInstance& meshInstance = m_MeshInstances[index];

	if (meshInstance.materialIdx == -1)
		return;

	const Material& material = m_AssetManager.GetMaterials()[meshInstance.materialIdx];

	// If light already in the scene, return or remove light
	for (size_t i = 0; i < m_Lights.size(); i++)
	{
		const Light& light = m_Lights[i];
		if (light.type == Light::Type::MESH_LIGHT && light.mesh.meshId == index)
		{
			if (fmaxf(material.intensity * material.emissive) == 0.0f)
			{
				m_Lights.erase(m_Lights.begin() + i);
				m_DeviceLights = m_Lights;
			}
			return;
		}
	}

	// If mesh has an emissive material, add it to the lights list
	if (material.emissiveMapId != -1 ||
		material.intensity * fmaxf(material.emissive) > 0.0f)
	{
		Light meshLight;
		meshLight.type = Light::Type::MESH_LIGHT;
		meshLight.mesh.meshId = index;
		m_Lights.push_back(meshLight);
		m_DeviceLights.PushBack(meshLight);
	}
}
