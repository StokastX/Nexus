#include "Scene.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Utils/cuda_math.h"
#include "Assets/IMGLoader.h"
#include "Assets/OBJLoader.h"


Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f,
		width, height, 5.0f, 0.0f)), m_DeviceTlas(GetDeviceTLASAddress()) { }

Scene::~Scene()
{
	NXB::FreeDeviceBVH(m_DeviceTlas.Instance());
}

void Scene::Reset()
{
	m_Invalid = true;
	m_InvalidMeshInstances.clear();
	m_MeshInstances.clear();
	m_AssetManager.Reset();
	m_Camera->Invalidate();
	NXB::FreeDeviceBVH(m_DeviceTlas.Instance());
}

void Scene::AddMaterial(Material& material)
{
	m_AssetManager.AddMaterial(material);
}

void Scene::Update()
{
	m_Camera->SetInvalid(false);

	if (m_InvalidMeshInstances.size() != 0)
	{
		for (uint32_t i : m_InvalidMeshInstances)
		{
			MeshInstance& meshInstance = m_MeshInstances[i];
			m_DeviceMeshInstances[i] = meshInstance;
		}

		BuildTLAS();

		m_InvalidMeshInstances.clear();
	}

	for (uint32_t i : m_AssetManager.GetInvalidMaterials())
		UpdateSceneLighting(i);

	m_AssetManager.SendDataToDevice();

	// Only punctual lights
	for (uint32_t i : m_InvalidLights)
		m_DeviceLights[i] = m_Lights[i];

	m_InvalidLights.clear();

	m_Invalid = false;
}

void Scene::BuildTLAS()
{
	std::vector<NXB::AABB> instancesBounds(m_MeshInstances.size());
	for (uint32_t i = 0; i < m_MeshInstances.size(); i++)
		instancesBounds[i] = m_MeshInstances[i].GetBounds();

	DeviceVector<NXB::AABB> deviceBounds = instancesBounds;

	#ifdef USE_BVH8
		m_DeviceTlas = NXB::BuildBVH8<NXB::AABB>(deviceBounds.Data(), instancesBounds.size());
	#else
		m_DeviceTlas = NXB::BuildBVH2<NXB::AABB>(deviceBounds.Data(), instancesBounds.size());
	#endif
}

MeshInstance& Scene::CreateMeshInstance(uint32_t meshId)
{
	Mesh& mesh = m_AssetManager.GetMeshes()[meshId];

	MeshInstance meshInstance(mesh, meshId, mesh.materialIdx);
	m_MeshInstances.push_back(meshInstance);
	m_DeviceMeshInstances.PushBack(meshInstance);

	const size_t instanceId = m_MeshInstances.size() - 1;

	// Create light if needed
	//UpdateInstanceLighting(instanceId);
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
	m_HdrMap.sRGB = false;
	m_DeviceHdrMap = Texture::ToDevice(m_HdrMap);
}

void Scene::InvalidateMeshInstance(uint32_t instanceId)
{
	m_InvalidMeshInstances.insert(instanceId);
}

void Scene::InvalidateLight(uint32_t lightIdx)
{
	m_InvalidLights.insert(lightIdx);
}

size_t Scene::AddLight(const Light& light)
{
	m_Lights.push_back(light);
	m_DeviceLights.PushBack(light);
	uint32_t lightIdx = m_Lights.size() - 1;
	InvalidateLight(lightIdx);
	std::cout << "added light of type " << (int)light.type << std::endl;
	return lightIdx;
}

void Scene::RemoveLight(const size_t index)
{
	m_Lights.erase(m_Lights.begin() + index);
}

D_Scene Scene::ToDevice(const Scene& scene)
{
	D_Scene deviceScene;

	const DeviceVector<Texture, cudaTextureObject_t>& deviceTextures = scene.m_AssetManager.GetDeviceTextures();
	const DeviceVector<Material, D_Material>& deviceMaterials = scene.m_AssetManager.GetDeviceMaterials();

	deviceScene.textures = deviceTextures.Data();
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

void Scene::UpdateSceneLighting(size_t index)
{
	bool lightingChanged = false;
	Material& material = m_AssetManager.GetMaterials()[index];
	// Remove lights that do not emit anymore
	if ((material.emissiveMapId == -1 && fmaxf(material.emissionColor) == 0.0f)
		|| material.intensity == 0.0f)
	{
	int counter = 0;
		for (uint32_t j = 0; j < m_Lights.size(); j++)
		{
			if (m_Lights[j].type == Light::Type::MESH)
			{
				MeshInstance instance = m_MeshInstances[m_Lights[j].mesh.meshId];
				if (instance.materialIdx == index)
				{
					m_Lights.erase(m_Lights.begin() + j);
					lightingChanged = true;
					counter++;
				}
			}
		}
	if (counter > 0)
		std::cout << "Removed " << counter << " lights" << std::endl;
	}

	// Add potentially new lights
	else if ((material.emissiveMapId != -1 || fmaxf(material.emissionColor) > 0.0f)
		&& material.intensity > 0.0f)
	{
	int counter = 0;
		for (uint32_t j = 0; j < m_MeshInstances.size(); j++)
		{
			bool addLight = true;
			MeshInstance instance = m_MeshInstances[j];
			if (instance.materialIdx == index)
			{
				for (uint32_t k = 0; k < m_Lights.size(); k++)
				{
					if (m_Lights[k].type == Light::Type::MESH && m_Lights[k].mesh.meshId == j)
					{
						addLight = false;
						break;
					}
				}
				if (addLight)
				{
					Light meshLight;
					meshLight.type = Light::Type::MESH;
					meshLight.mesh.meshId = j;
					m_Lights.push_back(meshLight);
					lightingChanged = true;
					counter++;
				}
			}
		}
	if (counter > 0)
		std::cout << "Added " << counter << " lights" << std::endl;
	}

	if (lightingChanged)
		m_DeviceLights = m_Lights;
}
