#include "Scene.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Utils/cuda_math.h"
#include "Assets/IMGLoader.h"
#include "Assets/SceneLoader.h"


namespace Nexus {

	Scene::Scene(uint2 resolution)
		: m_Camera(std::make_shared<Camera>(resolution)), m_DeviceTlas(GetDeviceTLASAddress())
	{
		m_RenderSettings.resolution = resolution;
	}

	Scene::~Scene()
	{
	}

	void Scene::Reset()
	{
		m_AccumulationInvalid = true;
		m_MeshInstances.Clear();
		m_Lights.Clear();
		m_AssetManager.Reset();
		m_Camera->MarkChanged();
		m_Tlas = NXB::BVH();
	}

	void Scene::AddMaterial(Material& material)
	{
		m_AssetManager.AddMaterial(material);
	}

	void Scene::Update()
	{
		// Mesh lights are derived from material emission, so they are re-derived from the materials
		// that changed -- before UploadToDevice below flushes that set and clears it. This pass
		// adds and removes lights, so it has to run before the lights are flushed too.
		for (uint32_t i : m_AssetManager.GetMaterials().DirtyIndices())
			UpdateSceneLighting(i);

		m_AssetManager.UploadToDevice();

		m_Lights.Flush();

		if (m_MeshInstances.Dirty())
		{
			m_MeshInstances.Flush();

			// Building a BVH over zero primitives is not worth asking NXB to handle; an empty
			// scene has nothing to trace against anyway.
			if (!m_MeshInstances.Empty())
				BuildTLAS();
		}

		// Both signals are cleared here, at the end, once the work they asked for is done.
		m_Camera->ClearChanged();
		m_AccumulationInvalid = false;
	}

	void Scene::BuildTLAS()
	{
		std::vector<NXB::AABB> instancesBounds(m_MeshInstances.Size());
		for (uint32_t i = 0; i < m_MeshInstances.Size(); i++)
			instancesBounds[i] = m_MeshInstances[i].GetBounds();

		DeviceVector<NXB::AABB> deviceBounds = instancesBounds;

#ifdef USE_BVH8
		m_Tlas = NXB::BuildBVH8<NXB::AABB>(deviceBounds.Data(), instancesBounds.size());
#else
		m_Tlas = NXB::BuildBVH2<NXB::AABB>(deviceBounds.Data(), instancesBounds.size());
#endif
		m_DeviceTlas = m_Tlas.View();
	}

	void Scene::CreateMeshInstance(uint32_t meshId, uint32_t materialId, float3 position, float3 direction, float3 scale)
	{
		const Mesh& mesh = m_AssetManager.GetMeshes()[meshId];

		MeshInstance meshInstance(mesh, meshId, materialId, position, direction, scale);
		m_MeshInstances.PushBack(meshInstance);
	}

	void Scene::CreateMeshInstanceFromFile(const std::string& path, const std::string& fileName)
	{
		SceneLoader::LoadScene(path, fileName, this, &m_AssetManager);
	}

	void Scene::AddHDRMap(const std::string& filePath, const std::string& fileName)
	{
		std::optional<Texture> hdrMap = IMGLoader::LoadIMG(filePath + fileName, Texture::Type::ENVIRONMENT);

		// Keep whatever map is already loaded if this one failed, rather than clearing it.
		if (!hdrMap)
			return;

		hdrMap->UploadToDevice();
		m_HdrMap = std::move(hdrMap);
	}

	size_t Scene::AddLight(const Light& light)
	{
		const size_t lightIdx = m_Lights.PushBack(light);
		std::cout << "added light of type " << (int)light.type << std::endl;
		return lightIdx;
	}

	void Scene::RemoveLight(const size_t index)
	{
		m_Lights.Erase(index);
	}

	D_Scene DeviceTraits<Scene>::ToDevice(const Scene& scene)
	{
		D_Scene deviceScene{};

		deviceScene.textures = scene.m_AssetManager.GetTextures().DeviceData();
		deviceScene.materials = scene.m_AssetManager.GetMaterials().DeviceData();
		deviceScene.meshInstances = scene.m_MeshInstances.DeviceData();
		deviceScene.lights = scene.m_Lights.DeviceData();
		deviceScene.lightCount = scene.m_Lights.Size();

		deviceScene.renderSettings = ConvertToDevice(scene.m_RenderSettings);

		deviceScene.hasHdrMap = scene.m_HdrMap.has_value();
		deviceScene.hdrMap = scene.m_HdrMap ? scene.m_HdrMap->deviceTexture.Handle() : 0;
		deviceScene.camera = ConvertToDevice(*scene.m_Camera);

		return deviceScene;
	}

	void Scene::UpdateSceneLighting(size_t index)
	{
		const Material& material = m_AssetManager.GetMaterials()[index];
		// Remove lights that do not emit anymore
		if ((material.emissiveMapId == -1 && fmaxf(material.emissionColor) == 0.0f)
			|| material.intensity == 0.0f)
		{
			int counter = 0;
			for (uint32_t j = 0; j < m_Lights.Size(); )
			{
				const Light& light = m_Lights[j];
				if (light.type == Light::Type::MESH
					&& m_MeshInstances[light.mesh.meshId].materialIdx == index)
				{
					// Not incremented here: erasing shifts the next light down into j, and stepping
					// over it would leave a light behind that should have gone.
					m_Lights.Erase(j);
					counter++;
				}
				else
					j++;
			}
			if (counter > 0)
				std::cout << "Removed " << counter << " lights" << std::endl;
		}

		// Add potentially new lights
		else if ((material.emissiveMapId != -1 || fmaxf(material.emissionColor) > 0.0f)
			&& material.intensity > 0.0f)
		{
			int counter = 0;
			for (uint32_t j = 0; j < m_MeshInstances.Size(); j++)
			{
				bool addLight = true;
				const MeshInstance& instance = m_MeshInstances[j];
				if (instance.materialIdx == index)
				{
					for (uint32_t k = 0; k < m_Lights.Size(); k++)
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
						m_Lights.PushBack(meshLight);
						counter++;
					}
				}
			}
			if (counter > 0)
				std::cout << "Added " << counter << " lights" << std::endl;
		}
	}

}
