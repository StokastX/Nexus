#pragma once

#include <iostream>
#include "Device/DeviceVector.h"

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Light.h"
#include "Renderer/RenderSettings.h"
#include "Assets/AssetManager.h"
#include "Scene/MeshInstance.h"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Scene/Scene.cuh"
#include "Cuda/Scene/Light.cuh"

class Scene
{
public:
	Scene(uint2 resolution);
	~Scene();
	void Reset();

	std::shared_ptr<Camera> GetCamera() { return m_Camera; }

	void AddMaterial(Material& material);
	std::vector<Material>& GetMaterials() { return m_AssetManager.GetMaterials(); }
	AssetManager& GetAssetManager() { return m_AssetManager; }
	const RenderSettings& GetRenderSettings() const { return m_RenderSettings; }
	RenderSettings& GetRenderSettings() { return m_RenderSettings; }

	bool IsEmpty() { return m_MeshInstances.size() == 0; }
	void Invalidate() { m_Invalid = true; }
	bool IsInvalid() { return m_Invalid || m_InvalidMeshInstances.size() > 0 || m_InvalidLights.size() > 0 || m_Camera->IsInvalid() || m_AssetManager.IsInvalid(); }

	void Update();
	void BuildTLAS();
	MeshInstance& CreateMeshInstance(uint32_t meshId);
	std::vector<MeshInstance>& GetMeshInstances() { return m_MeshInstances; }
	void CreateMeshInstanceFromFile(const std::string& filePath, const std::string& fileName);
	void AddHDRMap(const std::string& filePath, const std::string& fileName);
	void InvalidateMeshInstance(uint32_t instanceId);

	size_t AddLight(const Light& light);
	void InvalidateLight(uint32_t lightIdx);
	std::vector<Light> &GetLights() { return m_Lights; }
	void RemoveLight(const size_t index);

	// Create or update the device scene and returns a D_Scene object
	static D_Scene ToDevice(const Scene& scene);

private:
	// Update the list of lights based on the changed material given by index
	void UpdateSceneLighting(size_t index);

private:
	std::shared_ptr<Camera> m_Camera;

	std::vector<MeshInstance> m_MeshInstances;
	std::vector<Light> m_Lights;

	std::set<uint32_t> m_InvalidMeshInstances;
	std::set<uint32_t> m_InvalidLights;

	Texture m_HdrMap;

	AssetManager m_AssetManager;

	RenderSettings m_RenderSettings;

	bool m_Invalid = true;

	// Device members
	cudaTextureObject_t m_DeviceHdrMap;
	DeviceVector<MeshInstance, D_MeshInstance> m_DeviceMeshInstances;
	DeviceVector<Light, D_Light> m_DeviceLights;
	DeviceInstance<NXB::BVH> m_DeviceTlas;
};
