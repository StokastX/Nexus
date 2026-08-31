#pragma once

#include <iostream>
#include "Device/DeviceVector.h"
#include "Device/DeviceTraits.h"

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Light.h"
#include "Renderer/RenderSettings.h"
#include "Assets/AssetManager.h"
#include "Scene/MeshInstance.h"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Scene/Scene.cuh"
#include "Cuda/Scene/Light.cuh"

namespace Nexus {

	class Scene;

	// Declared up front so the class below can befriend it: the conversion reads private
	// state, which a non-intrusive trait otherwise cannot reach.
	template<>
	struct DeviceTraits<Scene>;

	class Scene
	{
	public:
		Scene(uint2 resolution = make_uint2(1));
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
		std::vector<Light>& GetLights() { return m_Lights; }
		void RemoveLight(const size_t index);

		friend struct DeviceTraits<Scene>;

	private:
		// Update the list of lights based on the changed material given by index
		void UpdateSceneLighting(size_t index);

	private:
		std::shared_ptr<Camera> m_Camera;

		std::vector<MeshInstance> m_MeshInstances;
		std::vector<Light> m_Lights;

		std::set<uint32_t> m_InvalidMeshInstances;
		std::set<uint32_t> m_InvalidLights;

		std::shared_ptr<Texture> m_HdrMap;
		NXB::BVH m_Tlas;

		AssetManager m_AssetManager;

		RenderSettings m_RenderSettings;

		bool m_Invalid = true;

		// Device members
		DeviceVector<MeshInstance> m_DeviceMeshInstances;
		DeviceVector<Light> m_DeviceLights;
		DeviceInstance<NXB::D_BVH> m_DeviceTlas;
	};


	// Gathers the device pointers the Scene and its AssetManager already own into a flat POD the
	// kernels read. Allocates nothing.
	template<>
	struct DeviceTraits<Scene>
	{
		using DeviceType = D_Scene;

		static D_Scene ToDevice(const Scene& scene);
	};

}
