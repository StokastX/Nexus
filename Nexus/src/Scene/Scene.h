#pragma once

#include <iostream>
#include <optional>
#include "Device/DeviceVector.h"
#include "Device/DeviceSymbol.h"
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
#include "Device/MirroredVector.h"

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
		AssetManager& GetAssetManager() { return m_AssetManager; }
		const RenderSettings& GetRenderSettings() const { return m_RenderSettings; }
		RenderSettings& GetRenderSettings() { return m_RenderSettings; }

		bool IsEmpty() { return m_MeshInstances.Empty(); }
		/*
		 * The two signals a frame depends on.
		 *
		 * NeedsUpload is a transfer question: some mirrored array holds elements the device copy
		 * has not seen. Update answers it.
		 *
		 * NeedsAccumulationReset is an image question: the picture being accumulated no longer
		 * matches the scene and has to start over. Every upload implies it, but not the reverse --
		 * moving the camera or changing a render setting implies it with nothing to upload at all,
		 * because both reach the kernels inside D_Scene, which ToDevice rebuilds every frame.
		 */
		bool NeedsUpload() const { return m_MeshInstances.Dirty() || m_Lights.Dirty() || m_AssetManager.NeedsUpload(); }
		bool NeedsAccumulationReset() const { return m_AccumulationInvalid || m_Camera->Changed() || NeedsUpload(); }

		// For a change that alters the image without changing anything stored on the device
		// separately -- a render setting, the resolution.
		void InvalidateAccumulation() { m_AccumulationInvalid = true; }

		void Update();
		void BuildTLAS();
		void CreateMeshInstance(uint32_t meshId, uint32_t materialId, float3 position, float3 direction, float3 scale);
		MirroredVector<MeshInstance>& GetMeshInstances() { return m_MeshInstances; }
		void CreateMeshInstanceFromFile(const std::string& filePath, const std::string& fileName);
		void AddHDRMap(const std::string& filePath, const std::string& fileName);

		size_t AddLight(const Light& light);
		MirroredVector<Light>& GetLights() { return m_Lights; }
		void RemoveLight(const size_t index);

		friend struct DeviceTraits<Scene>;

	private:
		// Update the list of lights based on the changed material given by index
		void UpdateSceneLighting(size_t index);

	private:
		std::shared_ptr<Camera> m_Camera;

		MirroredVector<MeshInstance> m_MeshInstances;
		MirroredVector<Light> m_Lights;

		std::optional<Texture> m_HdrMap;
		NXB::BVH m_Tlas;

		AssetManager m_AssetManager;

		RenderSettings m_RenderSettings;

		// True at construction: nothing has been accumulated yet.
		bool m_AccumulationInvalid = true;

		// Device members
		DeviceSymbol<NXB::D_BVH> m_DeviceTlas;
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
