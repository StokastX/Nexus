#pragma once

#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include "Device/DeviceVector.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"
#include "Geometry/BVH/BVH.h"
#include "Geometry/Triangle.h"

struct aiTexture;

namespace Nexus {

	class AssetManager
	{
	public:
		AssetManager();

		void Reset();

		uint32_t AddMesh(const std::string& name, uint32_t materialIdx, const std::vector<NXB::Triangle>& triangles, const std::vector<TriangleData>& triangleData);

		void AddMaterial();
		uint32_t AddMaterial(const Material& material);
		std::vector<Material>& GetMaterials() { return m_Materials; }
		std::set<uint32_t>& GetInvalidMaterials() { return m_InvalidMaterials; }
		void InvalidateMaterial(uint32_t index);
		std::string GetMaterialsString();
		std::vector<Mesh>& GetMeshes() { return m_Meshes; }

		DeviceVector<Material>& GetDeviceMaterials() { return m_DeviceMaterials; }
		DeviceVector<cudaTextureObject_t>& GetDeviceTextureHandles() { return m_DeviceTextureHandles; }
		DeviceVector<Mesh>& GetDeviceMeshes() { return m_DeviceMeshes; }

		const DeviceVector<Material>& GetDeviceMaterials() const { return m_DeviceMaterials; }
		const DeviceVector<cudaTextureObject_t>& GetDeviceTextureHandles() const { return m_DeviceTextureHandles; }

		/*
		 * Loads, uploads and registers a texture, returning its index -- or -1 if it could not be
		 * loaded, which is already what Material spells as "no map".
		 *
		 * The manager loads rather than being handed a built Texture, so that construction, device
		 * upload and registration cannot be done half-way, and so that repeated paths can be shared.
		 */
		int AddTexture(const std::string& filePath, Texture::Type type);
		int AddTexture(const aiTexture* embedded, Texture::Type type);
		void ApplyTextureToMaterial(int materialIdx, int diffuseMapId);

		bool SendDataToDevice();

		bool IsInvalid() { return m_InvalidMaterials.size() > 0; }

	private:
		// Uploads and registers an already-loaded texture; null means the load failed.
		int StoreTexture(std::shared_ptr<Texture> texture);

	private:
		std::vector<Material> m_Materials;
		std::set<uint32_t> m_InvalidMaterials;
		std::vector<std::shared_ptr<Texture>> m_Textures;

		// Index of an already-loaded texture, keyed by path and type. Without it a model whose
		// materials share a base colour map loads and uploads that file once per material.
		std::unordered_map<std::string, int> m_TextureIds;
		std::vector<Mesh> m_Meshes;

		// Device members
		DeviceVector<Material> m_DeviceMaterials;
		// The flat handle array the kernel indexes. Each texture's memory is owned by the
		// CudaTexture inside the Texture in m_Textures above; these are only its handles.
		DeviceVector<cudaTextureObject_t> m_DeviceTextureHandles;
		DeviceVector<Mesh> m_DeviceMeshes;
		DeviceInstance<D_Mesh*> m_DeviceMeshesAdress;
	};

}
