#pragma once

#include <iostream>
#include <set>
#include <string>
#include <optional>
#include <unordered_map>
#include "Device/DeviceVector.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"
#include "Geometry/BVH/BVH.h"
#include "Geometry/Triangle.h"
#include "Device/MirroredVector.h"

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
		MirroredVector<Material>& GetMaterials() { return m_Materials; }
		const MirroredVector<Material>& GetMaterials() const { return m_Materials; }
		std::string GetMaterialsString();
		MirroredVector<Mesh>& GetMeshes() { return m_Meshes; }
		const MirroredVector<Mesh>& GetMeshes() const { return m_Meshes; }

		MirroredVector<Texture>& GetTextures() { return m_Textures; }
		const MirroredVector<Texture>& GetTextures() const { return m_Textures; }

		/*
		 * Loads, uploads and registers a texture, returning its index -- or -1 if it could not be
		 * loaded, which is already what Material spells as "no map".
		 *
		 * The manager loads rather than being handed a built Texture, so that construction, device
		 * upload and registration cannot be done half-way, and so that repeated paths can be shared.
		 */
		int AddTexture(const std::string& filePath, Texture::Type type);
		int AddTexture(const aiTexture* embedded, Texture::Type type);

		void UploadToDevice();

		bool IsInvalid() { return m_Materials.Dirty(); }

	private:
		// Uploads and registers an already-loaded texture; null means the load failed.
		int StoreTexture(std::optional<Texture> texture);

	private:
		MirroredVector<Material> m_Materials;
		MirroredVector<Mesh> m_Meshes;
		// Each texture's pixels and sampler are owned by the CudaTexture inside it; the device
		// array holds only the handles the kernels index.
		MirroredVector<Texture> m_Textures;

		// Index of an already-loaded texture, keyed by path and type. Without it a model whose
		// materials share a base colour map loads and uploads that file once per material.
		std::unordered_map<std::string, int> m_TextureIds;

		// Device members
		DeviceInstance<D_Mesh*> m_DeviceMeshesAdress;
	};

}
