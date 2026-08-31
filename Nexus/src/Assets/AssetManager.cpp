#include "AssetManager.h"
#include "SceneLoader.h"
#include "IMGLoader.h"
#include "Cuda/PathTracer/PathTracer.cuh"

namespace Nexus {

	AssetManager::AssetManager() : m_DeviceMeshesAdress(GetDeviceMeshesAdress()) {}

	void AssetManager::Reset()
	{
		m_Materials.clear();
		m_InvalidMaterials.clear();
		m_Textures.clear();
		m_TextureIds.clear();
		m_DeviceTextureHandles.Clear();
		m_DeviceMaterials.Clear();
		m_Meshes.clear();
		m_DeviceMeshes.Clear();
	}

	uint32_t AssetManager::AddMesh(const std::string& name, uint32_t materialIdx, const std::vector<NXB::Triangle>& triangles, const std::vector<TriangleData>& triangleData)
	{
		m_Meshes.emplace_back(name, triangles, triangleData, materialIdx);

		// TODO: move this to a separate function
		m_DeviceMeshes = m_Meshes;
		m_DeviceMeshesAdress = m_DeviceMeshes.Data();

		return m_Meshes.size() - 1;
	}

	void AssetManager::AddMaterial()
	{
		Material material;
		AddMaterial(material);
	}

	uint32_t AssetManager::AddMaterial(const Material& material)
	{
		m_Materials.push_back(material);
		m_DeviceMaterials.PushBack(material);
		Material& m = m_Materials.back();
		uint32_t idx = m_Materials.size() - 1;

		// To update instances lighting
		m_InvalidMaterials.insert(idx);
		return idx;
	}

	void AssetManager::InvalidateMaterial(uint32_t index)
	{
		m_InvalidMaterials.insert(index);
	}

	int AssetManager::AddTexture(const std::string& filePath, Texture::Type type)
	{
		// Keyed on the type as well as the path: the same file can serve as a base colour map in one
		// material and a roughness map in another, and those want opposite sRGB handling. Sharing one
		// entry between them would give one of the two the wrong decode.
		const std::string key = filePath + '#' + std::to_string(static_cast<int>(type));

		auto cached = m_TextureIds.find(key);
		if (cached != m_TextureIds.end())
			return cached->second;

		const int id = StoreTexture(IMGLoader::LoadIMG(filePath, type));
		if (id != -1)
			m_TextureIds.emplace(key, id);

		return id;
	}

	int AssetManager::AddTexture(const aiTexture* embedded, Texture::Type type)
	{
		// Embedded textures carry no path to key the cache on.
		return StoreTexture(IMGLoader::LoadIMG(embedded, type));
	}

	int AssetManager::StoreTexture(std::shared_ptr<Texture> texture)
	{
		if (!texture)
			return -1;

		texture->UploadToDevice();
		m_Textures.push_back(texture);
		m_DeviceTextureHandles.PushBack(texture->deviceTexture.Handle());
		return static_cast<int>(m_Textures.size()) - 1;
	}

	bool AssetManager::SendDataToDevice()
	{
		bool invalid = false;
		for (uint32_t id : m_InvalidMaterials)
		{
			invalid = true;
			m_DeviceMaterials[id] = m_Materials[id];
		}
		m_InvalidMaterials.clear();
		return invalid;
	}

	std::string AssetManager::GetMaterialsString()
	{
		std::string materialsString;
		for (int i = 0; i < m_Materials.size(); i++)
		{
			materialsString.append("Material ");
			materialsString.append(std::to_string(i));
			materialsString.push_back('\0');
		}
		return materialsString;
	}
}
