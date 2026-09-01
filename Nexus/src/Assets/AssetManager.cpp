#include "AssetManager.h"
#include "SceneLoader.h"
#include "IMGLoader.h"
#include "Cuda/PathTracer/PathTracer.cuh"

namespace Nexus {

	AssetManager::AssetManager() : m_DeviceMeshesAdress(GetDeviceMeshesAdress()) {}

	void AssetManager::Reset()
	{
		m_Materials.Clear();
		m_Textures.Clear();
		m_TextureIds.clear();
		m_Meshes.Clear();
	}

	uint32_t AssetManager::AddMesh(const std::string& name, uint32_t materialIdx, const std::vector<NXB::Triangle>& triangles, const std::vector<TriangleData>& triangleData)
	{
		m_Meshes.EmplaceBack(name, triangles, triangleData, materialIdx);

		m_DeviceMeshesAdress = m_Meshes.DeviceData();

		return m_Meshes.Size() - 1;
	}

	void AssetManager::AddMaterial()
	{
		Material material;
		AddMaterial(material);
	}

	uint32_t AssetManager::AddMaterial(const Material& material)
	{
		const uint32_t idx = static_cast<uint32_t>(m_Materials.PushBack(material));

		// Marked dirty on creation so Scene::Update derives any mesh lights it implies. PushBack
		// has already uploaded it; this is about the lighting pass, not the transfer.
		m_Materials.MarkDirty(idx);
		return idx;
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

	int AssetManager::StoreTexture(std::optional<Texture> texture)
	{
		if (!texture)
			return -1;

		// Uploaded before it is stored: PushBack converts the element to its device form, which
		// for a Texture is the sampler handle -- so the sampler has to exist by then.
		texture->UploadToDevice();
		return static_cast<int>(m_Textures.PushBack(std::move(*texture)));
	}

	void AssetManager::UploadToDevice()
	{
		m_Materials.Flush();
		m_Meshes.Flush();
		m_Textures.Flush();
	}

	std::string AssetManager::GetMaterialsString()
	{
		std::string materialsString;
		for (int i = 0; i < m_Materials.Size(); i++)
		{
			materialsString.append("Material ");
			materialsString.append(std::to_string(i));
			materialsString.push_back('\0');
		}
		return materialsString;
	}
}
