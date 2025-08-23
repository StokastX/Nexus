#include "AssetManager.h"
#include "OBJLoader.h"
#include "IMGLoader.h"
#include "Cuda/PathTracer/PathTracer.cuh"

AssetManager::AssetManager() : m_DeviceMeshesAdress(GetDeviceMeshesAdress()) { }

void AssetManager::Reset()
{
	m_Materials.clear();
	m_InvalidMaterials.clear();
	m_Textures.clear();
	m_DeviceTextures.Clear();
	m_DeviceMaterials.Clear();
	m_Meshes.clear();
}

uint32_t AssetManager::AddMesh(Mesh&& mesh)
{
	m_Meshes.push_back(std::move(mesh));
	return m_Meshes.size() - 1;
}

uint32_t AssetManager::AddMesh(const std::string name, uint32_t materialIdx, const std::vector<NXB::Triangle>& triangles, const std::vector<TriangleData>& triangleData)
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

int AssetManager::AddTexture(const Texture& texture)
{
	if (texture.pixels == nullptr)
		return -1;

	m_Textures.push_back(texture);
	m_DeviceTextures.PushBack(texture);
	return m_Textures.size() - 1;
}

void AssetManager::ApplyTextureToMaterial(int materialIdx, int diffuseMapId)
{
	m_Materials[materialIdx].baseColorMapId = diffuseMapId;
	InvalidateMaterial(materialIdx);
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

std::string AssetManager::GetMaterialTypesString()
{
	std::string materialTypes;
	materialTypes.append("Diffuse");
	materialTypes.push_back('\0');
	materialTypes.append("Dielectric");
	materialTypes.push_back('\0');
	materialTypes.append("Conductor");
	materialTypes.push_back('\0');
	return materialTypes;
}
