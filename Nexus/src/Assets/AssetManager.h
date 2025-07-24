#pragma once

#include <iostream>
#include <set>
#include "Device/DeviceVector.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"
#include "Geometry/BVH/BVH.h"
#include "Geometry/Triangle.h"

class AssetManager
{
public:
	AssetManager();

	void Reset();

	uint32_t AddMesh(Mesh&& mesh);
	uint32_t AddMesh(const std::string name, uint32_t materialIdx, const std::vector<NXB::Triangle>& triangles, const std::vector<TriangleData>& triangleData);

	void AddMaterial();
	uint32_t AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

	DeviceVector<Material, D_Material>& GetDeviceMaterials() { return m_DeviceMaterials; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceDiffuseMaps() { return m_DeviceDiffuseMaps; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceEmissiveMaps() { return m_DeviceEmissiveMaps; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceNormalMaps() { return m_DeviceNormalMaps; }
	DeviceVector<Mesh, D_Mesh>& GetDeviceMeshes() { return m_DeviceMeshes; }

	const DeviceVector<Material, D_Material>& GetDeviceMaterials() const { return m_DeviceMaterials; }
	const DeviceVector<Texture, cudaTextureObject_t>& GetDeviceDiffuseMaps() const { return m_DeviceDiffuseMaps; }
	const DeviceVector<Texture, cudaTextureObject_t>& GetDeviceEmissiveMaps() const { return m_DeviceEmissiveMaps; }
	const DeviceVector<Texture, cudaTextureObject_t>& GetDeviceNormalMaps() const { return m_DeviceNormalMaps; }

	int AddTexture(const Texture& texture);
	void ApplyTextureToMaterial(int materialIdx, int diffuseMapId);

	bool SendDataToDevice();

	bool IsInvalid() { return m_InvalidMaterials.size() > 0; }

private:
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<Texture> m_DiffuseMaps;
	std::vector<Texture> m_EmissiveMaps;
	std::vector<Texture> m_NormalMaps;
	std::vector<Mesh> m_Meshes;

	// Device members
	DeviceVector<Material, D_Material> m_DeviceMaterials;
	DeviceVector<Texture, cudaTextureObject_t> m_DeviceDiffuseMaps;
	DeviceVector<Texture, cudaTextureObject_t> m_DeviceEmissiveMaps;
	DeviceVector<Texture, cudaTextureObject_t> m_DeviceNormalMaps;
	DeviceVector<Mesh, D_Mesh> m_DeviceMeshes;
	DeviceInstance<D_Mesh*> m_DeviceMeshesAdress;
};
