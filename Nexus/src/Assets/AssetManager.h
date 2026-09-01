#pragma once

#include <iostream>
#include <set>
#include <string>
#include <optional>
#include <unordered_map>
#include "Device/DeviceVector.h"
#include "Device/DeviceSymbol.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"
#include "Geometry/BVH/BVH.h"
#include "Geometry/Triangle.h"
#include "Device/MirroredVector.h"

struct aiTexture;

namespace Nexus {

	/*
	 * One image a material wants, named but not yet loaded.
	 *
	 * The loader records these instead of loading as it goes, so that a whole scene's images can
	 * be decoded in one parallel pass. Decoding is the bulk of a scene load and is the only part
	 * of it that touches neither CUDA nor OpenGL, which is exactly what makes it the part worth
	 * spreading across threads.
	 */
	struct TextureRequest
	{
		// The source, one way or the other: `embedded` when the image travels inside the model
		// file, `path` when it is a file on disk beside it.
		std::string path;
		const aiTexture* embedded = nullptr;

		// Decides the sRGB handling as much as the role, so two requests for the same image in
		// different roles are two different textures. See Texture::IsSRGB.
		Texture::Type type = Texture::Type::DIFFUSE;
	};


	class AssetManager
	{
	public:
		AssetManager();

		void Reset();

		/*
		 * Takes ownership of the arrays: they are the largest allocations a scene load makes, and
		 * the Mesh keeps both for the lifetime of the scene, so lending them here only meant
		 * copying them into it. Pass them with std::move.
		 */
		uint32_t AddMesh(std::string name, uint32_t materialIdx, std::vector<NXB::Triangle> triangles, std::vector<TriangleData> triangleData);

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
		 *
		 * Batched rather than one at a time because the decode runs on every core at once; asking
		 * for images one by one would serialise the most expensive part of a scene load. Returns
		 * one id per request, in request order, -1 where the image could not be loaded -- which is
		 * already what Material spells as "no map".
		 *
		 * Requests naming the same image in the same role are decoded once and share an id.
		 */
		std::vector<int> AddTextures(const std::vector<TextureRequest>& requests);

		void UploadToDevice();

		// Every mirrored array UploadToDevice flushes. Reporting only on materials left a mesh
		// or texture pushed without any material change able to sit unflushed indefinitely.
		bool NeedsUpload() const { return m_Materials.Dirty() || m_Meshes.Dirty() || m_Textures.Dirty(); }

	private:
		// Uploads and registers an already-loaded texture; null means the load failed.
		int StoreTexture(std::optional<Texture> texture);

	private:
		MirroredVector<Material> m_Materials;
		MirroredVector<Mesh> m_Meshes;
		// Each texture's pixels and sampler are owned by the CudaTexture inside it; the device
		// array holds only the handles the kernels index.
		MirroredVector<Texture> m_Textures;

		// Index of an already-loaded texture, keyed by file path and type. Without it a model whose
		// materials share a base colour map loads and uploads that file once per material.
		//
		// Only file-backed textures are recorded. An embedded one is identified by nothing but its
		// address inside the aiScene, which is reused by the allocator once that scene is released
		// -- an entry surviving the import that produced it would then answer for whatever landed
		// at the same address. Embedded images are therefore shared within one batch only.
		std::unordered_map<std::string, int> m_TextureIds;

		// Device members
		DeviceSymbol<D_Mesh*> m_DeviceMeshesAdress;
	};

}
