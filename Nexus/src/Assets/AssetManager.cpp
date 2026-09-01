#include "AssetManager.h"
#include "SceneLoader.h"
#include "TextureLoader.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Utils/ParallelFor.h"

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

	/*
	 * Identity of the image a request names, as a string that can be compared.
	 *
	 * Keyed on the type as well as the source: the same file can serve as a base colour map in one
	 * material and a roughness map in another, and those want opposite sRGB handling. Sharing one
	 * entry between them would give one of the two the wrong decode.
	 */
	static std::string TextureKey(const TextureRequest& request)
	{
		const std::string suffix = '#' + std::to_string(static_cast<int>(request.type));

		// An embedded image has no name of its own, so its address in the aiScene is what
		// distinguishes it. Valid for as long as the import, which is why keys of this shape are
		// only ever compared within one batch -- see m_TextureIds.
		if (request.embedded)
			return '@' + std::to_string(reinterpret_cast<uintptr_t>(request.embedded)) + suffix;

		return request.path + suffix;
	}

	std::vector<int> AssetManager::AddTextures(const std::vector<TextureRequest>& requests)
	{
		std::vector<int> ids(requests.size(), -1);

		// The requests that will actually be decoded, as indices into `requests`.
		std::vector<size_t> toDecode;

		// For a request that repeats one already in `toDecode`, the index of that first request;
		// npos for every request that is not a repeat. Resolved after the ids exist, because the
		// first occurrence has no id yet at the point the repeat is found.
		std::vector<size_t> repeatOf(requests.size(), std::string::npos);

		std::unordered_map<std::string, size_t> firstRequest;

		for (size_t i = 0; i < requests.size(); i++)
		{
			const std::string key = TextureKey(requests[i]);

			// Already loaded by an earlier call -- a second model sharing a texture with the first.
			auto cached = m_TextureIds.find(key);
			if (cached != m_TextureIds.end())
			{
				ids[i] = cached->second;
				continue;
			}

			const auto [entry, isFirst] = firstRequest.emplace(key, i);
			if (isFirst)
				toDecode.push_back(i);
			else
				repeatOf[i] = entry->second;
		}

		/*
		 * The expensive part, and the only part that runs on more than one thread.
		 *
		 * Every worker writes one element of a vector sized above and reads nothing shared, so no
		 * locking is involved. TextureLoader touches no CUDA and reports failure by returning nothing
		 * rather than throwing, which is what makes it safe to call from here.
		 */
		std::vector<std::optional<Texture>> decoded(toDecode.size());

		Utils::ParallelFor(toDecode.size(), [&](size_t slot)
		{
			const TextureRequest& request = requests[toDecode[slot]];

			decoded[slot] = request.embedded
				? TextureLoader::LoadIMG(request.embedded, request.type)
				: TextureLoader::LoadIMG(request.path, request.type);
		});

		// Uploading and registering stays here, on one thread: it allocates device memory and
		// appends to m_Textures, neither of which a worker may do. Walked in request order, so the
		// ids a given scene produces do not depend on how the decode happened to be scheduled.
		for (size_t slot = 0; slot < toDecode.size(); slot++)
		{
			const size_t requestIdx = toDecode[slot];
			const int id = StoreTexture(std::move(decoded[slot]));

			ids[requestIdx] = id;

			if (id != -1 && !requests[requestIdx].embedded)
				m_TextureIds.emplace(TextureKey(requests[requestIdx]), id);
		}

		for (size_t i = 0; i < requests.size(); i++)
		{
			if (repeatOf[i] != std::string::npos)
				ids[i] = ids[repeatOf[i]];
		}

		return ids;
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
