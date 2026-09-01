#include "SceneLoader.h"
#include <vector>
#include "stb_image.h"
#include "TextureLoader.h"

namespace Nexus {

	static std::tuple<std::vector<NXB::Triangle>, std::vector<TriangleData>> GetTrianglesFromAiMesh(const aiMesh* mesh)
	{
		/*
		 * Reserved, not sized: a face that is not a triangle is dropped, so these end up shorter
		 * than the face count. Sizing them up front instead left the dropped face's slot behind as
		 * a zeroed entry -- a degenerate triangle at the origin, which still went into the BVH and
		 * still answered intersection queries.
		 */
		std::vector<NXB::Triangle> triangles;
		std::vector<TriangleData> triangleData;
		triangles.reserve(mesh->mNumFaces);
		triangleData.reserve(mesh->mNumFaces);

		// Hoisted out of the loops below: each asks what the mesh as a whole carries, which cannot
		// change while it is being read. Inside the per-vertex loop they were three branches
		// re-evaluated for every vertex of every face.
		const bool hasNormals = mesh->HasNormals();
		const bool hasTangents = mesh->HasTangentsAndBitangents();

		// We only deal with one tex coord per vertex for now
		const bool hasTexCoords = mesh->HasTextureCoords(0);

		for (uint32_t i = 0; i < mesh->mNumFaces; i++)
		{
			const aiFace& face = mesh->mFaces[i];

			// Tested once per face rather than once per vertex, which also stops a single bad face
			// reporting itself three times.
			if (face.mNumIndices != 3)
			{
				std::cout << "SceneLoader: a non triangle primitive with " + std::to_string(face.mNumIndices)
					+ " vertices has been discarded\n";
				continue;
			}

			float3 pos[3] = { };
			float3 normal[3] = { };
			float3 tangent[3] = { };
			float2 texCoord[3] = { };

			for (uint32_t k = 0; k < 3; k++)
			{
				const uint32_t vertexIndex = face.mIndices[k];

				const aiVector3D& v = mesh->mVertices[vertexIndex];
				pos[k] = make_float3(v.x, v.y, v.z);

				if (hasNormals)
				{
					const aiVector3D& n = mesh->mNormals[vertexIndex];
					normal[k] = make_float3(n.x, n.y, n.z);
				}

				if (hasTangents)
				{
					const aiVector3D& t = mesh->mTangents[vertexIndex];
					tangent[k] = make_float3(t.x, t.y, t.z);
				}

				if (hasTexCoords)
				{
					const aiVector3D& uv = mesh->mTextureCoords[0][vertexIndex];
					texCoord[k] = make_float2(uv.x, uv.y);
				}
			}

			triangles.emplace_back(pos[0], pos[1], pos[2]);
			triangleData.emplace_back(
				normal[0], normal[1], normal[2],
				tangent[0], tangent[1], tangent[2],
				texCoord[0], texCoord[1], texCoord[2]
			);
		}

		// Moved out, not copied: these are the largest allocations in a scene load, and returning
		// them by name would copy both into the tuple.
		return std::make_tuple(std::move(triangles), std::move(triangleData));
	}

	/*
	 * One texture map a material can carry: the assimp slot it is read from, how its pixels are to
	 * be decoded, and the Material field that holds the resulting id.
	 *
	 * A table rather than six blocks of code, because the six differed only in these three values
	 * and in nothing else.
	 */
	struct TextureSlot
	{
		aiTextureType aiType;
		Texture::Type type;

		// Pointer to member: what lets one loop write six different fields of Material.
		int32_t Material::* id;
	};

	static constexpr TextureSlot s_TextureSlots[] = {
		{ aiTextureType_DIFFUSE,                 Texture::Type::DIFFUSE,           &Material::baseColorMapId },
		{ aiTextureType_NORMALS,                 Texture::Type::NORMALS,           &Material::normalMapId },
		{ aiTextureType_DIFFUSE_ROUGHNESS,       Texture::Type::ROUGHNESS,         &Material::roughnessMapId },
		{ aiTextureType_METALNESS,               Texture::Type::METALNESS,         &Material::metalnessMapId },
		{ aiTextureType_GLTF_METALLIC_ROUGHNESS, Texture::Type::METALLICROUGHNESS, &Material::metallicRoughnessMapId },
		{ aiTextureType_EMISSIVE,                Texture::Type::EMISSIVE,          &Material::emissiveMapId }
	};

	// Which material field one entry of the request list is destined for.
	struct SlotBinding
	{
		uint32_t materialIdx;
		int32_t Material::* id;
	};

	// Return the list of IDs of the created materials
	static std::vector<uint32_t > CreateMaterialsFromAiScene(const aiScene* scene, AssetManager* assetManager, const std::string& path)
	{
		std::vector<Material> materials(scene->mNumMaterials);

		// Every image the scene asks for, collected before any of it is loaded. The loop below
		// records what each material wants rather than fetching it, so that the whole set can be
		// decoded in one pass that uses every core.
		std::vector<TextureRequest> requests;
		std::vector<SlotBinding> bindings;

		for (uint32_t i = 0; i < scene->mNumMaterials; i++)
		{
			aiMaterial* material = scene->mMaterials[i];
			Material& newMaterial = materials[i];

			aiColor3D baseColor(0.0f);
			material->Get(AI_MATKEY_BASE_COLOR, baseColor);
			newMaterial.baseColor = clamp(make_float3(baseColor.r, baseColor.g, baseColor.b), 0.0f, 1.0f);
			material->Get(AI_MATKEY_METALLIC_FACTOR, newMaterial.metalness);
			material->Get(AI_MATKEY_ROUGHNESS_FACTOR, newMaterial.roughness);
			material->Get(AI_MATKEY_SPECULAR_FACTOR, newMaterial.specularWeight);

			aiColor3D specularColor(1.0f);
			material->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
			newMaterial.specularColor = clamp(make_float3(specularColor.r, specularColor.g, specularColor.b), 0.0f, 1.0f);

			material->Get(AI_MATKEY_TRANSMISSION_FACTOR, newMaterial.transmission);
			material->Get(AI_MATKEY_REFRACTI, newMaterial.ior);

			aiColor3D emission(0.0f);
			material->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
			newMaterial.emissionColor = clamp(make_float3(emission.r, emission.g, emission.b), 0.0f, 1.0f);

			if (fmaxf(newMaterial.emissionColor) > 0.0f)
				newMaterial.intensity = 1.0f;
			material->Get(AI_MATKEY_EMISSIVE_INTENSITY, newMaterial.intensity);
			material->Get(AI_MATKEY_OPACITY, newMaterial.opacity);

			for (const TextureSlot& slot : s_TextureSlots)
			{
				aiString mPath;

				// GetTexture already reports failure for a slot that holds no texture.
				if (material->GetTexture(slot.aiType, 0, &mPath, NULL, NULL, NULL, NULL, NULL) != AI_SUCCESS)
					continue;

				TextureRequest request;
				request.type = slot.type;
				request.embedded = scene->GetEmbeddedTexture(mPath.data);

				// A model that carries its images inside itself gives no path to resolve.
				if (!request.embedded)
					request.path = path + mPath.C_Str();

				requests.push_back(std::move(request));
				bindings.push_back({ i, slot.id });
			}
		}

		// One id per request, in request order, which is what lets the bindings be walked in step.
		const std::vector<int> textureIds = assetManager->AddTextures(requests);

		for (size_t i = 0; i < bindings.size(); i++)
			materials[bindings[i].materialIdx].*bindings[i].id = textureIds[i];

		// Registered only now: a material is uploaded by AddMaterial, so it has to carry its final
		// texture ids by the time it goes in.
		std::vector<uint32_t> materialIdx(scene->mNumMaterials);
		for (uint32_t i = 0; i < scene->mNumMaterials; i++)
			materialIdx[i] = assetManager->AddMaterial(materials[i]);

		return materialIdx;
	}

	static void CreateLightsFromScene(const aiScene* assimpScene, Scene* scene)
	{
		for (uint32_t i = 0; i < assimpScene->mNumLights; i++)
		{
			Light light;
			float3 color;
			switch (assimpScene->mLights[i]->mType)
			{
			case aiLightSource_POINT:
			case aiLightSource_UNDEFINED:
				light.type = Light::Type::POINT;
				light.point.position = *(float3*)&assimpScene->mLights[i]->mPosition;
				color = *(float3*)&assimpScene->mLights[i]->mColorDiffuse;
				light.point.intensity = fmaxf(color);
				light.point.color = color / light.point.intensity;
				std::cout << "Added point light of intensity " << light.point.intensity << " and color " << light.point.color.x << " " << light.point.color.y << " " << light.point.color.z << std::endl;
				std::cout << "Attenuation constant: " << assimpScene->mLights[i]->mAttenuationConstant << ", AttenuationLinear: " << assimpScene->mLights[i]->mAttenuationLinear << std::endl;
				break;
			case aiLightSource_SPOT:
				light.type = Light::Type::SPOT;
				light.spot.position = *(float3*)&assimpScene->mLights[i]->mPosition;
				light.spot.direction = *(float3*)&assimpScene->mLights[i]->mDirection;
				color = *(float3*)&assimpScene->mLights[i]->mColorDiffuse;
				light.spot.intensity = fmaxf(color);
				light.spot.color = color / light.spot.intensity;
				light.spot.falloffStart = 1.0f / assimpScene->mLights[i]->mAngleInnerCone;
				light.spot.falloffEnd = 1.0f / assimpScene->mLights[i]->mAngleOuterCone;
				std::cout << "Added spot light of intensity " << light.spot.intensity << " and color " << light.spot.color.x << " " << light.spot.color.y << " " << light.spot.color.z << std::endl;
				std::cout << "Attenuation constant: " << assimpScene->mLights[i]->mAttenuationConstant << ", AttenuationLinear: " << assimpScene->mLights[i]->mAttenuationLinear << std::endl;
				break;
			case aiLightSource_DIRECTIONAL:
				light.type = Light::Type::DIRECTIONAL;
				color = *(float3*)&assimpScene->mLights[i]->mColorDiffuse;
				light.directional.intensity = fmaxf(color);
				light.directional.color = color / light.directional.intensity;
				light.directional.direction = *(float3*)&assimpScene->mLights[i]->mDirection;
				std::cout << "Added directional light of intensity " << light.directional.intensity << " and color " << light.directional.color.x << " " << light.directional.color.y << " " << light.directional.color.z << std::endl;
				std::cout << "Attenuation constant: " << assimpScene->mLights[i]->mAttenuationConstant << ", AttenuationLinear: " << assimpScene->mLights[i]->mAttenuationLinear << std::endl;
				break;
			default:
				std::cout << "Warning: unhandled light type" << std::endl;
				break;
			}
			if (light.type != Light::Type::UNDEFINED)
			{
				scene->AddLight(light);
			}
		}
	}

	static std::vector<uint32_t> CreateMeshesFromScene(const aiScene* scene, AssetManager* assetManager, std::vector<uint32_t> materialIdx)
	{
		std::vector<uint32_t> meshIds;
		for (int i = 0; i < scene->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[i];
			auto [triangles, triangleData] = GetTrianglesFromAiMesh(mesh);

			std::string meshName = mesh->mName.data;
			uint32_t mIdx = materialIdx[mesh->mMaterialIndex];

			// Handed over rather than lent: nothing here reads the arrays again, and AddMesh is
			// the last stop before they come to rest inside the Mesh.
			uint32_t meshId = assetManager->AddMesh(std::move(meshName), mIdx, std::move(triangles), std::move(triangleData));
			meshIds.push_back(meshId);
		}
		return meshIds;
	}

	static void CreateMeshInstancesFromNode(const aiScene* assimpScene, Scene* scene, const aiNode* node, aiMatrix4x4 aiTransform, std::vector<uint32_t>& materialIds, std::vector<uint32_t>& meshIds)
	{
		aiTransform = aiTransform * node->mTransformation;

		aiVector3D aiPosition, aiRotation, aiScale;
		aiTransform.Decompose(aiScale, aiRotation, aiPosition);

		double scaleFactor = 1.0f;
		bool result = assimpScene->mMetaData->Get("UnitScaleFactor", scaleFactor);

		aiMatrix4x4 rotationMatrix;
		rotationMatrix = rotationMatrix.FromEulerAnglesXYZ(aiRotation);

		// For some reason in assimp the transform of a light is given by a node if they both have the same name
		for (uint32_t i = 0; i < assimpScene->mNumLights; i++)
		{
			aiLight* assimpLight = assimpScene->mLights[i];
			if (node->mName == assimpLight->mName)
			{
				auto light = scene->GetLights().Mutate(i);
				aiVector3D position, direction;
				switch (light->type)
				{
				case Light::Type::POINT:
					position = aiTransform * assimpLight->mPosition;
					light->point.position = *(float3*)&position / scaleFactor;
					break;
				case Light::Type::DIRECTIONAL:
					direction = rotationMatrix * assimpLight->mDirection;
					light->directional.direction = *(float3*)&direction;
					break;
				case Light::Type::SPOT:
					position = aiTransform * assimpLight->mPosition;
					direction = rotationMatrix * assimpLight->mDirection;
					light->spot.position = *(float3*)&position / scaleFactor;
					light->spot.direction = *(float3*)&direction;
					break;
				default:
					break;
				}
			}
		}

		// Same for the camera
		if (assimpScene->HasCameras())
		{
			aiCamera* assimpCamera = assimpScene->mCameras[0];
			if (assimpCamera->mName == node->mName)
			{
				std::shared_ptr<Camera> camera = scene->GetCamera();
				aiVector3D position = aiTransform * assimpCamera->mPosition;
				aiVector3D lookAt = rotationMatrix * assimpCamera->mLookAt;
				aiVector3D upDirection = rotationMatrix * assimpCamera->mUp;
				aiVector3D rightDirection = lookAt ^ upDirection;
				camera->SetPosition(*(float3*)&position);
				camera->SetForwardDirection(*(float3*)&lookAt);
				camera->SetRightDirection(*(float3*)&rightDirection);
				camera->SetHorizontalFOV(Utils::ToDegrees(assimpCamera->mHorizontalFOV));
			}
		}

		for (int i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = assimpScene->mMeshes[node->mMeshes[i]];
			int32_t meshId = meshIds[node->mMeshes[i]];

			float3 position = { aiPosition.x, aiPosition.y, aiPosition.z };
			float3 rotation = { Utils::ToDegrees(aiRotation.x), Utils::ToDegrees(aiRotation.y), Utils::ToDegrees(aiRotation.z) };
			float3 scale = { aiScale.x, aiScale.y, aiScale.z };

			scale /= scaleFactor;
			position /= scaleFactor;

			scene->CreateMeshInstance(meshId, materialIds[mesh->mMaterialIndex], position, rotation, scale);
		}

		for (int i = 0; i < node->mNumChildren; i++)
		{
			CreateMeshInstancesFromNode(assimpScene, scene, node->mChildren[i], aiTransform, materialIds, meshIds);
		}
	}


	void SceneLoader::LoadScene(const std::string& path, const std::string& filename, Scene* scene, AssetManager* assetManager)
	{
		const std::string filePath = path + filename;

		Assimp::Importer importer;
		const aiScene* objScene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_TransformUVCoords | aiProcess_CalcTangentSpace);

		std::vector<Mesh> meshes;

		if (!objScene || objScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !objScene->mRootNode)
		{
			std::cout << "SceneLoader: Error loading model " << filePath << std::endl;
			return;
		}

		//double factor = 100.0f;
		//// Fix for assimp scaling FBX with a factor 100
		//objScene->mMetaData->Set("UnitScaleFactor", factor);

		std::vector<uint32_t> materialIdx = CreateMaterialsFromAiScene(objScene, assetManager, path);
		std::vector<uint32_t> meshIdx = CreateMeshesFromScene(objScene, assetManager, materialIdx);
		CreateLightsFromScene(objScene, scene);
		CreateMeshInstancesFromNode(objScene, scene, objScene->mRootNode, aiMatrix4x4(), materialIdx, meshIdx);

		std::cout << "SceneLoader: loaded model " << filePath << " successfully" << std::endl;
	}
};
