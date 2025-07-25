#include "OBJLoader.h"
#include <vector>
#include "stb_image.h"
#include "IMGLoader.h"

Assimp::Importer OBJLoader::m_Importer;

static std::tuple<std::vector<NXB::Triangle>, std::vector<TriangleData>> GetTrianglesFromAiMesh(const aiMesh* mesh)
{
	std::vector<NXB::Triangle> triangles(mesh->mNumFaces);
	std::vector<TriangleData> triangleData(mesh->mNumFaces);

	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		float3 pos[3] = { };
		float3 normal[3] = { };
		float3 tangent[3] = { };
		float2 texCoord[3] = { };
		bool skipFace = false;

		for (int k = 0; k < 3; k++)
		{
			if (mesh->mFaces[i].mNumIndices != 3)
			{
				std::cout << "ObjLoader: a non triangle primitive with " << mesh->mFaces[i].mNumIndices << " vertices has been discarded" << std::endl;
				skipFace = true;
				continue;
			}
			unsigned int vertexIndex = mesh->mFaces[i].mIndices[k];

			aiVector3D v = mesh->mVertices[vertexIndex];
			pos[k].x = v.x;
			pos[k].y = v.y;
			pos[k].z = v.z;

			if (mesh->HasNormals())
			{
				v = mesh->mNormals[vertexIndex];
				normal[k].x = v.x;
				normal[k].y = v.y;
				normal[k].z = v.z;
			}

			if (mesh->HasTangentsAndBitangents())
			{
				v = mesh->mTangents[vertexIndex];
				tangent[k].x = v.x;
				tangent[k].y = v.y;
				tangent[k].z = v.z;
			}

			// We only deal with one tex coord per vertex for now
			if (mesh->HasTextureCoords(0))
			{
				v = mesh->mTextureCoords[0][vertexIndex];
				texCoord[k].x = v.x;
				texCoord[k].y = v.y;

			}
		}
		if (skipFace)
			continue;

		NXB::Triangle triangle(
			pos[0],
			pos[1],
			pos[2]
		);

		TriangleData data(
			normal[0],
			normal[1],
			normal[2],
			tangent[0],
			tangent[1],
			tangent[2],
			texCoord[0],
			texCoord[1],
			texCoord[2]
		);

		triangles[i] = triangle;
		triangleData[i] = data;
	}
	return std::make_tuple(triangles, triangleData);
}

// Return the list of IDs of the created materials
static std::vector<uint32_t > CreateMaterialsFromAiScene(const aiScene* scene, AssetManager* assetManager, const std::string& path)
{
	std::vector<uint32_t> materialIdx(scene->mNumMaterials);

	for (int i = 0; i < scene->mNumMaterials; i++)
	{
		aiMaterial* material = scene->mMaterials[i];
		Material newMaterial;
		newMaterial.type = Material::Type::PLASTIC;

		aiColor3D diffuse(0.0f);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
		newMaterial.plastic.albedo = make_float3(diffuse.r, diffuse.g, diffuse.b);

		aiColor3D emission(0.0f);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
		newMaterial.emissive = make_float3(emission.r, emission.g, emission.b);

		float intensity = 1.0f;
		material->Get(AI_MATKEY_EMISSIVE_INTENSITY, intensity);
		newMaterial.intensity = intensity;

		float opacity = 1.0f;
		material->Get(AI_MATKEY_OPACITY, opacity);
		newMaterial.opacity = opacity;

		float transmissionFactor = 0.0f;
		material->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor);

		// We assume every partially transmissive material is a dielectric
		if (transmissionFactor > 0.0f)
			newMaterial.type = Material::Type::DIELECTRIC;

		float ior = 1.45f;
		material->Get(AI_MATKEY_REFRACTI, ior);
		newMaterial.plastic.ior = ior;

		float shininess = 0.0f;
		if (AI_SUCCESS != aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess))
		{
			shininess = 20.0f;
		}
		newMaterial.plastic.roughness = clamp(1.0f - sqrt(shininess) / 31.62278f, 0.0f, 1.0f);

		if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
		{
			aiString mPath;
			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				Texture newTexture;
				const aiTexture* texture = scene->GetEmbeddedTexture(mPath.data);
				if (texture)
				{
					if (texture->mHeight == 0)
					{
						newTexture = IMGLoader::LoadIMG(texture);
					}
				}
				else{
					const std::string materialPath = path + mPath.C_Str();
					newTexture = IMGLoader::LoadIMG(materialPath);
				}
				newTexture.type = Texture::Type::DIFFUSE;
				newMaterial.diffuseMapId = assetManager->AddTexture(newTexture);
			}
		}
		if (material->GetTextureCount(aiTextureType_NORMALS) > 0)
		{
			aiString mPath;
			if (material->GetTexture(aiTextureType_NORMALS, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				//aiUVTransform transform;
				//material->Get(AI_MATKEY_UVTRANSFORM_NORMALS(0), transform);

				Texture newTexture;
				const aiTexture* texture = scene->GetEmbeddedTexture(mPath.data);
				if (texture)
				{
					if (texture->mHeight == 0)
					{
						newTexture = IMGLoader::LoadIMG(texture);
						newTexture.sRGB = false;
					}
				}
				else
				{
					const std::string materialPath = path + mPath.C_Str();
					newTexture = IMGLoader::LoadIMG(materialPath);
				}
				newTexture.type = Texture::Type::NORMALS;
				newMaterial.normalMapId = assetManager->AddTexture(newTexture);
			}
		}
		if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0)
		{
			aiString mPath;
			if (material->GetTexture(aiTextureType_EMISSIVE, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				Texture newTexture;
				const aiTexture* texture = scene->GetEmbeddedTexture(mPath.data);
				if (texture)
				{
					if (texture->mHeight == 0)
					{
						newTexture = IMGLoader::LoadIMG(texture);
					}
				}
				else
				{
					const std::string materialPath = path + mPath.C_Str();
					newTexture = IMGLoader::LoadIMG(materialPath);
				}
				newTexture.type = Texture::Type::EMISSIVE;
				newMaterial.emissiveMapId = assetManager->AddTexture(newTexture);
			}
		}
		materialIdx[i] = assetManager->AddMaterial(newMaterial);
	}
	return materialIdx;
}

static std::vector<uint32_t> CreateMeshesFromScene(const aiScene* scene, AssetManager* assetManager, std::vector<uint32_t> materialIdx)
{
	std::vector<uint32_t> meshIds;
	for (int i = 0; i < scene->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[i];
		auto data = GetTrianglesFromAiMesh(mesh);
		std::vector<NXB::Triangle> triangles = std::get<0>(data);
		std::vector<TriangleData> triangleData = std::get<1>(data);

		std::string meshName = mesh->mName.data;
		uint32_t mIdx = materialIdx[mesh->mMaterialIndex];
		uint32_t meshId = assetManager->AddMesh(meshName, mIdx, triangles, triangleData);
		meshIds.push_back(meshId);
	}
	return meshIds;
}

static void CreateMeshInstancesFromNode(const aiScene* assimpScene, Scene* scene, const aiNode* node, aiMatrix4x4 aiTransform, std::vector<uint32_t>& materialIds, std::vector<uint32_t>& meshIds)
{
	aiTransform = aiTransform * node->mTransformation;
	for (int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = assimpScene->mMeshes[node->mMeshes[i]];
		int32_t meshId = meshIds[node->mMeshes[i]];

		aiVector3D aiPosition, aiRotation, aiScale;
		aiTransform.Decompose(aiScale, aiRotation, aiPosition);
		float3 position = { aiPosition.x, aiPosition.y, aiPosition.z };
		float3 rotation = { Utils::ToDegrees(aiRotation.x), Utils::ToDegrees(aiRotation.y), Utils::ToDegrees(aiRotation.z) };
		float3 scale = { aiScale.x, aiScale.y, aiScale.z };

		double scaleFactor = 1.0f;
		bool result = assimpScene->mMetaData->Get("UnitScaleFactor", scaleFactor);
		scale /= scaleFactor;
		position /= scaleFactor;

		MeshInstance& meshInstance = scene->CreateMeshInstance(meshId);
		meshInstance.AssignMaterial(materialIds[mesh->mMaterialIndex]);
		meshInstance.SetTransform(position, rotation, scale);
	}

	for (int i = 0; i < node->mNumChildren; i++)
	{
		CreateMeshInstancesFromNode(assimpScene, scene, node->mChildren[i], aiTransform, materialIds, meshIds);
	}
}


void OBJLoader::LoadOBJ(const std::string& path, const std::string& filename, Scene* scene, AssetManager* assetManager)
{
	const std::string filePath = path + filename;

	// Pretransform all meshes for simplicity, but this will need to be removed
	// in the future to implement proper scene hierarchy
	const aiScene* objScene = m_Importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_TransformUVCoords | aiProcess_CalcTangentSpace);

	std::vector<Mesh> meshes;

	if (!objScene || objScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !objScene->mRootNode)
	{
		std::cout << "OBJLoader: Error loading model " << filePath << std::endl;
		return;
	}

	//double factor = 100.0f;
	//// Fix for assimp scaling FBX with a factor 100
	//scene->mMetaData->Set("UnitScaleFactor", factor);
	
	std::vector<uint32_t> materialIdx = CreateMaterialsFromAiScene(objScene, assetManager, path);
	std::vector<uint32_t> meshIdx = CreateMeshesFromScene(objScene, assetManager, materialIdx);
	CreateMeshInstancesFromNode(objScene, scene, objScene->mRootNode, aiMatrix4x4(), materialIdx, meshIdx);

	std::cout << "OBJLoader: loaded model " << filePath << " successfully" << std::endl;
}
