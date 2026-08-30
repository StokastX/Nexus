#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Geometry/Triangle.h"
#include "Assets/Mesh.h"
#include "Assets/AssetManager.h"
#include "Scene/Scene.h"

namespace Nexus {

	class SceneLoader
	{
	public:
		static void LoadScene(const std::string& path, const std::string& filename, Scene* scene, AssetManager* assetManager);
	};

}
