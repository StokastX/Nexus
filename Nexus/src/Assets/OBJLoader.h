#pragma once
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Geometry/Triangle.h"
#include "Assets/Mesh.h"
#include "Assets/AssetManager.h"
#include "Scene/Scene.h"

class OBJLoader
{
public:
	static void LoadOBJ(const std::string& path, const std::string& filename, Scene* scene, AssetManager* assetManager);

private:
	static Assimp::Importer m_Importer;
};
