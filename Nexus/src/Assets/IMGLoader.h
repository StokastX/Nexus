#pragma once
#include <iostream>
#include "Texture.h"
#include "assimp/scene.h"

class IMGLoader
{
public:
	IMGLoader();
	~IMGLoader();

	static std::shared_ptr<Texture> LoadIMG(const std::string& pathfile);

	// Load a texture embedded in an Assimp model
	static std::shared_ptr<Texture> LoadIMG(const aiTexture* texture);

private:

};

