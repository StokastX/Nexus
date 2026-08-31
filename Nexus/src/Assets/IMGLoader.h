#pragma once
#include <iostream>
#include "Texture.h"
#include "assimp/scene.h"

namespace Nexus {

	class IMGLoader
	{
	public:
		IMGLoader();
		~IMGLoader();

		static std::shared_ptr<Texture> LoadIMG(const std::string& pathfile, Texture::Type type);

		// Load a texture embedded in an Assimp model
		static std::shared_ptr<Texture> LoadIMG(const aiTexture* texture, Texture::Type type);

	private:

	};

}
