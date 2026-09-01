#pragma once
#include <iostream>
#include <optional>
#include "Texture.h"
#include "assimp/scene.h"

namespace Nexus {

	class TextureLoader
	{
	public:
		TextureLoader();
		~TextureLoader();

		static std::optional<Texture> LoadIMG(const std::string& pathfile, Texture::Type type);

		// Load a texture embedded in an Assimp model
		static std::optional<Texture> LoadIMG(const aiTexture* texture, Texture::Type type);

	private:

	};

}
