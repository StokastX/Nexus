#pragma once

#include "OpenGL/InteropTexture.h"
#include "Scene/Scene.h"
#include "PathTracer.h"

namespace Nexus {

	constexpr uint32_t MaxRenderResolution = 10000;

	class Renderer
	{
	public:
		Renderer(Scene* scene, uint2 resolution = make_uint2(1));
		~Renderer();

		void Reset();
		void OnResize(uint2 resolution);
		void Render();

		PathTracer* GetPathTracer() { return &m_PathTracer; }
		Scene* GetScene() { return m_Scene; }
		InteropTexture& GetTexture() { return m_RenderTexture; }
		uint2 GetResolution() { return m_RenderTexture.GetResolution(); }

	private:
		InteropTexture m_RenderTexture;
		Scene* m_Scene;

		PathTracer m_PathTracer;
	};

}
