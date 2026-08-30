#pragma once
#include <GL/glew.h>
#include "Scene/Scene.h"
#include "Shader.h"
#include "OGLTexture.h"
#include "Platform/OpenGL/GLVertexArray.h"

namespace Nexus {

	struct PixelQuery
	{
		uint2 pixel;
		int32_t instanceIdx = -1;
	};

	class OGLRenderer
	{
	public:
		OGLRenderer(Scene* scene, uint2 resolution = make_uint2(1));
		void Render(const SelectionContext& selectionContext);
		void OnResize(uint2 resolution);
		OGLTexture& GetTexture() { return m_RenderTexture; }
		OGLTexture& GetInstanceTexture() { return m_InstanceTexture; }
		void SetPixelQuery(uint2 pixel);
		PixelQuery GetPixelQuery();
		bool PixelQueryPending() { return m_PixelQueryPending; }
		Scene* GetScene() { return m_Scene; }
	private:
		void SynchronizePixelQuery();
	private:
		Scene* m_Scene;
		Shader m_Shader;
		Shader m_GridShader;
		GLFramebufferHandle m_FrameBuffer;
		GLRenderbufferHandle m_DepthStencilRbo;
		GLVertexArray m_GridVertexArray;
		OGLTexture m_RenderTexture;
		OGLTexture m_InstanceTexture;
		PixelQuery m_PixelQuery;
		bool m_PixelQueryPending = false;
	};

}