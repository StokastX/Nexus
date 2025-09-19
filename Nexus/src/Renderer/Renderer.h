#pragma once

#include "OpenGL/OGLTexture.h"
#include "Scene/Scene.h"
#include "PathTracer.h"

constexpr uint32_t MaxRenderResolution = 10000;

class Renderer
{
public:
	Renderer(uint2 resolution, Scene* scene);
	~Renderer();

	void Reset();
	void OnResize(uint2 resolution);
	void Render();
	void UnpackToTexture();

	PathTracer* GetPathTracer() { return &m_PathTracer; }
	Scene* GetScene() { return m_Scene; }
	OGLTexture& GetTexture() { return m_RenderTexture; }
	uint2 GetResolution() { return m_RenderTexture.GetResolution(); }

private:
	OGLTexture m_RenderTexture;
	Scene* m_Scene;

	PathTracer m_PathTracer;
};

