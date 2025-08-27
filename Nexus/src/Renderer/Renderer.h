#pragma once

#include "OpenGL/OGLTexture.h"
#include "Scene/Scene.h"
#include "PathTracer.h"

class Renderer
{
public:
	Renderer(uint2 resolution, Scene* scene);
	~Renderer();

	void Reset();
	void OnResize(uint2 resolution);
	void Render(Scene& scene, float deltaTime);
	void UnpackToTexture();

	PathTracer* GetPathTracer() { return &m_PathTracer; }
	Scene* GetScene() { return m_Scene; }
	OGLTexture& GetTexture() { return m_RenderTexture; }

private:
	uint2 m_Resolution;
	OGLTexture m_RenderTexture;
	Scene* m_Scene;

	PathTracer m_PathTracer;
};

