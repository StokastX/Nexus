#include "Renderer.h"
#include "Utils/Utils.h"


Renderer::Renderer(uint2 resolution, Scene* scene)
	: m_Scene(scene),
	m_RenderTexture(resolution), m_PathTracer(resolution)
{
}

Renderer::~Renderer()
{
}

void Renderer::Reset()
{
	m_PathTracer.ResetFrameNumber();
}

void Renderer::Render()
{ 
	if (m_Scene->IsInvalid())
	{
		m_Scene->Update();
		m_PathTracer.ResetFrameNumber();
	}

	// Launch cuda path tracing kernel, writes the viewport into the pixelbuffer
	if (!m_Scene->IsEmpty())
	{
		m_PathTracer.UpdateDeviceScene(*m_Scene);
		m_PathTracer.Render(*m_Scene);

		// Unpack the pixel buffer written by cuda to the renderer texture
		UnpackToTexture();
	}
	else
		m_PathTracer.ResetFrameNumber();
}

void Renderer::UnpackToTexture()
{
	m_RenderTexture.Bind();
	const PixelBuffer& pixelBuffer = m_PathTracer.GetPixelBuffer();
	pixelBuffer.Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_RenderTexture.GetWidth(), m_RenderTexture.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pixelBuffer.Unbind();
}

void Renderer::OnResize(uint2 resolution)
{
	if ((m_RenderTexture.GetResolution().x != resolution.x || m_RenderTexture.GetResolution().y != resolution.y)
		&& resolution.x != 0 && resolution.y != 0 && resolution.x <= MaxRenderResolution && resolution.y <= MaxRenderResolution)
	{
		RenderSettings& renderSettings = m_Scene->GetRenderSettings();
		renderSettings.resolution = resolution;
		m_PathTracer.OnResize(resolution);
		m_Scene->GetCamera()->OnResize(resolution);
		m_RenderTexture.OnResize(resolution);
	}
}
