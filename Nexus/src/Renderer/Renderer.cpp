#include "Renderer.h"
#include "Utils/Utils.h"


namespace Nexus {

	Renderer::Renderer(Scene* scene, uint2 resolution)
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

			m_PathTracer.Render(*m_Scene, m_RenderTexture);
		}
		else
			m_PathTracer.ResetFrameNumber();
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

}
