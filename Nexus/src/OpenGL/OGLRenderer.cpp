#include "OGLRenderer.h"
#include "GL/glew.h"
#include "imgui.h"

OGLRenderer::OGLRenderer(uint2 resolution, Scene* scene)
	: m_Scene(scene), m_RenderTexture(resolution), m_InstanceTexture(resolution), m_PixelQuery(),
	m_Shader("../../Nexus/src/OpenGL/Shaders/layout.vert", "../../Nexus/src/OpenGL/Shaders/layout.frag"),
	m_GridShader("../../Nexus/src/OpenGL/Shaders/grid.vert", "../../Nexus/src/OpenGL/Shaders/grid.frag")
{
	glEnable(GL_STENCIL_TEST);

	glGenFramebuffers(1, &m_FrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);

	glGenRenderbuffers(1, &m_DepthStencilRbo);
	glBindRenderbuffer(GL_RENDERBUFFER, m_DepthStencilRbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, resolution.x, resolution.y);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_DepthStencilRbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_RenderTexture.GetHandle(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_InstanceTexture.GetHandle(), 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	float gridVertices[18] = { -1000.0f, 0.0f, -1000.0f,
								-1000.0f, 0.0f, 1000.0f,
								1000.0f, 0.0f, -1000.0f,

								1000.0f, 0.0f, -1000.0f,
								-1000.0f, 0.0f, 1000.0f,
								1000.0f, 0.0f, 1000.0f };

	glGenVertexArrays(1, &m_GridVao);
	glGenBuffers(1, &m_GridVbo);
	glBindVertexArray(m_GridVao);

	glBindBuffer(GL_ARRAY_BUFFER, m_GridVbo);
	glBufferData(GL_ARRAY_BUFFER, 18 * sizeof(float), gridVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
	glEnableVertexAttribArray(0);
}

OGLRenderer::~OGLRenderer()
{

}

void OGLRenderer::Render(const SelectionContext& selectionContext)
{
	m_Shader.Use();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);

	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glClearColor(0.24f, 0.24f, 0.24f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

	if (m_PixelQueryPending)
	{
		glDrawBuffer(GL_COLOR_ATTACHMENT1);
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		uint32_t drawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, drawBuffers);

		m_Shader.SetBool("uWritePicking", true);
	}
	else
		m_Shader.SetBool("uWritePicking", false);

	glClearStencil(0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glStencilMask(0x00);

	std::shared_ptr<Camera> camera = m_Scene->GetCamera();
	m_Shader.SetMat4("uView", camera->GetViewMatrix());
	m_Shader.SetMat4("uProj", camera->GetProjectionMatrix());

	float3 lightDirCam = normalize(make_float3(-0.4f, 0.4f, 1.0f));
	float3 lightDirWorld = normalize(camera->GetTransform().TransformVector(lightDirCam));
	m_Shader.SetVec3("uLightDirWorld", lightDirWorld);
	m_Shader.SetVec3("uCamPosWorld", camera->GetTransform().GetTranslation());
	m_Shader.SetBool("uOutline", false);

	std::vector<Mesh>& meshes = m_Scene->GetAssetManager().GetMeshes();
	std::vector<MeshInstance>& meshInstances = m_Scene->GetMeshInstances();

	int32_t selectIdx = selectionContext.type == SelectionContext::Type::INSTANCE ? selectionContext.idx : -1;

	for (uint32_t i = 0; i < meshInstances.size(); i++)
	{
		if (m_PixelQueryPending)
		{
			uint32_t r = ((i + 1) & 0x000000FF) >> 0;
			uint32_t g = ((i + 1) & 0x0000FF00) >> 8;
			uint32_t b = ((i + 1) & 0x00FF0000) >> 16;
			m_Shader.SetVec4("uPickingColor", make_float4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f));
		}

		if (i == selectIdx)
		{
			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilMask(0xFF);
			glStencilOp(GL_KEEP, GL_REPLACE, GL_REPLACE);
		}

		MeshInstance& meshInstance = meshInstances[i];
		m_Shader.SetMat4("uModel", meshInstance.GetTransfrom());
		m_Shader.SetMat4("uModelInvTrans", meshInstance.GetTransfrom().Inverted().Transposed());
		glBindVertexArray(meshes[meshInstance.meshIdx].vao);
		glDrawArrays(GL_TRIANGLES, 0, 3 * meshes[meshInstance.meshIdx].triangles.size());

		if (i == selectIdx)
			glStencilMask(0x00);
	}

	// Draw selected object outline
	if (selectIdx >= 0)
	{
		MeshInstance& meshInstance = meshInstances[selectionContext.idx];

		glStencilMask(0xFF);
		glDepthFunc(GL_ALWAYS);
		glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

		m_Shader.SetMat4("uModel", meshInstance.GetTransfrom());
		m_Shader.SetBool("uOutline", true);

		glLineWidth(4.0f * ImGui::GetWindowDpiScale());
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		glBindVertexArray(meshes[meshInstance.meshIdx].vao);
		glDrawArrays(GL_TRIANGLES, 0, 3 * meshes[meshInstance.meshIdx].triangles.size());

		m_Shader.SetBool("uOutline", false);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDepthFunc(GL_LESS);
		glStencilFunc(GL_ALWAYS, 1, 0xFF);
	}

	// Draw grid
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	m_GridShader.Use();
	m_GridShader.SetMat4("uView", camera->GetViewMatrix());
	m_GridShader.SetMat4("uProj", camera->GetProjectionMatrix());
	m_GridShader.SetVec3("uCamPosWorld", camera->GetPosition());
	glBindVertexArray(m_GridVao);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OGLRenderer::OnResize(uint2 resolution)
{
	if ((m_RenderTexture.GetResolution().x != resolution.x || m_RenderTexture.GetResolution().y != resolution.y)
		&& resolution.x != 0 && resolution.y != 0)
	{
		m_RenderTexture.OnResize(resolution);
		m_InstanceTexture.OnResize(resolution);
		glBindRenderbuffer(GL_RENDERBUFFER, m_DepthStencilRbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, resolution.x, resolution.y);
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);
		glViewport(0, 0, resolution.x, resolution.y);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void OGLRenderer::SynchronizePixelQuery()
{
	glFlush();
	glFinish();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);
	GLubyte data[4];
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(m_PixelQuery.pixel.x, m_PixelQuery.pixel.y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_PixelQuery.instanceIdx = data[0] + data[1] * 256 + data[2] * 256 * 256 - 1;
}

void OGLRenderer::SetPixelQuery(uint2 pixel)
{
	m_PixelQueryPending = true;
	m_PixelQuery.pixel = pixel;
}

PixelQuery OGLRenderer::GetPixelQuery()
{
	SynchronizePixelQuery();
	m_PixelQueryPending = false;
	return m_PixelQuery;
}
