#include "Framebuffer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace Nexus {

	Framebuffer::Framebuffer(uint32_t width, uint32_t height, unsigned int textureHandle)
	{
		glGenFramebuffers(1, &m_Handle);
		Bind();

		// Attach the texture to the framebuffer's color attachment point
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);
	}

	void Framebuffer::Bind()
	{
		glBindFramebuffer(GL_READ_FRAMEBUFFER, m_Handle);
	}

	void Framebuffer::AttachToTextureHandle(unsigned int handle)
	{
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, handle, 0);
	}

	Framebuffer::~Framebuffer()
	{
		glDeleteFramebuffers(1, &m_Handle);
	}

}
