#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector_types.h>
#include "Utils/cuda_math.h"
#include "OGLTexture.h"

OGLTexture::OGLTexture(uint32_t width, uint32_t height)
    :m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_Handle);
    Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void OGLTexture::Bind()
{
	glBindTexture(GL_TEXTURE_2D, m_Handle);
}

void OGLTexture::OnResize(uint32_t width, uint32_t height)
{
	Bind();
	m_Width = width;
	m_Height = height;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

OGLTexture::~OGLTexture()
{
    glDeleteTextures(1, &m_Handle);
}


