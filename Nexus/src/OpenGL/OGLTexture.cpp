#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector_types.h>
#include "Utils/cuda_math.h"
#include "OGLTexture.h"

OGLTexture::OGLTexture(uint2 resolution)
	: m_Resolution(resolution)
{
	glGenTextures(1, &m_Handle);
    Bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution.x, resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void OGLTexture::Bind()
{
	glBindTexture(GL_TEXTURE_2D, m_Handle);
}

void OGLTexture::OnResize(uint2 resolution)
{
	Bind();
	m_Resolution = resolution;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution.x, resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

OGLTexture::~OGLTexture()
{
    glDeleteTextures(1, &m_Handle);
}


