#pragma once
#include <iostream>
#include <cstdint>

class Framebuffer
{
public:
	Framebuffer(uint32_t width, uint32_t height, unsigned int textureHandle);
	~Framebuffer();

	void Bind();
	void AttachToTextureHandle(unsigned int handle);

	unsigned int GetHandle() { return m_Handle; };

private:
	unsigned int m_Handle = 0;
};
