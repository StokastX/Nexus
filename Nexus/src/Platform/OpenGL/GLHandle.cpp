#include "GLHandle.h"

#include <GL/glew.h>

namespace Nexus::GLDelete {

	void Buffer(uint32_t handle) { glDeleteBuffers(1, &handle); }
	void VertexArray(uint32_t handle) { glDeleteVertexArrays(1, &handle); }
	void Texture(uint32_t handle) { glDeleteTextures(1, &handle); }
	void Framebuffer(uint32_t handle) { glDeleteFramebuffers(1, &handle); }
	void Renderbuffer(uint32_t handle) { glDeleteRenderbuffers(1, &handle); }
	void Program(uint32_t handle) { glDeleteProgram(handle); }

}
