#include "VertexArray.h"
#include "Platform/OpenGL/GLVertexArray.h"

namespace Nexus {

	std::shared_ptr<VertexArray> Create()
	{
		return std::make_shared<GLVertexArray>();
	}

}