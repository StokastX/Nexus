#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "Core/Application.h"
#include "EditorLayer.h"


int main()
{
	Nexus::ApplicationSpecification applicationSpec;
	applicationSpec.name = "Nexus Lab";
	applicationSpec.windowSpec.width = 1400;
	applicationSpec.windowSpec.height = 800;

	Nexus::Application application(applicationSpec);
	application.PushLayer<Nexus::EditorLayer>();
	application.Run();
}
