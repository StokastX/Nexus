#include "Core/Application.h"
#include "EditorLayer.h"


#ifdef _WIN32
// CUDA-OpenGL interop requires the GL context and the CUDA context to sit on the same
// physical device. On a hybrid-graphics laptop the GL context otherwise lands on the
// integrated GPU while CUDA runs on the discrete one, and every cudaGraphicsGLRegister*
// call fails with cudaErrorOperatingSystem (304). These exported symbols are read by the
// vendor drivers at process start to force the discrete GPU. They only have an effect when
// exported from the executable itself, so they must live here and not in the Nexus library.
extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif


int main()
{
	Nexus::ApplicationSpecification applicationSpec;
	applicationSpec.name = "Nexus Lab";
	applicationSpec.windowSpec.width = 1920;
	applicationSpec.windowSpec.height = 1080;
	applicationSpec.windowSpec.startMaximized = true;

	Nexus::Application application(applicationSpec);
	application.PushLayer<Nexus::EditorLayer>();
	application.Run();
}
