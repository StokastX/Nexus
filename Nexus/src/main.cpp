#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Input.h"
#include "RayTracerApplication.h"

int WIDTH = 1400, HEIGHT = 800;

int main(void)
{
    GLFWwindow* window;

    if (!glfwInit())
    {
        std::cout << "Error initializing glfw" << std::endl;
        return -1;
    }

    window = glfwCreateWindow(WIDTH, HEIGHT, "Nexus", NULL, NULL);
    if (!window)
    {
        std::cout << "Error creating glfw window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    Input::Init(window);
    // Disable vsync (frame rate / screen refresh rate synchronization)
    //glfwSwapInterval(0);

    if (glewInit() != GLEW_OK)
        std::cout << "Error initializing GLEW" << std::endl;

    // This scope allows to free everything in the app (textures, buffers) by calling the application destructor before glfwTerminate()
    {
        RayTracerApplication rayTracerApplication(WIDTH, HEIGHT, window);

        int width, height;
        double startTime, elapsedTime;
        startTime = glfwGetTime();
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            glClear(GL_COLOR_BUFFER_BIT);

            glfwGetWindowSize(window, &width, &height);
            rayTracerApplication.OnResize(width, height);

            elapsedTime = glfwGetTime() - startTime;
            startTime = glfwGetTime();
            rayTracerApplication.Update((float)elapsedTime * 1000.0f);

            glfwSwapBuffers(window);
        }
    }
    // Free all device allocations
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaDeviceReset());
    glfwTerminate();
    return 0;
}
