# Nexus Renderer

Interactive physically based GPU path tracer from scratch written in C++ using CUDA and OpenGL.

- [Screenshots](#screenshots)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Resources](#resources)
- [Dependencies](#dependencies)
- [Models](#models)

## Screenshots

![canelle_et_fromage](https://github.com/user-attachments/assets/aad0c9cc-7de3-4c7d-ad65-8e506db69156)
![stormtrooper](https://github.com/user-attachments/assets/e5e1dea7-7232-434c-ac56-114b24f26f67)
![glass_of_water](https://github.com/user-attachments/assets/ea53185c-85b2-47a2-b302-dcb7c8b9984f)
![kitchen](https://github.com/user-attachments/assets/94eeb762-c1e1-448c-83b4-f7ff83c80ab5)
![ford_mustang](https://github.com/user-attachments/assets/690fe199-18fb-486a-ac88-7bcc51e3d991)
![danish_mood](https://github.com/user-attachments/assets/593b537e-eff1-4967-a327-b67fe1bb8e03)
![bedroom](https://github.com/user-attachments/assets/d8387785-86cb-4c84-b1c3-0fd949b6b6c1)
![spider](https://github.com/user-attachments/assets/354a6c07-e181-4e8a-b3d3-5abd6a2570bf)
![bathroom](https://github.com/user-attachments/assets/60188c8e-1729-4d12-9eab-8592b02b38e9)
![piano](https://github.com/user-attachments/assets/bef492f7-769e-4966-a6a6-92c0ea2768bf)
![mis_comparison](https://github.com/user-attachments/assets/214bccc0-1c9f-48f8-ac89-63549ef719f8)
<p align="center"><em>Left: multiple importance sampling. Right: naive render (BSDF importance sampling). Image rendered at 24 spp.</em></p>


## Features
- Interactive camera with thin lens approximation: FOV, defocus blur.
- Wavefront path tracing, see [Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf). The path tracing algorithm is divided into specialized CUDA kernels accessing global work queues to get more coherent workloads and to reduce the amount of inactive threads. Kernel launches are optimized using CUDA graphs.
- Persistent threads with dynamic ray fetching, see [Aila and Laine 2009](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf). The trace kernel is launched with just enough threads to fill the device. During traversal, inactive threads will fetch new rays in the global trace queue to avoid wasting resources.
- BVH:
   - Standard SAH-based BVH (BVH2) using binned building
   - Compressed-wide BVH (BVH8), see [Ylitie et al. 2017](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf). BVH2 is collapsed into an 8-ary BVH. Nodes are compressed to 80 bytes encoding the child nodes' bounding boxes to limit memory bandwidth on the GPU.
   - GPU builder: implements the H-PLOC algorithm proposed by [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377), a high-performance GPU-oriented BVH construction method. H-PLOC builds high-quality BVHs through hierarchical clustering of spatially nearby primitives. The full algorithm is implemented in my [NexusBVH](https://github.com/StokastX/NexusBVH) library.
- The BVH is split into two parts: a top level structure (TLAS) and a bottom level structure (BLAS). This allows for multiple instances of the same mesh as well as dynamic scenes using object transforms.
- Model loader: obj, ply, fbx, glb, gltf, 3ds, blend with Assimp
- Materials:
   - Diffuse BSDF (Lambertian)
   - Rough dielectric BSDF (Beckmann microfacet model, see [Walter et al. 2007](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwilsq_av4qGAxWOSFUIHdm4A64QFnoECBMQAQ&url=https%3A%2F%2Fwww.graphics.cornell.edu%2F~bjw%2Fmicrofacetbsdf.pdf&usg=AOvVaw0iX18V7ncCyVX6K-TPfdO3&opi=89978449)).
   - Rough plastic BSDF (mix between diffuse and rough specular).
   - Rough conductor BSDF.
- Importance sampling: cosine weighted for diffuse materials, VNDF sampling for rough materials.
- Multiple importance sampling, see [Veach 1997](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf). BSDF importance sampling is combined with next event estimation (direct light sampling) and the results from both sampling strategies are weighted using the power heuristic to get low-variance results.
- Texture mapping (diffuse, emissive).
- HDR environment maps.

## Prerequisites
Nexus is a CMake-based project and requires the following dependencies:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (from NVIDIA)
- [CMake](https://cmake.org/download/) version 3.22 or higher

The project has been tested on both **Windows** (with Visual Studio) and **Ubuntu**.

## Build

1. **Clone the repository with submodules**:

   ``` sh
   git clone --recurse-submodules https://github.com/StokastX/Nexus
   ```

2. **Generate the solution via cmake**:

   ``` sh
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the project**:
- On Linux: Use ```make``` on your preferred build system:

   ``` sh
   make -j
   ```
- On Windows (Visual Studio): Open the generated solution file in Visual Studio. Right-click on the Nexus target, set it as the startup project, and press F5 to build and run.

## Resources
Here are the main resources I used for this project.

#### Path tracing in general
- [Eric Veach's thesis](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf). The best resource to understand all the theory behind Monte Carlo path tracing. It is code agnostic and fairly theorical but it helped me a lot to implement importance sampling, next event estimation and multiple importance sampling.
- [Physically based rendering book](https://www.pbr-book.org/4ed/contents), the reference book for path tracing detailing a complete path tracer implementation.
- [Ray Tracing Gems II: Next Generation Real-Time Rendering with DXR, Vulkan, and OptiX](https://www.realtimerendering.com/raytracinggems/rtg2/index.html)

#### Getting started on ray tracing
- The Cherno's [Ray tracing series](https://www.youtube.com/playlist?list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl)
- [Ray Tracing in one weekend book series](https://raytracing.github.io)
- [ScratchPixel website](https://scratchapixel.com)
- To get started with CUDA ray tracing: [Accelerated Ray Tracing in one weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)

#### BVH
- Jacco Bikker's [guides](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/) on SAH-based BVHs really helped me implement my first BVH and traversal on the GPU which was surprisingly fast.
- [Stich et al. 2009](https://www.nvidia.in/docs/IO/77714/sbvh.pdf) explain in details binned building and spatial splits for BVH2.
- [Ylitie et al. 2017](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf) for compressed wide BVHs.

#### PBR materials
- [Crash Course in BRDF Implementation](https://boksajak.github.io/files/CrashCourseBRDF.pdf) detailing the theory and implementation for diffuse and microfacet models.
- [Walter et al. 2007](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwilsq_av4qGAxWOSFUIHdm4A64QFnoECBMQAQ&url=https%3A%2F%2Fwww.graphics.cornell.edu%2F~bjw%2Fmicrofacetbsdf.pdf&usg=AOvVaw0iX18V7ncCyVX6K-TPfdO3&opi=89978449). I used this paper to implement my rough dielectric BSDF.
- [Weidlich and Wilkie 2007](https://www.cg.tuwien.ac.at/research/publications/2007/weidlich_2007_almfs/weidlich_2007_almfs-paper.pdf) for layered BSDFs (not yet implemented in my path tracer, but I will use it for my rough plastic BSDF).

#### Sampling
- [Computer Graphics at TU Wien videos](https://www.youtube.com/watch?v=FU1dbi827LY) for next event estimation and multiple importance sampling.

#### GPU optimization
- [Aila and Laine 2009](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf) to understand GPU architecture, traversal optimization and persistent threads.
- [Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) for wavefront path tracing.

I also had a look at other renderer implementations such as Blender's [cycles](https://github.com/blender/cycles), [Tungsten renderer](https://github.com/tunabrain/tungsten), and [Jan van Bergen's CUDA ray tracer](https://github.com/jan-van-bergen/GPU-Raytracer).

## Dependencies
- [GLFW](https://www.glfw.org) and [GLEW](https://glew.sourceforge.net)
- [CUDA](https://developer.nvidia.com/cuda-downloads) 12.4
- [CUDA math helper](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h) for common operations on CUDA vector types
- [Assimp](https://github.com/assimp/assimp) for model loading
- [ImGui](https://github.com/ocornut/imgui) for user interface
- [stb](https://github.com/nothings/stb) for importing and exporting images
- [tinyfiledialogs](https://sourceforge.net/projects/tinyfiledialogs/)

## Models
- [LuxCore example scenes](https://luxcorerender.org/example-scenes/)
- [Blender demo scenes](https://www.blender.org/download/demo-files/)
- [Stormtrooper](https://www.blendswap.com/blend/13953) by [ScottGraham](https://www.blendswap.com/profile/120125)
- [Ford mustang](https://sketchfab.com/3d-models/ford-mustang-1965-5f4e3965f79540a9888b5d05acea5943) by [Pooya_dh](https://sketchfab.com/Pooya_dh)
- [Bedroom](https://www.blendswap.com/blend/3391) by [SlykDrako](https://www.blendswap.com/profile/324)
- [Bathroom](https://www.blendswap.com/blend/12584) by [nacimus](https://www.blendswap.com/profile/72536)
- [Piano](https://blendswap.com/blend/29080) by [Roy](https://blendswap.com/profile/1508348)