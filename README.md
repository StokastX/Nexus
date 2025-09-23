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

![cannelle_et_fromage](https://github.com/user-attachments/assets/e02c6724-e3a1-4fbd-9866-8f13229ce71b)
![camera6](https://github.com/user-attachments/assets/dccb6e12-ba0d-4110-b665-d578748ce172)
![glass_of_water](https://github.com/user-attachments/assets/ea53185c-85b2-47a2-b302-dcb7c8b9984f)
![stormtrooper](https://github.com/user-attachments/assets/e5e1dea7-7232-434c-ac56-114b24f26f67)
![kitchen](https://github.com/user-attachments/assets/94eeb762-c1e1-448c-83b4-f7ff83c80ab5)
![ford_mustang](https://github.com/user-attachments/assets/690fe199-18fb-486a-ac88-7bcc51e3d991)
![bedroom](https://github.com/user-attachments/assets/d8387785-86cb-4c84-b1c3-0fd949b6b6c1)
![spider](https://github.com/user-attachments/assets/354a6c07-e181-4e8a-b3d3-5abd6a2570bf)
![bathroom](https://github.com/user-attachments/assets/60188c8e-1729-4d12-9eab-8592b02b38e9)

## Features
- Interactive camera with thin lens approximation: FOV, defocus blur.
- Wavefront path tracing, see [Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf). The path tracing algorithm is divided into specialized CUDA kernels accessing global work queues to get more coherent workloads and to reduce the amount of inactive threads. Kernel launches are optimized using CUDA graphs.
- Persistent threads with dynamic ray fetching, see [Aila and Laine 2009](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf). The trace kernel is launched with just enough threads to fill the device. During traversal, inactive threads will fetch new rays in the global trace queue to avoid wasting resources.
- BVH:
   - Standard SAH-based BVH (BVH2) using binned building
   - Compressed-wide BVH (BVH8), see [Ylitie et al. 2017](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf). BVH2 is collapsed into an 8-ary BVH. Nodes are compressed to 80 bytes encoding the child nodes' bounding boxes to limit memory bandwidth on the GPU.
   - GPU builder: implements the H-PLOC algorithm proposed by [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377), a high-performance GPU-oriented BVH construction method. H-PLOC builds high-quality BVHs through hierarchical clustering of spatially nearby primitives. The full algorithm is implemented in my [NexusBVH](https://github.com/StokastX/NexusBVH) library.
- The BVH is split into two parts: a top level structure (TLAS) and a bottom level structure (BLAS). This allows for multiple instances of the same mesh as well as dynamic scenes using object transforms.
- Model loader: obj, ply, fbx, gltf with Assimp
- The material system is based on the [OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/) model and supports both dielectric and metallic bases. Both bases use the GGX microfacet distribution, see [Walter et al. 2007](https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf).
   - Dielectric base:
      - Glossy-diffuse BRDF (non-physical mix between diffuse and specular).  
      - Translucent BSDF (no volumetric scattering).  

   - Metallic base: Rough conductor BSDF with the F82-tint model of [Kutz et al. 2021](https://helpx.adobe.com/substance-3d-general/adobe-standard-material/asm-technical-documentation.html).
- Importance sampling: cosine weighted for diffuse BSDFs, VNDF sampling for microfacet BSDFs (see [Heitz 2014](https://jcgt.org/published/0007/04/01/) and [Dupuy and Benyoub 2023](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14867)).
- Multiple importance sampling, see [Veach 1997](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf). BSDF sampling is combined with next event estimation (direct light sampling) and the results from both sampling strategies are weighted using the power heuristic to get low-variance results.
- Texture mapping (albedo, emissive, normal, metallic, roughness).
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

These are the main references I relied on while building this project. They range from theoretical foundations to practical guides, and I highly recommend them if you’re interested in ray tracing and physically based rendering.


#### Path Tracing (General)
- [**Eric Veach’s PhD Thesis**](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf) — The definitive resource on the theory of Monte Carlo path tracing. Although quite theoretical and code-agnostic, it helped me a lot to implement importance sampling, next event estimation, and multiple importance sampling.
- [**Physically Based Rendering: From Theory to Implementation**](https://www.pbr-book.org/4ed/contents) — This is the go-to reference for implementing a complete path tracer from the ground up.
- [**Ray Tracing Gems II**](https://www.realtimerendering.com/raytracinggems/rtg2/index.html) — A collection of modern techniques for real-time ray tracing using DXR, Vulkan, and OptiX.


#### Getting Started with Ray Tracing
- [**The Cherno’s YouTube Series**](https://www.youtube.com/playlist?list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl) — Great for beginners to understand the basics of ray tracing step by step.
- [**Ray Tracing in One Weekend Series**](https://raytracing.github.io) — A classic starting point for building a ray tracer in C++.
- [**Scratchapixel**](https://scratchapixel.com) — Excellent explanations of computer graphics fundamentals with code examples.
- [**Accelerated Ray Tracing in One Weekend in CUDA**](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) — A hands-on introduction to GPU-based ray tracing.


#### BVH (Bounding Volume Hierarchies)
- [**Jacco Bikker’s BVH Guides**](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/) — A practical introduction to SAH-based BVHs, which helped me implement my first GPU BVH builder and traversal (surprisingly fast!).
- [**Stich et al. 2009**](https://www.nvidia.in/docs/IO/77714/sbvh.pdf) — Detailed explanation of binned building and spatial splits for BVH2.
- [**Ylitie et al. 2017**](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf) — Reference for compressed wide BVHs, useful for GPU acceleration.


#### Physically Based Materials
- [**Crash Course in BRDF Implementation**](https://boksajak.github.io/files/CrashCourseBRDF.pdf) — A clear overview of diffuse and microfacet BRDFs, both theory and implementation.
- [**Walter et al. 2007**](https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf) — Basis for my implementation of rough dielectric BSDFs.
- [**Weidlich & Wilkie 2007**](https://www.cg.tuwien.ac.at/research/publications/2007/weidlich_2007_almfs/weidlich_2007_almfs-paper.pdf) — Explores layered BSDFs (planned for future use in my rough plastic BSDF).


#### Sampling
- [**TU Wien Computer Graphics Lectures**](https://www.youtube.com/watch?v=FU1dbi827LY) — Great explanation of next event estimation and multiple importance sampling.


#### GPU Optimization
- [**Aila & Laine 2009**](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf) — Key insights into GPU architecture, traversal efficiency, and persistent threads.
- [**Laine et al. 2013**](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) — About wavefront path tracing and why megakernels can be problematic.


#### Renderer Implementations
I also studied existing production and hobby renderers for reference and inspiration:
- [**Blender Cycles**](https://github.com/blender/cycles)  
- [**Tungsten Renderer**](https://github.com/tunabrain/tungsten)  
- [**Jan van Bergen’s CUDA Ray Tracer**](https://github.com/jan-van-bergen/GPU-Raytracer)  


## Dependencies

This project relies on the following libraries and frameworks:

- [**GLFW**](https://www.glfw.org) and [**GLEW**](https://glew.sourceforge.net) — Window and OpenGL context management.
- [**CUDA 12.4**](https://developer.nvidia.com/cuda-downloads) — For GPU acceleration.
- [**CUDA Helper Math**](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h) — Common operations for CUDA vector types.
- [**Assimp**](https://github.com/assimp/assimp) — Model import and loading.
- [**Dear ImGui**](https://github.com/ocornut/imgui) — User interface.
- [**stb**](https://github.com/nothings/stb) — Image import/export utilities.
- [**tinyfiledialogs**](https://sourceforge.net/projects/tinyfiledialogs/) — Simple cross-platform file dialogs.


## Models
- [LuxCore example scenes](https://luxcorerender.org/example-scenes/)
- [Blender demo scenes](https://www.blender.org/download/demo-files/)
- [Camera](https://github.com/LuisaGroup/LuisaRender) from LuisaRender
- [Stormtrooper](https://www.blendswap.com/blend/13953) by [ScottGraham](https://www.blendswap.com/profile/120125)
- [Ford mustang](https://sketchfab.com/3d-models/ford-mustang-1965-5f4e3965f79540a9888b5d05acea5943) by [Pooya_dh](https://sketchfab.com/Pooya_dh)
- [Bedroom](https://www.blendswap.com/blend/3391) by [SlykDrako](https://www.blendswap.com/profile/324)
- [Bathroom](https://www.blendswap.com/blend/12584) by [nacimus](https://www.blendswap.com/profile/72536)
