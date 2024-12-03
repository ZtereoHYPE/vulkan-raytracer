<div align="center">
    <img src="media/logo.png" width="350px" alt="Project Logo" />
    <h1>Vulkan Ray Tracer</h1>
</div>

The goal of this project is to build a moderately capable ray tracer, able to render sphere and triangle meshes with various material types, using the GPU API Vulkan to interface with the GPU.

## Context

A common project to get programmers familiar with photorealistic rendering is to build
a ray tracer, which is a program that simulates the path that individual light rays would take in a scene, and is able to output realistic-looking images because of it.

The main disadvantage of ray tracing is its speed, or rather lack of, as calculating the paths of all of the rays can become very expensive, especially if running on a CPU.

However, ray tracing can also be considered an [Embarassingly Parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem, meaning that it can scale incredibly well with parallelization. This renders it an ideal candidate to be executed on graphics cards.

## Screenshots
These showcase the depth of field as well as various materials such as glass.

<div align="middle" float="left">
    <img src="media/screenshot1.png" width="49%" alt="Project Logo" />
    <img src="media/screenshot2.png" width="49%" alt="Project Logo" />
</div>


## Running

### Dependencies
This ray tracer uses GLFW to create the window and retrieve the vulkan surface.
As it stands, the project depends on the C++ libraries `libglfw3`, `vulkan`, and `yaml-cpp`.

On Fedora, GLFW was built from source to work properly on Wayland as version 3.4 is not released yet in the repositories. However, depending on your distribution simply installing GLFW from the package manager might work.

`yaml-cpp` can be installed from your package manager (`yaml-cpp-devel` on fedora), or can be built from the following source: https://github.com/jbeder/yaml-cpp.

#### Vulkan SDK
A simple way of obtaining all the required vulkan libraries is installing the lunarg Vulkan SDK: https://www.lunarg.com/vulkan-sdk/

Alternatively, you can use your package manager's versions, but your mileage may vary.

### Building
The project is compiled and run by the included make file:

```bash
make compile    # to compile the project
make run        # to run it
```
## References
- [Ray Tracing Gems](https://www.realtimerendering.com/raytracinggems/rtg/index.html)
- [Ray Tracing in a Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [Ray Tracing the Next Weekend](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [The blog at the bottom of the sea](https://blog.demofox.org/2020/05/16/using-blue-noise-for-raytraced-soft-shadows/)
- [Fast, Minimum Storage Ray/Triangle Intersection](https://www.graphics.cornell.edu/pubs/1997/MT97.pdf)
- [Practical Hash-based Owen Scrambling](https://jcgt.org/published/0009/04/01/)
- [TU Wien Ray Tracing resources](https://www.iue.tuwien.ac.at/phd/ertl/node4.html)
- And many more coming soon...
