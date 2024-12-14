#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "pch.hpp"
#include "util/buffer-builder.hpp"
#include "util/gpu-types.hpp"

struct Material {
    gpu::vec3 baseColor;
    gpu::vec3 emission;
    gpu::f32 reflectiveness;
    gpu::f32 roughness;
    gpu::f32 ior;
    gpu::boolean isGlass;
    gpu::boolean shadeSmooth;
};

struct Triangle {
    gpu::vec3 vertices[3]; // these are actually vec3 in the shader
    gpu::vec3 normals[3];  // but there was no other way to pad them correctly
};

struct Sphere {
    gpu::vec3 center;
    float radius;
};

struct Mesh {
    gpu::u32 triangle_count;
    gpu::u32 offset;
    Material material;
};

struct CameraControlsUniform {
    gpu::vec2 resolution;
    gpu::vec2 viewportUv;
    gpu::f32 focalLength;
    gpu::f32 focusDistance;
    gpu::f32 apertureRadius;
    gpu::u32 time;
    gpu::vec4 origin;
    glm::mat4 rotation;
};


class Scene {
    YAML::Node sceneFile;

    public:
        Scene(std::filesystem::path path = "scene.yaml");
        std::pair<size_t, size_t> getBufferSizes();
        void populateBuffers(BufferBuilder &meshes, BufferBuilder &triangles);

    private:
        /* This method performs some very basic validation on the scene file */
        void validateFile();

        /* These methods handle the population of triangles and spheres respectively */
        void populateTriMesh(YAML::Node mesh, BufferBuilder &meshes, BufferBuilder &triangles);
        void populateSphere(YAML::Node mesh, BufferBuilder &meshes, BufferBuilder &triangles);

        /* Returns the Material object from the current material node */
        Material getMaterial(YAML::Node);
};

inline void assertTrue(bool value);
