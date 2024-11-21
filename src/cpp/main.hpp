#pragma once

#include <functional>

#include "pch.hpp"
#include "window.hpp"
#include "util.hpp"
#include "vulkan.hpp"
#include "buffer-builder.hpp"

struct UniformBufferObject {
    glm::vec2 resolution;
    glm::vec2 viewportUv;
    alignas(4) float focalLength;
    alignas(4) float focusDistance;
    alignas(4) float apertureRadius;
    alignas(4) uint time;
    alignas(16) glm::vec4 origin;
    alignas(16) glm::mat4 rotation;
};

struct Material {
    alignas(16) glm::vec4 baseColor;
    alignas(16) glm::vec4 emissiveStrength;
    alignas(4) float reflectivity;
    alignas(4) float roughness;
    alignas(4) int isGlass; // todo: figure out why booleans don't work
    alignas(4) float ior;
    alignas(4) int shadeSmooth;
};

struct Triangle {
    alignas(16) glm::vec4 vertices[3];
    alignas(16) glm::vec4 normals[3];
};

struct Sphere {
    alignas(16) glm::vec3 center;
    alignas(16) float radius;
};

struct Mesh {
    uint triangle_count;
    uint offset;
    Material material;
};