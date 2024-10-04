#pragma once

#include "pch.hpp"
#include "window.hpp"
#include "util.hpp"


struct UniformBufferObject {
    glm::vec2 resolution;
    glm::vec2 viewportUv;
    float focalLength;
    uint time;
    glm::vec3 origin;
};

struct Sphere {
    float radius;
    bool emissive;
    alignas(16) glm::vec3 color;
    alignas(16) glm::vec3 center;
};

struct SphereShaderBufferObject {
    uint32_t count;
    Sphere spheres[];
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    // describes the rate at which to load data from memory thru vertices (vertex format)
    // here we decide that this will be bound at 0 and be of size X per vertex
    static VkVertexInputBindingDescription getBindingDescription();

    // describes the layout of the attributes for each vertex
    // here we specify how to read the various attributes we need for our bound buffers
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete();
};
