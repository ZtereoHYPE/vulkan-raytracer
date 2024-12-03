#pragma once

#include <functional>

#include "pch.hpp"
#include "window.hpp"
#include "util.hpp"
#include "vulkan.hpp"
#include "buffer-builder.hpp"
#include "gpu-types.hpp"
#include "scene.hpp"

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
