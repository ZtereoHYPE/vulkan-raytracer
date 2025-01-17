#pragma once

#include <functional>
#include <sys/time.h>
#include <chrono>
#include <iostream>
#include <tuple>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "window.hpp"
#include "vulkan/init.hpp"
#include "vulkan/drawing-tools.hpp"
#include "vulkan/pipeline-stages.hpp"
#include "util/util.hpp"
#include "util/buffer-builder.hpp"
#include "config/scene.hpp"
#include "config/parameters.hpp"

class RayTracerProgram {
   public:
    void run();

   private:
    Scene scene{ params.SCENE_FILE };
    Window window{
        "Vulkan RayTracer",
        scene.getCameraControls().resolution[0],
        scene.getCameraControls().resolution[1]
    };

    // Vulkan environment
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::SwapchainKHR swapChain;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Queue presentQueue;
    vk::Queue computeQueue;
    vk::CommandPool commandPool;
    vk::CommandBuffer computeCommandBuffer;

    // Buffers
    vk::Buffer uniformBuffer;
    vk::Buffer uniformStagingBuffer;
    void* uniformStagingMap = nullptr;
    vk::Buffer sceneBuffer;
    vk::Buffer rayBuffer;
    vk::Buffer hitBuffer;

    // Pipelines
    PipelineStages stages;

    // Synchronization
    vk::Fence computeInFlightFence;
    vk::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;

    // Performance measure
    uint32_t frameCounter = 0;
    std::chrono::_V2::system_clock::time_point lastFrame;

    void initVulkan();
    void initEnvironment();
    void initBuffers();
    void initPipelines();
    void mainLoop();
    void drawFrame();
    void submitCompute(uint32_t imageIndex);
    void submitPresent(uint32_t imageIndex);
    void recordComputeCommandBuffer(vk::CommandBuffer &commandBuffer, uint32_t imageIndex);
    void updateUniformBuffer(vk::CommandBuffer commandBuffer);
};