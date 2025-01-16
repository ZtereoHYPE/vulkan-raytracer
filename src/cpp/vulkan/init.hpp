#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <iostream>
#include <set>
#include <vulkan/vulkan.hpp>

#include "../util/util.hpp"
#include "../window.hpp"
#include "../config/parameters.hpp"
#include "../util/buffer-builder.hpp"

/**
 * This header contains functions used to initialize the vulkan context.
 */


/* 
 * Helper struct representing various supported formats by the SwapChain 
 */
struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

/* 
 * Helper struct containing the indices to various queues in the device 
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return computeFamily.has_value() && presentFamily.has_value();
    };

    bool areDifferent() const {
        return computeFamily != presentFamily;
    }
};

void retrieveDebugMessengerPointers(vk::Instance const &instance);

vk::Instance createInstance();

bool allValidationLayersSupported(const std::vector<const char*>& validationLayers);

vk::DebugUtilsMessengerEXT setupDebugMessenger(vk::Instance const &instance);

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                             VkDebugUtilsMessageTypeFlagsEXT messageType,
                                             const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                             void* pUserData);

vk::PhysicalDevice pickPhysicalDevice(vk::Instance const &instance, vk::SurfaceKHR const &surface);

bool isDeviceSuitable(vk::PhysicalDevice const &device, vk::SurfaceKHR const &surface);

bool checkDeviceExtensionSupport(vk::PhysicalDevice const &device, std::vector<const char*> deviceExtensions);

SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice &device, vk::SurfaceKHR const &surface);

int getDeviceScore(vk::PhysicalDevice const &device, vk::SurfaceKHR const &surface);

QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice const &device, vk::SurfaceKHR const &surface);

vk::Device createLogicalDevice(vk::PhysicalDevice const &physicalDevice, QueueFamilyIndices const &queueIndices);

vk::SwapchainKHR createSwapChain(Window &window,
                                 vk::PhysicalDevice const &physicalDevice,
                                 vk::Device const &device,
                                 vk::SurfaceKHR const &surface,
                                 QueueFamilyIndices queueFamilies,
                                 std::vector<vk::Image>& swapChainImages,
                                 vk::Format& swapChainImageFormat,
                                 vk::Extent2D& swapChainExtent);

vk::SurfaceFormatKHR chooseSwapSurfaceMode(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

vk::Extent2D chooseSwapExtent(Window &window, vk::SurfaceCapabilitiesKHR const &capabilities);

std::vector<vk::ImageView> createSwapchainViews(vk::Device const &device, std::vector<vk::Image> const &swapChainImages, vk::Format swapChainImageFormat);

uint32_t findMemoryType(vk::PhysicalDevice const &physicalDevice, uint32_t suitableMemoryTypes, vk::MemoryPropertyFlags properties);

vk::CommandPool createCommandPool(vk::Device const &device, uint32_t queueFamilyIndex);

std::tuple<vk::Image, vk::ImageView, vk::DeviceMemory>
createImage(vk::PhysicalDevice const &physicalDevice,
            vk::Device const &device,
            vk::Extent2D extent,
            vk::Format format);

vk::Sampler createSampler(vk::Device const &device);

vk::CommandBuffer createCommandBuffer(vk::Device const &device, vk::CommandPool const &commandPool);

vk::DescriptorSet createGenerateDescriptorSet(vk::Device const &device,
                                              vk::DescriptorPool const &pool,
                                              vk::Buffer const &uniformBuffer,
                                              vk::Buffer const &rayBuffer,
                                              vk::DescriptorSetLayout &layout);

vk::DescriptorSet createIntersectDescriptorSet(vk::Device const &device,
                                               vk::DescriptorPool const &pool,
                                               vk::Buffer const &uniformBuffer,
                                               vk::Buffer const &rayBuffer,
                                               vk::Buffer const &hitBuffer,
                                               vk::Buffer const &sceneBuffer,
                                               uint bvhSize,
                                               uint matSize,
                                               vk::DescriptorSetLayout &layout);

vk::DescriptorSet createShadeDescriptorSet(vk::Device const &device,
                                           vk::DescriptorPool const &pool,
                                           vk::Buffer const &uniformBuffer,
                                           vk::Buffer const &rayBuffer,
                                           vk::Buffer const &sceneBuffer,
                                           vk::Buffer const &hitBuffer,
                                           uint bvhSize,
                                           uint matSize,
                                           vk::Sampler sampler,
                                           vk::DescriptorSetLayout &layout);

vk::DescriptorSet createPostProcessDescriptorSet(vk::Device const &device,
                                                 vk::DescriptorPool const &pool,
                                                 vk::Buffer const &uniformBuffer,
                                                 vk::Buffer const &rayBuffer,
                                                 vk::DescriptorSetLayout &layout);

std::vector<vk::DescriptorSet> createFramebufferDescriptorSets(vk::Device const &device,
                                                               vk::DescriptorPool const &pool,
                                                               std::vector<vk::ImageView> &swapchainViews,
                                                               vk::Sampler sampler,
                                                               vk::DescriptorSetLayout &layout);

vk::Pipeline createComputePipeline(vk::Device const &device,
                                   std::vector<vk::DescriptorSetLayout> const &descriptorSetLayouts,
                                   std::string const &shaderPath,
                                   vk::PipelineLayout &layout);

vk::DescriptorPool createDescriptorPool(vk::Device const &device, size_t swapchainSize);

vk::Buffer createMappedBuffer(vk::PhysicalDevice const &physicalDevice,
                              vk::Device const &device,
                              vk::DeviceSize size,
                              vk::BufferUsageFlags usage,
                              vk::DeviceMemory &memory,
                              void* &memoryMap);

vk::Buffer createBuffer(vk::PhysicalDevice const &physicalDevice,
                        vk::Device const &device,
                        vk::DeviceSize size,
                        vk::BufferUsageFlags usageFlags,
                        vk::MemoryPropertyFlags properties,
                        vk::DeviceMemory &bufferMemory);


