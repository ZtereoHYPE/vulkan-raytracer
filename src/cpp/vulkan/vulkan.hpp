#pragma once

#include "../pch.hpp"
#include "../util/util.hpp"
#include "../window.hpp"

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

std::tuple<vk::Image, vk::ImageView, vk::DeviceMemory>
createImage(vk::PhysicalDevice const &physicalDevice,
            vk::Device const &device,
            vk::Extent2D extent,
            vk::Format format);

std::tuple<vk::Buffer, vk::DeviceMemory, void*> createMappedBuffer(vk::PhysicalDevice const &physicalDevice,
                                                                   vk::Device const &device,
                                                                   vk::DeviceSize size,
                                                                   vk::BufferUsageFlags usage);

vk::Sampler createSampler(vk::Device const &device);

vk::CommandBuffer createCommandBuffer(vk::Device const &device, vk::CommandPool const &commandPool);

vk::DescriptorSetLayout createComputeDescriptorSetLayout(vk::Device const &device);

std::vector<vk::DescriptorSet> createComputeDescriptorSets(vk::Device const &device,
                                                           vk::DescriptorSetLayout const &descriptorSetLayout,
                                                           vk::DescriptorPool const &descriptorPool,
                                                           vk::Buffer const &uniformBuffer,
                                                           vk::Buffer const &shaderBuffer,
                                                           uint offset,
                                                           vk::ImageView const &accumulatorImageView,
                                                           std::vector<vk::ImageView> swapChainImageViews,
                                                           vk::Sampler sampler);

vk::Pipeline createComputePipeline(vk::Device const &device,
                                   vk::DescriptorSetLayout const &descriptorSetLayout,
                                   vk::PipelineLayout &pipelineLayout);

vk::DescriptorPool createDescriptorPool(vk::Device const &device, int maxSets);

void copyBuffer(vk::Device const &device,
                vk::CommandPool const &commandPool,
                vk::Queue const &queue,
                vk::Buffer &src,
                vk::Buffer &dst,
                vk::DeviceSize size);

std::tuple<vk::Buffer, vk::DeviceMemory> createBuffer(vk::PhysicalDevice const &physicalDevice,
                                                      vk::Device const &device,
                                                      vk::DeviceSize size,
                                                      vk::BufferUsageFlags usageFlags,
                                                      vk::MemoryPropertyFlags properties);

void transitionImageLayout(vk::Device const &device,
                           vk::CommandPool const &commandPool,
                           vk::Queue const &queue,
                           vk::Image const &image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout);

uint32_t findMemoryType(vk::PhysicalDevice const &physicalDevice, uint32_t suitableMemoryTypes, vk::MemoryPropertyFlags properties);

vk::CommandPool createCommandPool(vk::Device const &device, uint32_t queueFamilyIndex);
