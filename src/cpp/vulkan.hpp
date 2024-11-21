#pragma once

#include "pch.hpp"
#include "window.hpp"
#include "util.hpp"

/* 
 * Helper struct representing various supported formats by the SwapChain 
 */
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

/* 
 * Helper struct containing the indices to various queues in the device 
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return computeFamily.has_value() && presentFamily.has_value();
    };

    bool areDifferent() {
        return computeFamily != presentFamily;
    }
};

VkInstance createInstance();

bool allValidationLayersSupported(std::vector<const char*> validationLayers);

VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance);

VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);

int isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);

bool checkDeviceExtensionSupport(VkPhysicalDevice device, std::vector<const char*> deviceExtensions);

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

int getDeviceScore(VkPhysicalDevice device, VkSurfaceKHR surface);

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, QueueFamilyIndices queueIndices);

VkSwapchainKHR createSwapChain(Window *window,
                               VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkSurfaceKHR surface,
                               QueueFamilyIndices queueFamilies,
                               std::vector<VkImage>& swapChainImages, 
                               VkFormat& swapChainImageFormat, 
                               VkExtent2D& swapChainExtent);

VkSurfaceFormatKHR chooseSwapSurfaceMode(const std::vector<VkSurfaceFormatKHR>& availableFormats);

VkExtent2D chooseSwapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, Window *window, const VkSurfaceCapabilitiesKHR& capabilities);

std::vector<VkImageView> createSwapchainViews(VkDevice device, std::vector<VkImage> swapChainImages, VkFormat swapChainImageFormat);

VkRenderPass createRenderPass(VkDevice device, VkFormat swapChainImageFormat);

VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice device);

VkPipeline createComputePipeline(VkDevice device,
                                 VkDescriptorSetLayout descriptorSetLayout,
                                 VkPipelineLayout& pipelineLayout);

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

VkDescriptorPool createDescriptorPool(VkDevice device, int maxSets);

std::vector<VkDescriptorSet> createDescriptorSets(VkDevice device, 
                          VkDescriptorSetLayout descriptorSetLayout, 
                          VkDescriptorPool descriptorPool,
                          std::vector<VkBuffer>& uniformBuffers,
                          VkBuffer& shaderBuffer);

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size);

VkBuffer createBuffer(VkPhysicalDevice physicalDevice, 
                      VkDevice device, 
                      VkDeviceSize size, 
                      VkBufferUsageFlags usageFlags, 
                      VkMemoryPropertyFlags properties, 
                      VkDeviceMemory& bufferMemory);

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t suitableMemoryTypes, VkMemoryPropertyFlags properties);

VkCommandPool createCommandPool(VkDevice device, VkPhysicalDevice physicalDevice, u_int32_t queueFamilyIndex);

std::vector<VkFramebuffer> createFramebuffers(VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent, std::vector<VkImageView> swapChainImageViews);

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool);

void cleanupSwapChain(VkDevice device, VkSwapchainKHR swapChain, std::vector<VkFramebuffer> swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews);

VkImage createImage(VkPhysicalDevice physicalDevice, VkDevice device, VkExtent2D extent, VkFormat format, VkImageView &imageView, VkDeviceMemory &imageMemory);

VkBuffer createUniformBuffer(VkPhysicalDevice physicalDevice, 
                             VkDevice device, 
                             VkDeviceSize size,
                             VkDeviceMemory &uniformBufferMemory,
                             void* &uniformBuffersMap);

VkBuffer createShaderBuffer(VkPhysicalDevice physicalDevice, 
                            VkDevice device, 
                            VkDeviceSize size,
                            VkDeviceMemory& shaderBufferMemory, 
                            void*& shaderBufferMapped);

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool);

std::vector<VkDescriptorSet> createComputeDescriptorSets(VkDevice device, 
                                                         VkDescriptorSetLayout descriptorSetLayout, 
                                                         VkDescriptorPool descriptorPool,
                                                         VkBuffer uniformBuffer,
                                                         VkBuffer shaderBuffer,
                                                         uint offset,
                                                         VkImageView accumulatorImageView,
                                                         std::vector<VkImageView> swapChainImageViews,
                                                         VkSampler sampler);

VkSampler createSampler(VkDevice device);

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);
