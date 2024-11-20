#pragma once

#include "pch.hpp"
#include "window.hpp"
#include "util.hpp"

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
    alignas(4) bool isGlass;
    alignas(4) float ior;
    alignas(4) bool shadeSmooth;
};

struct Triangle {
    alignas(16) glm::vec4 vertices[3];
    alignas(16) glm::vec4 normals[3];
};

struct Mesh {
    int is_sphere;
    float sphere_radius;
    uint triangle_count;
    uint offset;
    Material material;
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


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger);

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator);

VkInstance createInstance();
std::vector<const char*> getRequiredExtensions();
bool allValidationLayersSupported(std::vector<const char*> validationLayers);
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance);
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData);
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
                               std::vector<VkImage>& swapChainImages, 
                               VkFormat& swapChainImageFormat, 
                               VkExtent2D& swapChainExtent,
                               QueueFamilyIndices queueFamilies);
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
void createBuffer(VkPhysicalDevice physicalDevice, 
                  VkDevice device, 
                  VkDeviceSize size, 
                  VkBufferUsageFlags usageFlags, 
                  VkMemoryPropertyFlags properties, 
                  VkBuffer& buffer, 
                  VkDeviceMemory& bufferMemory);
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t suitableMemoryTypes, VkMemoryPropertyFlags properties);
VkCommandPool createCommandPool(VkDevice device, VkPhysicalDevice physicalDevice, QueueFamilyIndices queueFamilyIndices);
std::vector<VkFramebuffer> createFramebuffers(VkDevice device, VkRenderPass renderPass, VkExtent2D swapChainExtent, std::vector<VkImageView> swapChainImageViews);
VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool);
void cleanupSwapChain(VkDevice device, VkSwapchainKHR swapChain, std::vector<VkFramebuffer> swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews);
VkImage createImage(VkPhysicalDevice physicalDevice, VkDevice device, VkExtent2D extent, VkFormat format, VkImageView &imageView, VkDeviceMemory &imageMemory);
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
