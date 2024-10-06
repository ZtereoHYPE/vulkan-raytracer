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
VkSwapchainKHR createSwapChain(Window window,
                               VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkSurfaceKHR surface, 
                               std::vector<VkImage>& swapChainImages, 
                               VkFormat& swapChainImageFormat, 
                               VkExtent2D& swapChainExtent,
                               QueueFamilyIndices queueFamilies);
VkSurfaceFormatKHR chooseSwapSurfaceMode(const std::vector<VkSurfaceFormatKHR>& availableFormats);
//VkExtent2D chooseSwapExtent(Window window, const VkSurfaceCapabilitiesKHR& capabilities);
VkExtent2D chooseSwapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, Window window, const VkSurfaceCapabilitiesKHR& capabilities);
std::vector<VkImageView> createImageViews(VkDevice device, std::vector<VkImage> swapChainImages, VkFormat swapChainImageFormat);
VkRenderPass createRenderPass(VkDevice device, VkFormat swapChainImageFormat);
VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device);
VkPipeline createGraphicsPipeline(VkDevice device,
                                  VkDescriptorSetLayout descriptorSetLayout,
                                  VkRenderPass renderPass, 
                                  VkPipelineLayout& pipelineLayout);
VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
VkDescriptorPool createDescriptorPools(VkDevice device, int maxSets);
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
std::vector<VkCommandBuffer> createCommandBuffers(VkDevice device, VkCommandPool commandPool);

void createSyncObjects(VkDevice device, 
                       std::vector<VkSemaphore>& imageAvailableSemaphores,
                       std::vector<VkSemaphore>& renderFinishedSemaphores, 
                       std::vector<VkFence>& inFlightFences);
void cleanupSwapChain(VkDevice device, VkSwapchainKHR swapChain, std::vector<VkFramebuffer> swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews);
