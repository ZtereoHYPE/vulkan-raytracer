#include "vulkan.hpp"

const bool USE_LLVMPIPE = false;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
//const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_api_dump"};

const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};


// Retrieved function pointers
// todo: make the variables static such that they only have to retrieved on the first call
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

std::array<VkVertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
            std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions {};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
}

VkVertexInputBindingDescription Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription {};
    bindingDescription.binding = 0; 
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // move to the next data after each vertex and not instance

    return bindingDescription; 
}

bool QueueFamilyIndices::isComplete() {
    return graphicsFamily.has_value() && computeFamily.has_value() && presentFamily.has_value();
}


VkInstance createInstance() {
    // info about program
    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello woooorld";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // info about "vulkan context"
    VkInstanceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto requiredExts = getRequiredExtensions();

    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExts.size());
    createInfo.ppEnabledExtensionNames = requiredExts.data();

    // validation layers
    if (enableValidationLayers) {
        if (!allValidationLayersSupported(validationLayers)) {
            throw new std::runtime_error("Some validation layers i need aren't supported");
        }

        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    // create actual instance
    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }

    printf("log: Made instance wtih %d extensions!\n", (int)requiredExts.size());

    return instance;
}

// todo: this could be moved to the glfw stuff as it's the extensions required by glfw
std::vector<const char*> getRequiredExtensions() {
    uint32_t count = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&count);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool allValidationLayersSupported(std::vector<const char*> validationLayers) {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            printf("log: log: could not find %s\n", layerName);
            return false;
        }
    }

    return true;
}

// todo: we might have to keep the debug messenger to free it later
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance) {
    if (!enableValidationLayers) return nullptr;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;

    VkDebugUtilsMessengerEXT debugMessenger;
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }

    printf("log: Enabled all validation layers. NOTE: Instance creation is not covered!\n");

    return debugMessenger;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

/* 
 * Picks the best physical device to present to the created surface.
 * Depends on the surface because a specific device might not be able to present on one.
 */
VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());


    int bestScore = 0;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    char bestName[256];

    for (const auto& device : devices) {
        if (!isDeviceSuitable(device, surface)) continue;

        int score = getDeviceScore(device, surface);

        if (score > bestScore) {
            bestDevice = device;

            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            memcpy(bestName, deviceProperties.deviceName, 255);
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    printf("log: picked device %s\n", bestName);

    return bestDevice;
}

/* 
 * Checks if a device is suitable for rendering on a surface.
 *
 * Suitable in this case means that it has a queue families that support all of the required
 * features, and that it supports creating a swapchain on the surface.
 */
int isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    bool extensionsSupported = checkDeviceExtensionSupport(device, deviceExtensions);
    bool swapChainAdequate = false;

    // only query swapchain details after we've made sure the extension is supported
    if (extensionsSupported) {
        SwapChainSupportDetails details = querySwapChainSupport(device, surface);
        swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
    }

    QueueFamilyIndices familyIndexes = findQueueFamilies(device, surface);

    bool suitable = familyIndexes.isComplete() && swapChainAdequate;
    return suitable;
}

/* 
 * Checks if a device supports all the given extensions
 */
bool checkDeviceExtensionSupport(VkPhysicalDevice device, std::vector<const char*> deviceExtensions) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    // remove from requiredExtensions the extensions, and check if empty at the end.
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

int getDeviceScore(VkPhysicalDevice device, VkSurfaceKHR surface) {
    // discrete GPUs get the absolute priority
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    bool isDiscrete = deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;

    // devices whose queues are in the same family are better
    QueueFamilyIndices indices = findQueueFamilies(device, surface);
    bool queuesInSameFamily = (indices.graphicsFamily == indices.presentFamily);

    // if we need LLVMpipe then it gets the highest priority
    bool isLlvmpipe = !strcmp(deviceProperties.deviceName, "llvmpipe (LLVM 19.1.0, 256 bits)");
    int needLlvmpipe = USE_LLVMPIPE ? isLlvmpipe : -isLlvmpipe; // if we use llvmpipe then it has positive, else negative value

    int score = needLlvmpipe * 100 + isDiscrete * 10 + queuesInSameFamily * 1;

    return score;
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = {0};

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int family = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = family;
        }

        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = family;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, family, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = family;
        }

        family++;

        // we stop once found all the needed queue families
        if (indices.computeFamily.has_value() && indices.graphicsFamily.has_value() && indices.presentFamily.has_value()) {
            break;
        }
    }

    return indices;
}

VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, QueueFamilyIndices queueIndices) {
    // create a queueCreateInfo for each of the required ones
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    std::set<uint32_t> uniqueQueueFamilies = {
        queueIndices.graphicsFamily.value(), 
        queueIndices.presentFamily.value(),
        //queueIndices.computeFamily.value()
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures {};

    VkDeviceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = uniqueQueueFamilies.size();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    // these are ignored and only set for compatibility purposes
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    printf("log: Created logical device.\n");

    return device;
}

VkSwapchainKHR createSwapChain(Window *window,
                               VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkSurfaceKHR surface, 
                               std::vector<VkImage>& swapChainImages, 
                               VkFormat& swapChainImageFormat, 
                               VkExtent2D& swapChainExtent,
                               QueueFamilyIndices queueFamilies) {

    SwapChainSupportDetails details = querySwapChainSupport(physicalDevice, surface);


    if (!(details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT) 
        || !(details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT)) {
        throw std::runtime_error("Unfortunately the current GPU's swapchain images do not support being rendered to from a compute shader");
    }

    // out of the ones available for our current device, pick the "best" modes
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceMode(details.formats);
    swapChainExtent = chooseSwapExtent(physicalDevice, surface, window, details.capabilities);

    uint32_t imageCount = details.capabilities.minImageCount + 1; // to allow "internal operations" to complete

    // check if it's not more than we can afford (0 == unlimited)
    if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount) {
        imageCount = details.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = swapChainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
    createInfo.clipped = VK_FALSE; // for temporal effects the whole image needs to be rendered

    // if an image owned by single queue family, no need to check for implicit transfers: best performance
    //  might be interesting starting point to look into gpu archtectures
    if (queueFamilies.graphicsFamily != queueFamilies.presentFamily != queueFamilies.computeFamily) {
        uint32_t queueFamilyIndices[] = {queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value(), queueFamilies.computeFamily.value()};

        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; 
        createInfo.queueFamilyIndexCount = 3;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
        printf("log: Unfortunately our queues are split.\n");
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
        printf("log: Nice! Our queues are all in the same family!\n");
    }

    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // alpha used for blending in compositor
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // fifo should always be supported
    createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;

    createInfo.oldSwapchain = VK_NULL_HANDLE; // we don't have an old swapchain to "recycle"

    VkSwapchainKHR swapChain;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;

    printf("log: Created swapchain.\n");

    return swapChain;
}

VkSurfaceFormatKHR chooseSwapSurfaceMode(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_R8G8B8A8_UNORM && // changed to supported format for compute shaders
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    // else just return whatever the first format is
    return availableFormats[0];
}

VkExtent2D chooseSwapExtent(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, Window *window, const VkSurfaceCapabilitiesKHR& capabilities) {
    // 0xFFFFFFFF -> "the surface size/ will be determined by the extent of a swapchain targeting it"
    if (capabilities.currentExtent.width != -1) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        window->getFramebufferSize(&width, &height);

        VkExtent2D extent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return extent;
    }
}

std::vector<VkImageView> createSwapchainViews(VkDevice device, std::vector<VkImage> swapChainImages, VkFormat swapChainImageFormat) {
    std::vector<VkImageView> swapChainImageViews;
    swapChainImageViews.resize(swapChainImages.size());

    for (uint64_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    printf("log: Created %d image views for our swapchain!\n", swapChainImageViews.size());

    return swapChainImageViews;
}

VkImage createImage(VkPhysicalDevice physicalDevice, VkDevice device, VkExtent2D extent, VkFormat format, VkImageView &imageView, VkDeviceMemory &imageMemory) {
    VkImageCreateInfo imageInfo {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D(extent.width, extent.height, 1);
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // only ever accessed by 1 family at a time
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT; // both stored to and sampled
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate the needed memory
    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);

    VkImageViewCreateInfo viewInfo {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }

    return image;
}

VkSampler createSampler(VkDevice device) {
    VkSamplerCreateInfo samplerInfo {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.unnormalizedCoordinates = VK_TRUE;

    VkSampler sampler;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    return sampler;
}

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  //directly submitted to queue, but not called from primary
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    printf("log: Created command buffers\n");

    return commandBuffer;
}

VkRenderPass createRenderPass(VkDevice device, VkFormat swapChainImageFormat) {
    // Everything in vulkan is drawn thru render passes made of steps called sub-passes
    //  that our commands "can invoke".
    // Attachments are input/output data and images used in a sub-pass.

    // in this case we have a color attachment (no stencil or whatever)
    VkAttachmentDescription colorAttachment {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear to black before render
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // we keep in memory the contents of the framebuffer
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we dont care what state it was in before
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // to be presented in swapchain by the end

    // this is used such that we can refer to the same attachment in multiple subpasses
    //  without having to duplicate it or whatnot
    VkAttachmentReference colorAttachmentRef {};
    colorAttachmentRef.attachment = 0; // reference in the attachment array (0 is first index)
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // we intend to draw to it
    // todo: since the layout here will be applied before doing anything to the attachment
    //       then why do we set initialLayout there to undefined/we dont care??

    // we just have 1 subpass rn
    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // it's a graphic subpass
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // todo: investigate further.
    // This is meant to solve the problem that we might start transitioning the memory layout (to colorattachment)
    //  before we acquire the image because the semaphore only blocks at the output to the image. We could solve this by
    //  just changing to start of pipeline but this is tighter and more efficient (still allows for *some* prep to happen)
    // VK_SUBPASS_EXTERNAL is the "implicit" subpass that happens around our pass (src = operation right before, whatever it was)

    // "subpass dst depends on src. Specifically, its own pipeline's dstStage stdAccess operation depends on srcStage's srcAccess op"
    VkSubpassDependency dependency {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL; 
    dependency.dstSubpass = 0; // this refers to subpass number 0

    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_NONE;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  // writing to our color attachment is gonna wait on the right stage
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkRenderPass renderPass;
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!!!!!!!");
    }
    printf("log: Created basic render pass!\n");

    return renderPass;
}

VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice device) {
    // This uniform will contain the camera data
    VkDescriptorSetLayoutBinding uboLayoutBinding {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // only for compute

    // This buffer will contain the scene data
    VkDescriptorSetLayoutBinding sboLayoutBinding {};
    sboLayoutBinding.binding = 1;
    sboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sboLayoutBinding.descriptorCount = 1;
    sboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // This is the image where the ray traced frames will be accumulated
    VkDescriptorSetLayoutBinding accumulatorLayoutBinding {};
    accumulatorLayoutBinding.binding = 2;
    accumulatorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    accumulatorLayoutBinding.descriptorCount = 1;
    accumulatorLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // This is the swapchain image that will be bound
    VkDescriptorSetLayoutBinding swapchainLayoutBinding {};
    swapchainLayoutBinding.binding = 3;
    swapchainLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; // todo: maybe storage_texel instead?
    swapchainLayoutBinding.descriptorCount = 1;
    swapchainLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        uboLayoutBinding, sboLayoutBinding, accumulatorLayoutBinding, swapchainLayoutBinding
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout");
    }

    return layout;
}

std::vector<VkDescriptorSet> createComputeDescriptorSets(VkDevice device, 
                                           VkDescriptorSetLayout descriptorSetLayout, 
                                           VkDescriptorPool descriptorPool,
                                           VkBuffer uniformBuffer,
                                           VkBuffer shaderBuffer,
                                           VkImageView accumulatorImageView,
                                           std::vector<VkImageView> swapChainImageViews,
                                           VkSampler sampler) {

    // Create a copy of the layout for each descriptor set
    std::vector<VkDescriptorSetLayout> layouts = std::vector { swapChainImageViews.size(), descriptorSetLayout };

    // We first allocate all of the descriptor sets
    VkDescriptorSetAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = swapChainImageViews.size();
    allocInfo.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> descriptorSets;
    descriptorSets.resize(swapChainImageViews.size());

    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    // And then we populate the descriptor sets (one for each swapchain, pointing to the right one)
    for (int i = 0; i < swapChainImageViews.size(); i++) {
        VkDescriptorBufferInfo uniformBufferInfo {};
        uniformBufferInfo.buffer = uniformBuffer;
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo shaderBufferInfo {};
        shaderBufferInfo.buffer = shaderBuffer;
        shaderBufferInfo.offset = 0;
        shaderBufferInfo.range = VK_WHOLE_SIZE;

        VkDescriptorImageInfo accumulatorImageInfo {};
        accumulatorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // todo: find a more optimal one
        accumulatorImageInfo.imageView = accumulatorImageView;
        accumulatorImageInfo.sampler = sampler;

        VkDescriptorImageInfo swapchainImageInfo {};
        swapchainImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // todo: find a more optimal one
        swapchainImageInfo.imageView = swapChainImageViews[i];
        swapchainImageInfo.sampler = sampler;
        
        VkWriteDescriptorSet descriptorWrites[4] = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &shaderBufferInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &accumulatorImageInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pImageInfo = &swapchainImageInfo;

        vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, nullptr);
    }

    return descriptorSets;
}


VkPipeline createGraphicsPipeline(VkDevice device,
                                  VkDescriptorSetLayout descriptorSetLayout,
                                  VkRenderPass renderPass, 
                                  VkPipelineLayout& pipelineLayout) {

    /* SHADERS */

    auto vertexShaderSrc = readFile("build/shaders/main.vert.spv");
    auto fragmentShaderSrc = readFile("build/shaders/main.frag.spv");

    // these are just a thin wrapper around the SPIR-V bytecode
    VkShaderModule vertexShader = createShaderModule(device, vertexShaderSrc);
    VkShaderModule fraglemtnShader = createShaderModule(device, fragmentShaderSrc);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertexShader;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fraglemtnShader;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    

    // begin fixed function setup

    /* VERTEX INPUT AND ASSEMBLY */

    // our vertex input format
    VkPipelineVertexInputStateCreateInfo vertexInputInfo {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;


    VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE; // dont allow breaking strip with special flag


    /* VIEWPORTS AND SCISSORS */

    // Viewport defines how the normalized device coordinates are transformed into pixel
    //  coordinates of the framebuffer -> transformation / mapping
    //  eg. setting to half the screen will squish the swapchain image to half the framebuffer

    // Scissor is simply the area we can render in the frame buffer.
    //  eg. setting it to half the screen will "crop" the output (discard)

    // We are making these dynamic to render operations like resizing windows easier
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;


    /* RASTERIZER AND MULTISAMPLING */

    VkPipelineRasterizationStateCreateInfo rasterizer {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // the "disable rasterization" option
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // fill or wireframe or points
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // backface culling
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // order in which triangles should be drawn
    rasterizer.depthBiasEnable = VK_FALSE; // no biasing to depth values (used to fix zfighting)

    VkPipelineMultisampleStateCreateInfo multisampling {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE; // no
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


    /* COLOR BLENDING */

    // describes generic blending operations for one framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    // wraps around "all of our blending operations" for all framebuffers
    VkPipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE; // whether we just want to blend our buffer using bitwise operations
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // end fixed function setup


    /* PIPELINE LAYOUT (uniforms, etc) */

    // stores the layout of all the shader constants
    VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; 
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }  

    VkGraphicsPipelineCreateInfo pipelineInfo {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // shaders
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    // fixed-fn
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    // uniforms and stuff
    pipelineInfo.layout = pipelineLayout; // vulkan handle and not struct; todo: why
    // render passes: this pipeline will be used for this render pass at this subpass
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    // we dont derive from other pipelines (can be less expensive)
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    printf("log: Created pipeline!\n");

    // we need to delete the thin wrappers...;
    vkDestroyShaderModule(device, vertexShader, nullptr);
    vkDestroyShaderModule(device, fraglemtnShader, nullptr);

    return graphicsPipeline;
}

VkPipeline createComputePipeline(VkDevice device,
                                 VkDescriptorSetLayout descriptorSetLayout,
                                 VkPipelineLayout& pipelineLayout) {

    VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; 
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline layout!");
    }

    auto computeShaderSrc = readFile("build/shaders/main.comp.spv");

    VkShaderModule computeShader = createShaderModule(device, computeShaderSrc);

    VkPipelineShaderStageCreateInfo compShaderStageInfo {};
    compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compShaderStageInfo.module = computeShader;
    compShaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = compShaderStageInfo;

    VkPipeline computePipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline!");
    }

    vkDestroyShaderModule(device, computeShader, nullptr);

    return computePipeline;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shader;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shader) != VK_SUCCESS) {
        std::runtime_error("falied to create shader");
    }

    return shader;
}

VkDescriptorPool createDescriptorPool(VkDevice device, int maxSets) {
    VkDescriptorPoolSize poolSize[3];
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = maxSets;
    poolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize[1].descriptorCount = maxSets;
    poolSize[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize[2].descriptorCount = 2 * maxSets;

    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSize;
    poolInfo.maxSets = maxSets; // max number of allocated descriptor sets

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    return descriptorPool;
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    // we need to create a command buffer to submit a command to do this
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    VkBufferCopy copyRegion {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo {};

    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // only used by graphics queue (unlike potentially the swapchain imgs)

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    // allocate the needed memory
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    // bind it to our buffer
    vkBindBufferMemory(device, buffer, bufferMemory, 0); // we can set an offset
}

void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    // barriers are an easy way to transition image layouts
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // we aren't transferring ownership
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    // the image is not being accessed until the barrier is done, so we don't need to specify the stages
    barrier.srcAccessMask = VK_ACCESS_NONE;
    barrier.dstAccessMask = VK_ACCESS_NONE; 

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // doesn't matter as we manually sync this
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t suitableMemoryTypes, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (suitableMemoryTypes & (1 << i) // it has to be a suitable memory type (memoryTypeBits indicate the supported indices of memory types of physical device)
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) { // it has to match all of the required properties
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

VkCommandPool createCommandPool(VkDevice device, VkPhysicalDevice physicalDevice, QueueFamilyIndices queueFamilyIndices) {
    VkCommandPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // command buffers can be rerecorded individually
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // assumes that all the queues are in the same family

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
    printf("log: Created command pool\n");

    return commandPool;
}

// todo: this is suboptimal: we could reuse the old swapchain when recreating which allows to do it in-flight
//void recreateSwapChain(Window window, 
//                       VkPhysicalDevice physicalDevice, 
//                       VkDevice device, 
//                       VkSurfaceKHR surface, 
//                       VkSwapchainKHR swapChain, 
//                       VkRenderPass renderPass,
//                       std::vector<VkFramebuffer> swapChainFramebuffers, 
//                       std::vector<VkImageView> &swapChainImageViews,
//                       std::vector<VkImage> &swapChainImages, 
//                       VkFormat &swapChainImageFormat, 
//                       VkExtent2D &swapChainExtent) {

//    // we could be minimized, in which case do nothing until we no longer are
//    int width = 0, height = 0;
//    window.getFramebufferSize(&width, &height);
//    while (width == 0 || height == 0) {
//        window.waitEvents();
//        window.getFramebufferSize(&width, &height);// should these be flipped?
//    }

//    vkDeviceWaitIdle(device); // this is bad

//    cleanupSwapChain(device, swapChain, swapChainFramebuffers, swapChainImageViews);

//    createSwapChain(window, physicalDevice, device, surface, swapChainImages, swapChainImageFormat, swapChainExtent);
//    createImageViews(device, swapChainImages, swapChainImageFormat);
//    createFramebuffers(device, renderPass, swapChainExtent, swapChainImageViews);
//}



void cleanupSwapChain(VkDevice device, VkSwapchainKHR swapChain, std::vector<VkFramebuffer> swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews) {
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

// todo: hopefully replace most of this by RAII / vulkan.hpp
void cleanup(VkInstance instance, 
             VkPhysicalDevice physicalDevice, 
             VkDevice device, 
             VkSurfaceKHR surface,
             std::vector<VkSemaphore> imageAvailableSemaphores,
             std::vector<VkSemaphore> renderFinishedSemaphores,
             std::vector<VkFence> inFlightFences,
             std::vector<VkBuffer> uniformBuffers,
             std::vector<VkDeviceMemory> uniformBuffersMemory,
             VkBuffer shaderBuffer,
             VkDeviceMemory shaderBufferMemory,
             VkCommandPool commandPool,
             VkDescriptorPool descriptorPool,
             VkDescriptorSetLayout descriptorSetLayout,
             VkPipeline graphicsPipeline,
             VkPipelineLayout pipelineLayout,
             VkRenderPass renderPass,
             VkDebugUtilsMessengerEXT debugMessenger,
             int framesInFlight) {

    for (size_t i = 0; i < framesInFlight; i++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    for (size_t i = 0; i < framesInFlight; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    vkDestroyBuffer(device, shaderBuffer, nullptr);
    vkFreeMemory(device, shaderBufferMemory, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    //cleanupSwapChain(device, swapChain, framebuffers, imageViews);

    vkDestroyDescriptorPool(device,  descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}
