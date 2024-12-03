#include "vulkan.hpp"

/* Vulkan parameters
 *
 * USE_LLVMPIPE forces the rendering to happen on the CPU
 * VALIDATION_LAYERS_ENABLE enables vulkan validation layers 
 */
const bool USE_LLVMPIPE = false;
const bool VALIDATION_LAYERS_ENABLE = true;

// add "VK_LAYER_LUNARG_api_dump" to dump Vulkan calls in stdout
const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};  // "VK_LAYER_PRINTF_ONLY_PRESET"
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

/* 
 * Retrieves the pointer to the debug messenger creation function.
 *
 * Such function has to be retrieved because it's a vulkan extension so it's not
 * included by default in the header.
 */
VkResult createDebugUtilsMessengerEXT(VkInstance instance, 
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

/* 
 * Retrieves pointer to debug messenger destructor function. 
 */
void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

/* 
 * Creates a Vulkan instance 
 */
VkInstance createInstance() {
    // info about program
    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan RayTracer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // info about "vulkan context"
    VkInstanceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto requiredExts = Window::getRequiredExtensions();

    // validation layers
    if (VALIDATION_LAYERS_ENABLE) {
        if (!allValidationLayersSupported(validationLayers)) {
            throw new std::runtime_error("Some validation layers needed aren't supported");
        }

        requiredExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExts.size());
    createInfo.ppEnabledExtensionNames = requiredExts.data();

    // create actual instance
    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }

    printf("log: Made instance wtih %d extensions!\n", (int)requiredExts.size());

    return instance;
}

/* 
 * Checks if all given validation layers are supported by the current instance 
 */
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

/* 
 * Creates a debug messenger and configures it to show detailed errors 
 */
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance) {
    if (!VALIDATION_LAYERS_ENABLE) return nullptr;

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
    if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }

    printf("log: Enabled all validation layers. NOTE: Instance creation is not covered!\n");

    return debugMessenger;
}

/* 
 * This function serves as a function pointer to be called by the debug messenger
 * whenver a log needs to be printed.
 *
 * Only prints something if the message is of WARNING priority or above.
 */
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                             VkDebugUtilsMessageTypeFlagsEXT messageType,
                                             const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                             void* pUserData) {
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << std::endl << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

/* 
 * Picks the best physical device to present to the created surface.
 * Depends on the surface because a specific device might not be able to present on one.
 *
 * Which device gets picked depends on a score assigned to it by getDeviceScore().
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

    for (const auto& device : devices) {
        if (!isDeviceSuitable(device, surface)) continue;

        int score = getDeviceScore(device, surface);

        if (score > bestScore) {
            bestDevice = device;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    // get device name
    char bestName[256];
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(bestDevice, &deviceProperties);
    memcpy(bestName, deviceProperties.deviceName, 255);

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
 * Checks if a physical device supports all the given extensions.
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

/*
 * Checks what capabilities and formats are supported by the device's SwapChain.
 */
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

/*
 * Gives a score to a physical device for rendering to a certain surface.
 * 
 * If LLVMPIPE is needed, then its corresponding "device" gets maximum score.
 * Else, discrete graphics cards get a higher score, followed by the various
 * queues being in the same family.
 */
int getDeviceScore(VkPhysicalDevice device, VkSurfaceKHR surface) {
    // discrete GPUs get the absolute priority
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    bool isDiscrete = deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;

    // devices whose queues are in the same family are better
    QueueFamilyIndices indices = findQueueFamilies(device, surface);
    bool queuesInSameFamily = !indices.areDifferent();

    // if we need LLVMpipe then it gets the highest priority
    bool isLlvmpipe = !strcmp(deviceProperties.deviceName, "llvmpipe (LLVM 19.1.0, 256 bits)");
    int needLlvmpipe = USE_LLVMPIPE ? isLlvmpipe : -isLlvmpipe; // if we use llvmpipe then it has positive, else negative value

    int score = needLlvmpipe * 100 + isDiscrete * 10 + queuesInSameFamily * 1;

    return score;
}

/*
 * Finds a device's queue family indices, including whether the device supports
 * presenting to the given surface.
 */
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = {0};

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int family = 0;
    for (const auto& queueFamily : queueFamilies) {
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
        if (indices.computeFamily.has_value() && indices.presentFamily.has_value()) {
            break;
        }
    }

    return indices;
}

/*
 * Creates a logical device from the given physical device and chosen queue 
 * family indices. This device represents the interface between our program and
 * the GPU.
 */
VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, QueueFamilyIndices queueIndices) {
    // create a queueCreateInfo for each of the required ones
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    std::set<uint32_t> uniqueQueueFamilies = {
        queueIndices.presentFamily.value(),
        queueIndices.computeFamily.value()
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
    if (VALIDATION_LAYERS_ENABLE) {
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

/*
 * Creates a SwapChain of images used to present frames to the screen.
 *
 * Returns the swapchain handle, its images, their formats, and their extents.
 * The extents represent the dimentions of the images.
 */
VkSwapchainKHR createSwapChain(Window *window,
                               VkPhysicalDevice physicalDevice,
                               VkDevice device,
                               VkSurfaceKHR surface,
                               QueueFamilyIndices queueFamilies,
                               std::vector<VkImage>& swapChainImages, 
                               VkFormat& swapChainImageFormat, 
                               VkExtent2D& swapChainExtent) {

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
    createInfo.clipped = VK_FALSE; // we want the whole image to be rendered

    // if an image owned by single queue family, no need to check for implicit transfers: best performance
    //  might be interesting starting point to look into gpu archtectures
    if (queueFamilies.areDifferent()) {
        uint32_t queueFamilyIndices[] = {queueFamilies.presentFamily.value(), queueFamilies.computeFamily.value()};

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

/*
 * Returns the image/color format to be used by the SwapChain images.
 */
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

/*
 * Returns the extent (dimentions) that the swapchain images will have.
 *
 * This function makes sure that the created images won't be larger than what the
 * GPU can support.
 */
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

/*
 * Creates the Image Views for given swapchain images to be able to read and 
 * write to them.
 */
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

/*
 * Creates an Image of a given size, allocates its memory, and creates a 
 * VkImageView to be able to read and write said image.
 * 
 * Images created by this can be used both for storage and for sampling.
 */
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

    // todo: some images are just one or the other
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

/*
 * Creates a buffer ready to be used as Uniform Buffer.
 */
VkBuffer createUniformBuffer(VkPhysicalDevice physicalDevice, 
                                VkDevice device, 
                                VkDeviceSize size,
                                VkDeviceMemory &uniformBufferMemory,
                                void* &uniformBuffersMap) {

    VkBuffer uniformBuffer = createBuffer(physicalDevice, 
                                          device,
                                          size, 
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                                          uniformBufferMemory);

    vkMapMemory(device, uniformBufferMemory, 0, size, 0, &uniformBuffersMap);

    return uniformBuffer;
}

/*
 * Creates a buffer ready to be used as a Shader Buffer Object.
 */
VkBuffer createShaderBuffer(VkPhysicalDevice physicalDevice, 
                            VkDevice device, 
                            VkDeviceSize size,
                            VkDeviceMemory& shaderBufferMemory, 
                            void*& shaderBufferMapped) {

    VkBuffer shaderBuffer = createBuffer(physicalDevice, 
                                         device, 
                                         size, 
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         shaderBufferMemory);
                         
    vkMapMemory(device, shaderBufferMemory, 0, VK_WHOLE_SIZE, 0, &shaderBufferMapped);

    memset(shaderBufferMapped, 0, size);

    return shaderBuffer;
}

/*
 * Creates a basic 2D image sampler with nearest neighbour sampling, no mipmaps,
 * and clamping the coordinates withing the image's size.
 */
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

/*
 * Allocates a command buffer for a specific command pool to store all the
 * to-be-submitted commands.
 */
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

/*
 * Creates the layout that the descriptor sets will have on the shader, describing
 * how and where each of them will be bound.
 */
VkDescriptorSetLayout createComputeDescriptorSetLayout(VkDevice device) {
    // This uniform will contain the camera data
    VkDescriptorSetLayoutBinding uboLayoutBinding {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // only for compute

    // This buffer will contain the sphere data
    VkDescriptorSetLayoutBinding sphereLayoutBinding {};
    sphereLayoutBinding.binding = 1;
    sphereLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sphereLayoutBinding.descriptorCount = 1;
    sphereLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // This buffer will contain the triangle data
    VkDescriptorSetLayoutBinding triangleLayoutBinding {};
    triangleLayoutBinding.binding = 2;
    triangleLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleLayoutBinding.descriptorCount = 1;
    triangleLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // This is the image where the ray traced frames will be accumulated
    VkDescriptorSetLayoutBinding accumulatorLayoutBinding {};
    accumulatorLayoutBinding.binding = 3;
    accumulatorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    accumulatorLayoutBinding.descriptorCount = 1;
    accumulatorLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // This is the swapchain image that will be bound
    VkDescriptorSetLayoutBinding swapchainLayoutBinding {};
    swapchainLayoutBinding.binding = 4;
    swapchainLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; // todo: maybe storage_texel instead?
    swapchainLayoutBinding.descriptorCount = 1;
    swapchainLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        uboLayoutBinding, sphereLayoutBinding, triangleLayoutBinding, accumulatorLayoutBinding, swapchainLayoutBinding
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

/*
 * Creates the descriptor sets required for the compute shader to execute.
 *
 * There are 5 descriptors for each of the swapchain images:
 *  - Uniform Buffer: Stores and binds the uniform data to the shader
 *  - Mesh Info Buffer: Shader Buffer Object storing information about the meshes to be rendered.
 *  - Triangle Buffer: Shader Buffer Object storing triangles or sphere coordinates.
 *  - Accumulator Image: Image used to accumulate ray traced frames.
 *  - SwapChain Image: Image acquired from the SwapChain to be able to output those frames
 *                     directly from the compute shader.
 * 
 * Note that Mesh Info and Triangle descriptors point to the same buffer in the GPU memory,
 * just with an offset that gets decided at runtime depending on the mesh data.
 */
std::vector<VkDescriptorSet> createComputeDescriptorSets(VkDevice device, 
                                                         VkDescriptorSetLayout descriptorSetLayout, 
                                                         VkDescriptorPool descriptorPool,
                                                         VkBuffer uniformBuffer,
                                                         VkBuffer shaderBuffer,
                                                         uint offset,
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

        VkDescriptorBufferInfo meshInfoBufferInfo {};
        meshInfoBufferInfo.buffer = shaderBuffer;
        meshInfoBufferInfo.offset = 0;
        meshInfoBufferInfo.range = offset;

        VkDescriptorBufferInfo triangleBufferInfo {};
        triangleBufferInfo.buffer = shaderBuffer;
        triangleBufferInfo.offset = offset;
        triangleBufferInfo.range = VK_WHOLE_SIZE;

        VkDescriptorImageInfo accumulatorImageInfo {};
        accumulatorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // todo: find a more optimal one
        accumulatorImageInfo.imageView = accumulatorImageView;
        accumulatorImageInfo.sampler = sampler;

        VkDescriptorImageInfo swapchainImageInfo {};
        swapchainImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        swapchainImageInfo.imageView = swapChainImageViews[i];
        swapchainImageInfo.sampler = sampler;
        
        VkWriteDescriptorSet descriptorWrites[5] = {};

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
        descriptorWrites[1].pBufferInfo = &meshInfoBufferInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &triangleBufferInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pImageInfo = &accumulatorImageInfo;

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = descriptorSets[i];
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &swapchainImageInfo;

        vkUpdateDescriptorSets(device, 5, descriptorWrites, 0, nullptr);
    }

    return descriptorSets;
}

/*
 * Creates a basic compute pipeline and returns it and its layout.
 *
 * The pipeline entrypoint is a shader called "main.comp" and located in 
 * build/shaders/main.comp.spv
 */
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

/*
 * Creates a vulkan shader module for a device from given spir-v code.
 */
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

/*
 * Creates a descriptor pool ready for all the required descriptor sets.
 */
VkDescriptorPool createDescriptorPool(VkDevice device, int maxSets) {
    VkDescriptorPoolSize poolSize[3];
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = maxSets;
    poolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize[1].descriptorCount = 2 * maxSets;
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

/*
 * Helper function to begin a single-time command to the GPU.
 *
 * Used for setting up images or buffers.
 */
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

/*
 * Helper function to end and issue a single-time command to the GPU.
 *
 * Used for setting up images or buffers.
 */
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

/*
 * Issues a command to copy a buffer from a source to a destination.
 *
 * Should only be used while setting up the vulkan context and not while drawing
 * frames.
 */
void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    // we need to create a command buffer to submit a command to do this
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    VkBufferCopy copyRegion {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

/*
 * Creates a buffer of given size, properties, and adequate to a given set of
 * usages.
 * 
 * Returns the buffer and its bound VkDeviceMemory object.
 */
VkBuffer createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags properties, VkDeviceMemory& bufferMemory) {
    VkBuffer buffer;
    
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

    return buffer;
}

/*
 * Issues a command to initially transition an image to a required
 * layout.
 * 
 * Should only be used while setting up the vulkan context and not while drawing
 * frames.
 */
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

/*
 * Finds a suitable memory type amongst physicalDevice's supported ones given a 
 * list of properties it should have. Used to create efficient buffers.
 */
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

/*
 * Creates command pools for the various queues that we will submit to.
 *
 * The array contains first the compute command pool, and next the presentation one.
 * If the indices are the same, then the array will contain 2 copies of the same VkCommandPool.
 */
VkCommandPool createCommandPool(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex) {
    VkCommandPool commandPool;

    VkCommandPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // command buffers can be rerecorded individually
    poolInfo.queueFamilyIndex = queueFamilyIndex;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
    printf("log: Created command pool\n");

    return commandPool;
}

/*
 * Function to cleanup a SwapChain
 */
void cleanupSwapChain(VkDevice device, VkSwapchainKHR swapChain, std::vector<VkFramebuffer> swapChainFramebuffers, std::vector<VkImageView> swapChainImageViews) {
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

/*
 * Function to clean up the entire vulkan state. Outdated for now, and hopefully
 * to be replaced with RAII implementations present in vulkan.hpp.
 */
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

// todo: hopefully replace most of this by RAII / vulkan.hpp
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

    if (VALIDATION_LAYERS_ENABLE) {
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}
