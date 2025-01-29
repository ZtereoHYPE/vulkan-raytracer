#include "init.hpp"

using namespace vk;

const std::vector deviceExtensions = params.HEADLESS 
    ? std::vector<const char *>{} 
    : std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

/**
 * Pointers retrieved and used by vkCreateDebugUtilsMessengerEXT and vkDestroyDebugUtilsMessengerEXT.
 */
PFN_vkCreateDebugUtilsMessengerEXT  createDebugUtilsMessengerPointer;
PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessengerPointer;

/* 
 * Retrieves the pointers to the debug messenger creation and destruction functions.
 *
 * Such functions have to be retrieved because they are a vulkan extension so they aren't
 * included by default in the header.
 */
void retrieveDebugMessengerPointers(Instance const &instance) {
    createDebugUtilsMessengerPointer = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>( instance.getProcAddr( "vkCreateDebugUtilsMessengerEXT" ) );
    destroyDebugUtilsMessengerPointer = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>( instance.getProcAddr( "vkCreateDebugUtilsMessengerEXT" ) );
}

/**
 * Creates a debug messenger.
 *
 * Since this function is retrieved dynamically at runtime, it is not defined in vulkan.h(pp). This means that
 * unless it is defined here, linking will fail.
 *
 * This implementation just ends up calling the retrieved pointer.
 */
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                                                              const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                                              const VkAllocationCallbacks *pAllocator,
                                                              VkDebugUtilsMessengerEXT *pMessenger) {
    return createDebugUtilsMessengerPointer(instance, pCreateInfo, pAllocator, pMessenger);
}

/**
 * Destroyes a debug messenger.
 *
 * See: above.
 */
VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
                                                           VkDebugUtilsMessengerEXT messenger,
                                                           VkAllocationCallbacks const * pAllocator) {
    return destroyDebugUtilsMessengerPointer(instance, messenger, pAllocator);
}

/* 
 * Creates a Vulkan instance 
 */
Instance createInstance() {
    // check version
    uint32_t vkVersion = enumerateInstanceVersion();
    if (apiVersionMinor(vkVersion) != 3) {
        std::cout << "Warning: this program was tested on vulkan 1.3" << std::endl;
    }

    auto requiredExtensions = Window::getRequiredExtensions();
    std::vector<char const *> layers;

    // validation layers
    if (params.USE_VALIDATION_LAYERS) {
        if (!allValidationLayersSupported(params.VALIDATION_LAYERS)) {
            throw std::runtime_error("Some validation layers needed aren't supported");
        }

        layers = params.VALIDATION_LAYERS;
        requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    ApplicationInfo appInfo {
        .sType = StructureType::eApplicationInfo,
        .pApplicationName = "Vulkan RayTracer",
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = ApiVersion13,
    };

    // info about the instance
    InstanceCreateInfo createInfo {
        .sType = StructureType::eInstanceCreateInfo,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data(),
    };

    printf("log: Made instance wtih %ld extensions!\n", requiredExtensions.size());

    Instance instance = vk::createInstance(createInfo);

    // retrieve the function pointers for debug messenger
    retrieveDebugMessengerPointers(instance);

    return instance;
}

/* 
 * Checks if all given validation layers are supported by the current instance 
 */
bool allValidationLayersSupported(const std::vector<const char*>& validationLayers) {
    auto availableLayers = enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            printf("log: could not find %s\n", layerName);
            return false;
        }
    }

    return true;
}

/* 
 * Creates a debug messenger and configures it to show detailed errors 
 */
DebugUtilsMessengerEXT setupDebugMessenger(Instance const &instance) {
    if (!params.USE_VALIDATION_LAYERS) return nullptr;

    using severity = DebugUtilsMessageSeverityFlagBitsEXT;
    using type = DebugUtilsMessageTypeFlagBitsEXT;

    DebugUtilsMessengerCreateInfoEXT createInfo {
        .sType = StructureType::eDebugUtilsMessengerCreateInfoEXT,
        .messageSeverity = severity::eVerbose | severity::eWarning | severity::eError,
        .messageType = type::eGeneral | type::eValidation | type::ePerformance,
        .pfnUserCallback = debugCallback,
        .pUserData = nullptr,
    };

    printf("log: Enabled all validation layers. NOTE: Instance creation is not covered!\n");
    return instance.createDebugUtilsMessengerEXT(createInfo);
}

/* 
 * This function serves as a function pointer to be called by the debug messenger
 * whenver a log needs to be printed.
 *
 * Only prints something if the message is of WARNING priority or above.
 */
VKAPI_ATTR Bool32 VKAPI_CALL debugCallback(DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                             Flags<DebugUtilsMessageTypeFlagBitsEXT> messageFlags,
                                             const DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                             void* pUserData) {
    if (messageSeverity >= DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        std::cerr << std::endl << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

/* 
 * Picks the best physical device to present to the created surface.
 * Depends on the surface because a specific device might not be able to present on one.
 *
 * Which device gets picked depends on a score assigned to it by getDeviceScore().
 */
PhysicalDevice pickPhysicalDevice(Instance const &instance, SurfaceKHR const &surface) {
    auto physicalDevices = instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    int bestScore = 0;
    PhysicalDevice const *bestDevice = VK_NULL_HANDLE;

    for (const auto& device : physicalDevices) {
        if (!isDeviceSuitable(device, surface)) continue;

        if (const int score = getDeviceScore(device, surface); score > bestScore) {
            bestDevice = &device;
            bestScore = score;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    // get device name
    auto properties = bestDevice->getProperties();

    printf("log: picked device %s\n", properties.deviceName.data());

    return *bestDevice;
}

/**
 * Headless version of pickPhysicalDevice that doesn't need to concern itself
 * with the surface the device needs to present to.
 */
PhysicalDevice pickHeadlessPhysicalDevice(Instance const &instance) {
    auto physicalDevices = instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    int bestScore = 0;
    PhysicalDevice const *bestDevice = VK_NULL_HANDLE;

    for (const auto& device : physicalDevices) {
        QueueFamilyIndices familyIndexes = findHeadlessQueueFamilies(device);
    
        if (!familyIndexes.computeFamily.has_value()) continue;

        if (const int score = getHeadlessDeviceScore(device); score > bestScore) {
            bestDevice = &device;
            bestScore = score;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU (headless)!");
    }

    // get device name
    auto properties = bestDevice->getProperties();

    printf("log: picked headless device %s\n", properties.deviceName.data());

    return *bestDevice;
}

/* 
 * Checks if a device is suitable for rendering on a surface.
 *
 * Suitable in this case means that it has a queue families that support all of the required
 * features, and that it supports creating a swapchain on the surface.
 */
bool isDeviceSuitable(PhysicalDevice const &device, SurfaceKHR const &surface) {
    bool extensionsSupported = checkDeviceExtensionSupport(device, deviceExtensions);
    bool swapChainAdequate = false;

    // only query swapchain details after we've made sure the extension is supported
    if (extensionsSupported) {
        SwapChainSupportDetails details = querySwapChainSupport(device, surface);
        swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
    }

    QueueFamilyIndices familyIndexes = findQueueFamilies(device, surface);

    return familyIndexes.isComplete() && swapChainAdequate;
}

/* 
 * Checks if a physical device supports all the given extensions.
 */
bool checkDeviceExtensionSupport(PhysicalDevice const &device, std::vector<const char*> deviceExtensions) {
    auto availableExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    // remove from requiredExtensions the extensions, and check if empty at the end.
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

/*
 * Returns what capabilities and formats are supported by the device's SwapChain.
 */
SwapChainSupportDetails querySwapChainSupport(PhysicalDevice const &device, SurfaceKHR const &surface) {
    return SwapChainSupportDetails {
        .capabilities = device.getSurfaceCapabilitiesKHR(surface),
        .formats = device.getSurfaceFormatsKHR(surface),
        .presentModes = device.getSurfacePresentModesKHR(surface)
    };
}

/*
 * Gives a score to a physical device for rendering to a certain surface.
 * 
 * If LLVMPIPE is needed, then its corresponding "device" gets maximum score.
 * Else, discrete graphics cards get a higher score, followed by the various
 * queues being in the same family.
 */
int getDeviceScore(PhysicalDevice const &device, SurfaceKHR const &surface) {
    // discrete GPUs get the absolute priority
    const auto properties = device.getProperties();
    const bool isDiscrete = properties.deviceType == PhysicalDeviceType::eDiscreteGpu;

    // devices whose queues are in the same family are better
    const QueueFamilyIndices indices = findQueueFamilies(device, surface);
    const bool queuesInSameFamily = !indices.areDifferent();

    // if we need to use llvmpipe then finding it has positive value, else negative value
    const bool isLlvmpipe = std::string(properties.deviceName).find("llvmpipe") != std::string::npos;
    const int llvmpipeScore = params.USE_LLVMPIPE ? 100 : -100;

    const int score = isLlvmpipe * llvmpipeScore + isDiscrete * 10 + queuesInSameFamily * 1;

    return score;
}

/*
 * Gives a score to a physical device for headless rendering.
 * 
 * If LLVMPIPE is needed, then its corresponding "device" gets maximum score.
 * Else, discrete graphics cards get a higher score, followed by the various
 * queues being in the same family.
 */
int getHeadlessDeviceScore(PhysicalDevice const &device) {
    // discrete GPUs get the absolute priority
    const auto properties = device.getProperties();
    const bool isDiscrete = properties.deviceType == PhysicalDeviceType::eDiscreteGpu;

    // if we need to use llvmpipe then finding it has positive value, else negative value
    const bool isLlvmpipe = std::string(properties.deviceName).find("llvmpipe") != std::string::npos;
    const int llvmpipeScore = params.USE_LLVMPIPE ? 100 : -100;

    const int score = isLlvmpipe * llvmpipeScore + isDiscrete;

    return score;
}

/*
 * Finds a device's queue family indices, including whether the device supports
 * presenting to the given surface.
 */
QueueFamilyIndices findQueueFamilies(PhysicalDevice const &device, SurfaceKHR const &surface) {
    QueueFamilyIndices indices{};

    auto families = device.getQueueFamilyProperties();

    int currentFamily = 0;
    for (const auto& family : families) {
        if (family.queueFlags & QueueFlagBits::eCompute) {
            indices.computeFamily = currentFamily;
        }

        // if the device supports presenting to that surface, add it to the present family
        if (device.getSurfaceSupportKHR(currentFamily, surface)) {
            indices.presentFamily = currentFamily;
        }

        currentFamily++;

        // we stop once found all the needed queue families
        if (indices.computeFamily.has_value() && indices.presentFamily.has_value()) {
            break;
        }
    }

    return indices;
}

/**
 * Headless version of findQueueFamilies, doesn't need to concern itself with
 * the surface being rendered to.
 */
QueueFamilyIndices findHeadlessQueueFamilies(PhysicalDevice const &device) {
    QueueFamilyIndices indices{};

    auto families = device.getQueueFamilyProperties();

    int currentFamily = 0;
    for (const auto& family : families) {
        if (family.queueFlags & QueueFlagBits::eCompute) {
            indices.computeFamily = currentFamily;
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
Device createLogicalDevice(PhysicalDevice const &physicalDevice, QueueFamilyIndices const &queueIndices) {
    // create a queueCreateInfo for each of the required ones
    std::vector<DeviceQueueCreateInfo> queueCreateInfos;

    std::set uniqueQueueFamilies = {
        queueIndices.computeFamily.value()
    };

    // optional present family for offscreen rendering
    if (queueIndices.presentFamily.has_value())
        uniqueQueueFamilies.insert(queueIndices.presentFamily.value());

    float queuePriority = 1.0f;
    for (uint32_t family : uniqueQueueFamilies) {
        queueCreateInfos.push_back({
            .sType = StructureType::eDeviceQueueCreateInfo,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        });
    }

    DeviceCreateInfo createInfo {
        .sType = StructureType::eDeviceCreateInfo,
        .queueCreateInfoCount = static_cast<uint32_t>(uniqueQueueFamilies.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(params.VALIDATION_LAYERS.size()),
        .ppEnabledLayerNames = params.VALIDATION_LAYERS.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = {},
    };

    printf("log: Created logical device.\n");

    return physicalDevice.createDevice(createInfo);
}

/*
 * Creates a SwapChain of images used to present frames to the screen.
 *
 * Returns the swapchain handle, its images, their formats, and their extents.
 * The extents represent the dimentions of the images.
 */
SwapchainKHR createSwapChain(Window &window,
                             PhysicalDevice const &physicalDevice,
                             Device const &device,
                             SurfaceKHR const &surface,
                             QueueFamilyIndices queueFamilies,
                             std::vector<Image>& swapChainImages,
                             Format& swapChainImageFormat,
                             Extent2D& swapChainExtent) {

    SwapChainSupportDetails details = querySwapChainSupport(physicalDevice, surface);

    if (!(details.capabilities.supportedUsageFlags & ImageUsageFlagBits::eSampled)
        || !(details.capabilities.supportedUsageFlags & ImageUsageFlagBits::eStorage)) {
        throw std::runtime_error("Unfortunately the current GPU's swapchain images do not support being rendered to from a compute shader");
    }

    // out of the ones available for our current device, pick the "best" modes
    SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceMode(physicalDevice, details.formats);
    swapChainExtent = chooseSwapExtent(window, details.capabilities);

    uint32_t imageCount = details.capabilities.minImageCount + 1; // to allow "internal operations" to complete

    // check if it's not more than we can afford (0 == unlimited)
    if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount) {
        imageCount = details.capabilities.maxImageCount;
    }

    // if an image owned by single queue family, no need to check for implicit transfers: best performance
    //  might be interesting starting point to look into gpu archtectures
    SharingMode sharingMode;
    std::vector<uint32_t> queueFamilyIndices = {};
    if (queueFamilies.areDifferent()) {
        queueFamilyIndices = {queueFamilies.presentFamily.value(), queueFamilies.computeFamily.value()};
        sharingMode = SharingMode::eConcurrent;
        printf("log: Unfortunately our queues are split.\n");
    } else {
        sharingMode = SharingMode::eExclusive;
        printf("log: Nice! Our queues are all in the same family!\n");
    }

    SwapchainCreateInfoKHR createInfo {
        .sType = StructureType::eSwapchainCreateInfoKHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = ImageUsageFlagBits::eStorage | ImageUsageFlagBits::eTransferSrc,
        .imageSharingMode = sharingMode,
        .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size()),
        .pQueueFamilyIndices = queueFamilyIndices.data(),
        .preTransform = SurfaceTransformFlagBitsKHR::eIdentity,
        .compositeAlpha = CompositeAlphaFlagBitsKHR::eOpaque,   // alpha used for blending in compositor
        .presentMode = PresentModeKHR::eMailbox,
        .clipped = True,
        .oldSwapchain = VK_NULL_HANDLE,                      // we don't have an old swapchain to "recycle"
    };

    SwapchainKHR swapChain = device.createSwapchainKHR(createInfo);
    swapChainImages = device.getSwapchainImagesKHR(swapChain);
    swapChainImageFormat = surfaceFormat.format;

    printf("log: Created swapchain.\n");

    return swapChain;
}

/*
 * Returns the image/color format to be used by the SwapChain images.
 */
SurfaceFormatKHR chooseSwapSurfaceMode(PhysicalDevice const &physicalDevice, const std::vector<SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(availableFormat.format);
        if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage) {
            return availableFormat;
        }
    }

    throw std::runtime_error("No supported swapchain format found");
}

/*
 * Returns the extent (dimentions) that the swapchain images will have.
 *
 * This function makes sure that the created images won't be larger than what the
 * GPU can support.
 */
Extent2D chooseSwapExtent(Window &window, SurfaceCapabilitiesKHR const &capabilities) {
    // 0xFFFFFFFF -> "the surface size/ will be determined by the extent of a swapchain targeting it"
    if (capabilities.currentExtent.width != -1) {
        return capabilities.currentExtent;
    }

    int width, height;
    window.getFramebufferSize(&width, &height);

    Extent2D extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

/*
 * Creates the Image Views for given swapchain images to be able to read and 
 * write to them.
 */
std::vector<ImageView> createSwapchainViews(Device const &device, std::vector<Image> const &swapChainImages, Format swapChainImageFormat) {
    std::vector<ImageView> swapChainImageViews;

    for (auto const swapChainImage : swapChainImages) {
        ImageViewCreateInfo createInfo {
            .sType = StructureType::eImageViewCreateInfo,
            .image = swapChainImage,
            .viewType = ImageViewType::e2D,
            .format = swapChainImageFormat,
            .subresourceRange = {
                .aspectMask = ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }
        };

        swapChainImageViews.push_back(device.createImageView(createInfo));
    }

    printf("log: Created %lu image views for our swapchain!\n", swapChainImageViews.size());

    return swapChainImageViews;
}

/*
 * Creates an Image of a given size, allocates its memory, and creates a 
 * VkImageView to be able to read and write said image.
 * 
 * Images created by this can be used both for storage and for sampling.
 */
std::tuple<Image, ImageView, DeviceMemory> createImage(PhysicalDevice const &physicalDevice,
                                                       Device const &device,
                                                       Extent2D extent,
                                                       Format format) {
    ImageCreateInfo imageInfo {
        .sType = StructureType::eImageCreateInfo,
        .imageType = ImageType::e2D,
        .format = format,
        .extent = Extent3D(extent.width, extent.height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = SampleCountFlagBits::e1,
        .tiling = ImageTiling::eOptimal,
        .usage = ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eStorage,
        .sharingMode = SharingMode::eExclusive, // only ever accessed by 1 family at a time
        .initialLayout = ImageLayout::eUndefined
    };

    Image image = device.createImage(imageInfo);

    MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

    DeviceMemory imageMemory = device.allocateMemory({
        .sType = StructureType::eMemoryAllocateInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, MemoryPropertyFlagBits::eDeviceLocal),
    });

    device.bindImageMemory(image, imageMemory, 0);

    ImageViewCreateInfo createInfo {
        .sType = StructureType::eImageViewCreateInfo,
        .image = image,
        .viewType = ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    ImageView imageView = device.createImageView(createInfo);

    return std::make_tuple(image, imageView, imageMemory);
}

/*
 * Creates a memory-mapped buffer ready for use.
 */
Buffer createMappedBuffer(PhysicalDevice const &physicalDevice,
                          Device const &device,
                          DeviceSize size,
                          BufferUsageFlags usage,
                          DeviceMemory &memory,
                          void* &memoryMap) {

    Buffer buffer = createBuffer(
        physicalDevice,
        device,
        size,
        usage,
        MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent,
        memory
    );

    memoryMap = device.mapMemory(memory, 0, size);
    memset(memoryMap, 0, size);

    return buffer;
}

/*
 * Creates a basic 2D image sampler with nearest neighbour sampling, no mipmaps,
 * and clamping the coordinates withing the image's size.
 */
Sampler createSampler(Device const &device) {
    return device.createSampler({
        .sType = StructureType::eSamplerCreateInfo,
        .magFilter = Filter::eNearest,
        .minFilter = Filter::eNearest,
        .mipmapMode = SamplerMipmapMode::eNearest,
        .addressModeU = SamplerAddressMode::eClampToBorder,
        .addressModeV = SamplerAddressMode::eClampToBorder,
        .addressModeW = SamplerAddressMode::eClampToBorder,
        .mipLodBias = 0.0f,
        .anisotropyEnable = False,
        .compareEnable = False,
        .compareOp = CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .unnormalizedCoordinates = True,
    });
}

/*
 * Allocates a command buffer for a specific command pool to store all the
 * to-be-submitted commands.
 */
CommandBuffer createCommandBuffer(Device const &device, CommandPool const &commandPool) {
    CommandBufferAllocateInfo allocInfo {
        .sType = StructureType::eCommandBufferAllocateInfo,
        .commandPool = commandPool,
        .commandBufferCount = static_cast<uint32_t>(1)
    };

    printf("log: Created command buffers\n");

    return device.allocateCommandBuffers(allocInfo)[0];
}

/*
 * Creates a descriptor set based on a vector of descriptor types, and the buffer and image
 * informations passed in. Each type is matched with the first info from the corresponding
 * arrays.
 */
DescriptorSet createDescriptorSet(Device const &device,
                                  DescriptorSetLayout const &descriptorSetLayout,
                                  DescriptorPool const &descriptorPool,
                                  std::vector<DescriptorType> types,
                                  DescriptorBufferInfo bufferInfos[],
                                  DescriptorImageInfo imageInfos[]) {

    // We first allocate all of the descriptor sets
    DescriptorSetAllocateInfo allocInfo {
        .sType = StructureType::eDescriptorSetAllocateInfo,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    DescriptorSet descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

    // Populate the descriptor set with the buffers / images it points to
    std::vector<WriteDescriptorSet> descriptorWrites{};
    size_t buffers = 0, images = 0;
    for (uint j = 0; j < types.size(); ++j) {
        WriteDescriptorSet descriptorWrite {
            .sType = StructureType::eWriteDescriptorSet,
            .dstSet = descriptorSet,
            .dstBinding = j,
            .descriptorCount = 1,
            .descriptorType = types[j],
        };

        if (types[j] == DescriptorType::eStorageBuffer || types[j] == DescriptorType::eUniformBuffer) {
            descriptorWrite.pBufferInfo = &bufferInfos[buffers++];
        } else if (types[j] == DescriptorType::eStorageImage) {
            descriptorWrite.pImageInfo = &imageInfos[images++];
        } else throw std::runtime_error("Encountered unknown descriptor!");

        descriptorWrites.push_back(descriptorWrite);
    }

    device.updateDescriptorSets(descriptorWrites, nullptr);

    return descriptorSet;
}

/*
 * Creates a descriptor set layout for a compute shader based on a vector of descriptor types.
 * Each type has exactly one matching descriptor in the layout.
 */
DescriptorSetLayout createDescriptorSetLayout(Device const &device, std::vector<DescriptorType> types) {
    std::vector<DescriptorSetLayoutBinding> bindings{};
    for (uint idx = 0; idx < types.size(); ++idx) {
        bindings.push_back(DescriptorSetLayoutBinding {
            .binding = idx,
            .descriptorType = types[idx],
            .descriptorCount = 1,
            .stageFlags = ShaderStageFlagBits::eCompute,
        });
    }

    DescriptorSetLayoutCreateInfo layoutInfo {
        .sType = StructureType::eDescriptorSetLayoutCreateInfo,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return device.createDescriptorSetLayout(layoutInfo);
}

/*
 * Creates description layout and set for the ray generation compute stage
 */
DescriptorSet createGenerateDescriptorSet(Device const &device,
                                          DescriptorPool const &pool,
                                          Buffer const &uniformBuffer,
                                          Buffer const &rayBuffer,
                                          DescriptorSetLayout &layout) {

    std::vector types{
        DescriptorType::eUniformBuffer, // camera uniform
        DescriptorType::eStorageBuffer, // ray buffer
    };
    DescriptorBufferInfo bufferInfos[] {
        {
            .buffer = uniformBuffer,
            .offset = 0,
            .range = WholeSize,
        },{
            .buffer = rayBuffer,
            .offset = 0,
            .range = WholeSize,
        },
    };

    layout = createDescriptorSetLayout(device, types);
    return createDescriptorSet(device, layout, pool, types, bufferInfos, {});
}

/*
 * Creates description layout and set for the ray-mesh intersection compute stage
 */
DescriptorSet createIntersectDescriptorSet(Device const &device,
                                           DescriptorPool const &pool,
                                           Buffer const &uniformBuffer,
                                           Buffer const &rayBuffer,
                                           Buffer const &hitBuffer,
                                           Buffer const &sceneBuffer,
                                           uint bvhSize,
                                           uint matSize,
                                           DescriptorSetLayout &layout) {

    std::vector types {
        DescriptorType::eUniformBuffer, // camera
        DescriptorType::eStorageBuffer, // bvh
        DescriptorType::eStorageBuffer, // material
        DescriptorType::eStorageBuffer, // triangles
        DescriptorType::eStorageBuffer, // ray buffer
        DescriptorType::eStorageBuffer  // hit records
    };
    DescriptorBufferInfo bufferInfos[] {
        {
            .buffer = uniformBuffer,
            .offset = 0,
            .range = WholeSize
        }, {
            .buffer = sceneBuffer,
            .offset = 0,
            .range = bvhSize,
        },{
            .buffer = sceneBuffer,
            .offset = bvhSize,
            .range = bvhSize + matSize,
        },{
            .buffer = sceneBuffer,
            .offset = bvhSize + matSize,
            .range = WholeSize,
        },{
            .buffer = rayBuffer,
            .offset = 0,
            .range = WholeSize,
        },{
            .buffer = hitBuffer,
            .offset = 0,
            .range = WholeSize,
        }
    };

    layout = createDescriptorSetLayout(device, types);
    return createDescriptorSet(device, layout, pool, types, bufferInfos, {});
}

/*
 * Creates description layout and set for the shading compute stage
 */
DescriptorSet createShadeDescriptorSet(Device const &device,
                         DescriptorPool const &pool,
                         Buffer const &uniformBuffer,
                         Buffer const &rayBuffer,
                         Buffer const &sceneBuffer,
                         Buffer const &hitBuffer,
                         uint bvhSize,
                         uint matSize,
                         Sampler sampler,
                         DescriptorSetLayout &layout) {

    std::vector types {
        DescriptorType::eUniformBuffer, // camera
        DescriptorType::eStorageBuffer, // material
        DescriptorType::eStorageBuffer, // ray buffer
        DescriptorType::eStorageBuffer, // hit records
        // todo: ^ could be samplers? (also below)
    };
    DescriptorBufferInfo bufferInfos[] {
        {
            .buffer = uniformBuffer,
            .offset = 0,
            .range = WholeSize
        }, {
            .buffer = sceneBuffer,
            .offset = bvhSize,
            .range = matSize,
        },{
            .buffer = rayBuffer,
            .offset = 0,
            .range = WholeSize,
        },{
            .buffer = hitBuffer,
            .offset = 0,
            .range = WholeSize,
        }
    };

    layout = createDescriptorSetLayout(device, types);
    return createDescriptorSet(device, layout, pool, types, bufferInfos, {});
}

/*
 * Creates description layout and set for the post processing compute stage.
 * This however lacks one descriptor: the image that the result is rendered to.
 * This image is provided via a different descriptor set.
 */
DescriptorSet createPostProcessDescriptorSet(Device const &device,
                                             DescriptorPool const &pool,
                                             Buffer const &uniformBuffer,
                                             Buffer const &rayBuffer,
                                             DescriptorSetLayout &layout) {

    std::vector<DescriptorSet> sets{};

    std::vector types {
        DescriptorType::eUniformBuffer,
        DescriptorType::eStorageBuffer, // ray buffer
    };
    DescriptorBufferInfo bufferInfos[] {
        {
            .buffer = uniformBuffer,
            .offset = 0,
            .range = WholeSize,
        },{
            .buffer = rayBuffer,
            .offset = 0,
            .range = WholeSize
        }
    };

    // Accumulator descriptor set
    layout = createDescriptorSetLayout(device, types);
    return createDescriptorSet(device, layout, pool, types, bufferInfos, {});
}

/*
 * Creates second description layout and set for the post processing compute stage.
 * This is responsible for attaching the framebuffer to the compute shader.
 */
std::vector<DescriptorSet> createFramebufferDescriptorSets(Device const &device,
                                                           DescriptorPool const &pool,
                                                           std::vector<ImageView> &swapchainViews,
                                                           Sampler sampler,
                                                           DescriptorSetLayout &layout) {

    std::vector types { DescriptorType::eStorageImage };
    layout = createDescriptorSetLayout(device, types);

    std::vector<DescriptorSet> sets{};
    for (auto const view : swapchainViews) {
        DescriptorImageInfo imageInfos[] {{
            .sampler = sampler,
            .imageView = view,
            .imageLayout = ImageLayout::eGeneral, // todo: find more optimal?
        }};

        sets.push_back(createDescriptorSet(device, layout, pool, types, {}, imageInfos));
    }
    return sets;
}

/*
 * Creates a basic compute pipeline and returns it and its layout.
 *
 * The pipeline's shader path is provided as an argument, and has a function
 * called "main" as entrypoint.
 */
Pipeline createComputePipeline(Device const &device,
                               std::vector<DescriptorSetLayout> const &descriptorSetLayouts,
                               std::string const &shaderPath,
                               PipelineLayout &layout) {

    auto shaderSrc = readFile(shaderPath);

    layout = device.createPipelineLayout({
        .sType = StructureType::ePipelineLayoutCreateInfo,
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.data(),
    });

    ComputePipelineCreateInfo info = {
        .sType = StructureType::eComputePipelineCreateInfo,
        .stage = {
            .sType = StructureType::ePipelineShaderStageCreateInfo,
            .stage = ShaderStageFlagBits::eCompute,
            .module = device.createShaderModule({
                .sType = StructureType::eShaderModuleCreateInfo,
                .codeSize = shaderSrc.size(),
                .pCode = reinterpret_cast<const uint32_t*>(shaderSrc.data()),
            }),
            .pName = "main",
        },
        .layout = layout,
    };

    // we don't have a pipeline cache
    auto [_, computePipeline] = device.createComputePipeline(nullptr, info);

    return computePipeline;
}

/*
 * Creates a descriptor pool ready for all the required descriptor sets.
 */
DescriptorPool createDescriptorPool(Device const &device, size_t swapchainSize) {
    DescriptorPoolSize poolSize[3];
    poolSize[0].type = DescriptorType::eUniformBuffer;
    poolSize[0].descriptorCount = 4; // one per shader
    poolSize[1].type = DescriptorType::eStorageBuffer;
    poolSize[1].descriptorCount = 10; // total amount of buffer bindings
    poolSize[2].type = DescriptorType::eStorageImage;
    poolSize[2].descriptorCount = swapchainSize; // (framebuff * swapchainSize)

    return device.createDescriptorPool({
        .sType = StructureType::eDescriptorPoolCreateInfo,
        .flags = DescriptorPoolCreateFlagBits::eFreeDescriptorSet, // to allow descriptor sets to destroy themselves
        .maxSets = static_cast<uint32_t>(4 + swapchainSize), // 4 stages + framebuffers
        .poolSizeCount = 3,
        .pPoolSizes = poolSize,
    });
}

/*
 * Creates a buffer of given size, properties, and adequate to a given set of
 * usages.
 * 
 * Returns the buffer and its bound VkDeviceMemory object.
 */
Buffer createBuffer(PhysicalDevice const &physicalDevice,
                    Device const &device,
                    DeviceSize const size,
                    BufferUsageFlags const usageFlags,
                    MemoryPropertyFlags properties,
                    DeviceMemory &memory) {

    // create a bugger
    Buffer buffer = device.createBuffer({
        .sType = StructureType::eBufferCreateInfo,
        .size = size,
        .usage = usageFlags,
        .sharingMode = SharingMode::eExclusive, // only used by one queue
    });

    // allocate the needed memory
    MemoryRequirements memReqs = device.getBufferMemoryRequirements(buffer);

    memory = device.allocateMemory({
        .sType = StructureType::eMemoryAllocateInfo,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = findMemoryType(physicalDevice, memReqs.memoryTypeBits, properties),
    });

    // bind it to our buffer
    device.bindBufferMemory(buffer, memory, 0);

    return buffer;
}

/*
 * Finds a suitable memory type amongst physicalDevice's supported ones given a 
 * list of properties it should have. Used to create efficient buffers.
 */
uint32_t findMemoryType(PhysicalDevice const &physicalDevice, uint32_t suitableMemoryTypes, MemoryPropertyFlags properties) {
    PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

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
CommandPool createCommandPool(Device const &device, uint32_t queueFamilyIndex) {
    printf("log: Creating command pool\n");

    return device.createCommandPool({
        .sType = StructureType::eCommandPoolCreateInfo,
        .flags = CommandPoolCreateFlagBits::eResetCommandBuffer, // command buffers can be rerecorded individually
        .queueFamilyIndex = queueFamilyIndex,
    });
}
