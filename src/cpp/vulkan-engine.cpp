#include "vulkan-engine.hpp"

/**
 * Main run function that initializes vulkan and starts the loop
 */
void RayTracerProgram::run() {
    initVulkan();
    mainLoop();
}

/**
 * Initialize Vulkan to get it ready to run the ray tracer
 */
void RayTracerProgram::initVulkan() try {
    initEnvironment();
    initBuffers();
    initPipelines();

    // Synchronization structure initialization
    vk::SemaphoreCreateInfo semaphoreInfo { .sType = vk::StructureType::eSemaphoreCreateInfo };
    imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
    computeInFlightFence = device.createFence({
        .sType = vk::StructureType::eFenceCreateInfo,
        .flags = vk::FenceCreateFlagBits::eSignaled // pre-signaled for first frame
    });

    printf("log: initialized vulkan context\n");

} catch (std::runtime_error &) {
    // error handler
    std::cout << "A runtime error occurred! The program will now terminate" << std::endl;
    throw; // rethrow to halt execution
}

/**
 * Initializes the environment of the vulkan program.
 * This includes the instance, the surface to render on, the device picked,
 * the queues of the device, as well as the command pool and buffers.
 */
void RayTracerProgram::initEnvironment() {
    vk::Instance instance = createInstance();
    vk::DebugUtilsMessengerEXT debugMsgr = setupDebugMessenger(instance);

    vk::SurfaceKHR surface = window.createVulkanSurface(instance);
    physicalDevice = pickPhysicalDevice(instance, surface);
    QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface); //queue families that will be used
    device = createLogicalDevice(physicalDevice, queueFamilies);

    swapChain = createSwapChain(window,
                                physicalDevice,
                                device,
                                surface,
                                queueFamilies,
                                swapChainImages,
                                swapChainImageFormat,
                                swapChainExtent);

    presentQueue = device.getQueue(queueFamilies.presentFamily.value(), 0);
    computeQueue = device.getQueue(queueFamilies.computeFamily.value(), 0);
    commandPool = createCommandPool(device, queueFamilies.computeFamily.value());
    computeCommandBuffer = createCommandBuffer(device, commandPool);
}

/**
 * Allocates and initializes all the needed buffers by the ray tracer.
 */
void RayTracerProgram::initBuffers() {
    auto [bvhSize, matSize, triSize] = scene.getBufferSizes();
    printf("log: Total scene size: %lu + %lu + %lu = %lu B\n", bvhSize, matSize, triSize, bvhSize + matSize + triSize);

    size_t sceneBufferSize = bvhSize + matSize + triSize;
    size_t rayBufferSize = swapChainExtent.width * swapChainExtent.height * 64; // size of ray
    size_t hitBufferSize = swapChainExtent.width * swapChainExtent.height * 120; // size of hit record

    vk::DeviceMemory uniformMemory, sceneMemory, uniformStagingMemory, rayMemory, hitMemory;
    uniformBuffer = createBuffer(physicalDevice,
                                 device,
                                 sizeof(CameraControls),
                                 vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                                 uniformMemory);

    sceneBuffer = createBuffer(physicalDevice,
                               device,
                               sceneBufferSize,
                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal,
                               sceneMemory);

    uniformStagingBuffer = createMappedBuffer(physicalDevice, device, sizeof(CameraControls), vk::BufferUsageFlagBits::eTransferSrc, uniformStagingMemory, uniformStagingMap);

    rayBuffer = createBuffer(physicalDevice, device, rayBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, rayMemory);
    hitBuffer = createBuffer(physicalDevice, device, hitBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, hitMemory);

    // populate the scene data
    singleTimePopulateBuffer(physicalDevice, device, commandPool, computeQueue, scene.getBuffer(), sceneBuffer);
}

/**
 * Creates the four pipelines that will be used by the ray tracer.
 */
void RayTracerProgram::initPipelines() {
    auto [bvhSize, matSize, triSize] = scene.getBufferSizes();
    std::vector<vk::ImageView> swapChainImageViews = createSwapchainViews(device, swapChainImages, swapChainImageFormat);

    vk::DescriptorPool descriptorPool = createDescriptorPool(device, (size_t) swapChainImageViews.size());
    vk::Sampler sampler = createSampler(device);

    vk::PipelineLayout genLayout, intLayout, shadeLayout, postLayout;
    vk::DescriptorSetLayout genDescrLayout, intDescrLayout, shadeDescrLayout, postDescrLayout, frameDescrLayout;

    stages.addDescriptorSet(createGenerateDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, genDescrLayout));
    stages.addDescriptorSet(createIntersectDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, hitBuffer, sceneBuffer, bvhSize, matSize, intDescrLayout));
    stages.addDescriptorSet(createShadeDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, sceneBuffer, hitBuffer, bvhSize, matSize, sampler, shadeDescrLayout));
    stages.addDescriptorSet(createPostProcessDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, postDescrLayout));
    stages.addDescriptorSets(createFramebufferDescriptorSets(device, descriptorPool, swapChainImageViews, sampler, frameDescrLayout));

    vk::Pipeline genPipeline = createComputePipeline(device, {genDescrLayout}, "build/shaders/generate.comp.spv", genLayout);
    vk::Pipeline intPipeline = createComputePipeline(device, {intDescrLayout}, "build/shaders/intersect.comp.spv", intLayout);
    vk::Pipeline shadePipeline = createComputePipeline(device, {shadeDescrLayout}, "build/shaders/shade.comp.spv", shadeLayout);
    vk::Pipeline postPipeline = createComputePipeline(device, {postDescrLayout, frameDescrLayout}, "build/shaders/postprocess.comp.spv", postLayout);

    stages.addStage(genPipeline, genLayout);
    stages.addStage(intPipeline, intLayout);
    stages.addStage(shadePipeline, shadeLayout);
    stages.addStage(postPipeline, postLayout);
}

/**
 * Main loop of the program that continuously draws frames.
 */
void RayTracerProgram::mainLoop() {
    while (!window.shouldClose()) {
        window.pollEvents();
        drawFrame();
    }

    device.waitIdle(); // wait until all submitted stuff is done
}

/**
 * Draws a frame to the screen.
 */
void RayTracerProgram::drawFrame() {
    // Frame time calculations
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFrame);

    std::cout << "\x1B[2K\r" << "ms: " << duration.count() << std::flush;
    lastFrame = currentTime;

    // No need to fence on the presentation as we only start computing when the next swapchain image is available
    // We do need to fence on compute because we'll get a new image while the previous is still computing!
    if (device.waitForFences(computeInFlightFence, vk::True, UINT64_MAX) == vk::Result::eTimeout)
        throw std::runtime_error("Fence timed out");

    auto [result, imageIndex] = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphore);

    // Suboptimal swapchain images are still considered good as they can still be presented
    // If the swapchain is completely out of date, throw an error as it shouldn't happen.
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    submitCompute(imageIndex);
    submitPresent(imageIndex);

    frameCounter++;
}

/**
 * Records and submit compute shader dispatches to the GPU
 */
void RayTracerProgram::submitCompute(uint32_t imageIndex) {
    computeCommandBuffer.reset();

    recordComputeCommandBuffer(computeCommandBuffer, imageIndex);
    device.resetFences(computeInFlightFence);

    vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eComputeShader;
    vk::SubmitInfo submitInfo {
        .sType = vk::StructureType::eSubmitInfo,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvailableSemaphore,
        .pWaitDstStageMask = &stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &computeCommandBuffer,
        .signalSemaphoreCount = 0,
    };

    // signal computeInFlightFence when the command buffer can be reused
    computeQueue.submit(submitInfo, computeInFlightFence);
}

/**
 * Submits a screen presentation request to the GPU
 */
void RayTracerProgram::submitPresent(uint32_t imageIndex) {
    vk::PresentInfoKHR presentInfo {
        .sType = vk::StructureType::ePresentInfoKHR,
        .waitSemaphoreCount = 0,
        .swapchainCount = 1,
        .pSwapchains = &swapChain,
        .pImageIndices = &imageIndex
    };

    if (presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }
}

/**
 * Records the command buffer for the compute shader.
 * This includes transitioning the image layouts, binding the correct pipelines
 * and descriptor sets, and memory barriers, as well as updating the uniforms.
 */
void RayTracerProgram::recordComputeCommandBuffer(vk::CommandBuffer &commandBuffer, uint32_t imageIndex) {
    commandBuffer.begin({
        .sType = vk::StructureType::eCommandBufferBeginInfo
    });

    updateUniformBuffer(commandBuffer);

    // Transition the layout of the image to one compute shaders can output to (VK_IMAGE_LAYOUT_GENERAL)
    transitionImageCommand(
        commandBuffer,
        swapChainImages[imageIndex],
        vk::AccessFlagBits::eNone, // flush these caches from things happening before VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        vk::AccessFlagBits::eShaderWrite, // invalidate these caches for VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT and later: the layout changed
        vk::ImageLayout::eUndefined, // discard the previous contents of the image
        vk::ImageLayout::eGeneral,
        vk::PipelineStageFlagBits::eTopOfPipe, // start the transition at the very beginning of the pipeline
        vk::PipelineStageFlagBits::eComputeShader // have it be finished by the time compute shaders start
    );

    size_t dispatchWidth = swapChainExtent.width / params.TILE_SIZE + 1;
    size_t dispatchHeight = swapChainExtent.height / params.TILE_SIZE + 1;

    vk::MemoryBarrier memoryBarrier {
        .sType = vk::StructureType::eMemoryBarrier,
        .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
        .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
    };

    // This map describes which stage (left) binds which descriptor sets (right)
    std::vector<std::pair<size_t, std::vector<size_t>>> stageSetMap = {
        {0, {0}},                   // generate stage only binds first descriptor set
        {1, {1}},                   // ...
        {2, {2}},
        {3, {3, 4 + imageIndex}},   // postProcess stage binds 2 sets: third + image index
    };

    for (auto [stage, sets] : stageSetMap) {
        stages.bindStage(commandBuffer, stage, sets);
        commandBuffer.dispatch(dispatchWidth, dispatchHeight, 1);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(0), memoryBarrier, nullptr, nullptr
        );
    }

    // Transition the image to a presentable layout (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    transitionImageCommand(
        commandBuffer,
        swapChainImages[imageIndex],
        vk::AccessFlagBits::eShaderWrite, // because the shader just wrote to it, caches must be flushed
        vk::AccessFlagBits::eNone, // invalidate these caches for VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT and later: the layout changed
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eBottomOfPipe
    );

    commandBuffer.end();
}

/**
 * Updates the uniform data for the compute shaders.
 */
void RayTracerProgram::updateUniformBuffer(vk::CommandBuffer commandBuffer) {
    CameraControls ubo = scene.getCameraControls();
    ubo.time = frameCounter;
    // to support scaled monitors
    ubo.resolution = glm::uvec2(swapChainExtent.width, swapChainExtent.height);

    memcpy(uniformStagingMap, &ubo, sizeof(ubo));

    vk::BufferCopy copy {
        .size = sizeof(ubo)
    };

    commandBuffer.copyBuffer(uniformStagingBuffer, uniformBuffer, 1, &copy);
}
