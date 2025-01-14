#include "main.hpp"

class RayTracerProgram {
   public:
    void run() {
        initVulkan();
        mainLoop();
        // cleanup();
    }

   private:
    Scene scene{ params.SCENE_FILE };
    Window window{ 
        "Vulkan RayTracer", 
        scene.getCameraControls().resolution[0], 
        scene.getCameraControls().resolution[1] 
    };

    vk::Device device;
    vk::SwapchainKHR swapChain;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    vk::Queue presentQueue;
    vk::Queue computeQueue;

    vk::Pipeline genPipeline;
    vk::Pipeline intPipeline;
    vk::Pipeline shadePipeline;
    vk::Pipeline postPipeline;
    vk::PipelineLayout genLayout;
    vk::PipelineLayout intLayout;
    vk::PipelineLayout shadeLayout;
    vk::PipelineLayout postLayout;

    vk::DescriptorSet genDescriptorSet;
    vk::DescriptorSet intDescriptorSet;
    vk::DescriptorSet shadeDescriptorSet;
    vk::DescriptorSet postDescriptorSet;
    std::vector<vk::DescriptorSet> frameDescriptorSet;
    vk::CommandBuffer computeCommandBuffer;

    vk::Fence computeInFlightFence;
    vk::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    vk::Semaphore computeFinishedSemaphore = VK_NULL_HANDLE;

    void* uniformMemoryMap = nullptr;
    void* sceneMemoryMap = nullptr;

    // performance measure
    uint32_t frameCounter = 0;
    std::chrono::_V2::system_clock::time_point lastFrame;

    // std::move_only_function<void()> cleanup;

    void initVulkan() try {
        vk::Instance instance = createInstance();
        vk::DebugUtilsMessengerEXT debugMsgr = setupDebugMessenger(instance);


        // Window, device, and swapchain initialization
        vk::SurfaceKHR surface = window.createVulkanSurface(instance);
        vk::PhysicalDevice physicalDevice = pickPhysicalDevice(instance, surface);
        QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface); //queue families that will be used
        device = createLogicalDevice(physicalDevice, queueFamilies);

        vk::Format swapChainImageFormat;
        swapChain = createSwapChain(window,
                                    physicalDevice,
                                    device,
                                    surface,
                                    queueFamilies,
                                    swapChainImages,
                                    swapChainImageFormat,
                                    swapChainExtent);
        std::vector<vk::ImageView> swapChainImageViews = createSwapchainViews(device, swapChainImages, swapChainImageFormat);


        // Queue and command pool initialization
        presentQueue = device.getQueue(queueFamilies.presentFamily.value(), 0);
        computeQueue = device.getQueue(queueFamilies.computeFamily.value(), 0);
        vk::CommandPool commandPool = createCommandPool(device, queueFamilies.computeFamily.value());
        computeCommandBuffer = createCommandBuffer(device, commandPool);


        // Images and Buffers allocation and initialization
        auto [bvhSize, matSize, triSize] = scene.getBufferSizes();
        printf("log: Total scene size: %lu + %lu + %lu = %lu B\n", bvhSize, matSize, triSize, bvhSize + matSize + triSize);

        auto [uniformBuffer, uniformMemory, uniformMemoryMap] = createMappedBuffer(physicalDevice, device, sizeof(CameraControlsUniform), vk::BufferUsageFlagBits::eUniformBuffer);
        this->uniformMemoryMap = uniformMemoryMap;

        auto [sceneBuffer, sceneMemory, sceneMemoryMap] = createMappedBuffer(physicalDevice, device, bvhSize + matSize + triSize, vk::BufferUsageFlagBits::eStorageBuffer);
        this->sceneMemoryMap = sceneMemoryMap;

        size_t rayBufferSize = swapChainExtent.width * swapChainExtent.height * 64; // size of ray
        size_t hitBufferSize = swapChainExtent.width * swapChainExtent.height * 120; // size of hit record
        auto [rayBuffer, rayMemory] = createBuffer(physicalDevice, device, rayBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto [hitBuffer, hitMemory] = createBuffer(physicalDevice, device, hitBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
        clearBuffer(device, commandPool, computeQueue, rayBuffer, rayBufferSize); // clear ray buffer


        // Descriptor sets and pipeline creation
        vk::DescriptorPool descriptorPool = createDescriptorPool(device, (size_t) swapChainImageViews.size());
        vk::Sampler sampler = createSampler(device);

        vk::DescriptorSetLayout genDescrLayout, intDescrLayout, shadeDescrLayout, postDescrLayout, frameDescrLayout;

        genDescriptorSet = createGenerateDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, genDescrLayout);
        intDescriptorSet = createIntersectDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, hitBuffer, sceneBuffer, bvhSize, matSize, intDescrLayout);
        shadeDescriptorSet = createShadeDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, sceneBuffer, hitBuffer, bvhSize, matSize, sampler, shadeDescrLayout);
        postDescriptorSet = createPostProcessDescriptorSet(device, descriptorPool, uniformBuffer, rayBuffer, sampler, postDescrLayout);
        frameDescriptorSet = createFramebufferDescriptorSets(device, descriptorPool, swapChainImageViews, sampler, frameDescrLayout);

        genPipeline = createComputePipeline(device, {genDescrLayout}, "build/shaders/generate.comp.spv", genLayout); // todo: move main in function
        intPipeline = createComputePipeline(device, {intDescrLayout}, "build/shaders/intersect.comp.spv", intLayout);
        shadePipeline = createComputePipeline(device, {shadeDescrLayout}, "build/shaders/shade.comp.spv", shadeLayout);
        postPipeline = createComputePipeline(device, {postDescrLayout, frameDescrLayout}, "build/shaders/postprocess.comp.spv", postLayout);


        // Synchronization structure initialization
        vk::SemaphoreCreateInfo semaphoreInfo {
            .sType = vk::StructureType::eSemaphoreCreateInfo
        };
        imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
        computeFinishedSemaphore = device.createSemaphore(semaphoreInfo);
        computeInFlightFence = device.createFence({
            .sType = vk::StructureType::eFenceCreateInfo,
            .flags = vk::FenceCreateFlagBits::eSignaled // it should start pre-signaled so the first frame doesnt have to wait forever
        });

        //cleanup = std::bind([&] {});

        printf("log: initialized vulkan context\n");

    } catch (std::runtime_error &) {
        // error handler
        std::cout << "A runtime error occurred! The program will now terminate" << std::endl;

        throw; // rethrow to halt execution
    }

    void mainLoop() {
        scene.writeBuffers(sceneMemoryMap);

        while (!window.shouldClose()) {
            window.pollEvents();
            drawFrame();
        }

        // wait until all submitted stuff is done
        device.waitIdle();
    }

    void drawFrame() {
        // Frame time calculations
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFrame);

        std::cout << "\x1B[2K\r" << "ms: " << duration.count() << std::flush;
        lastFrame = currentTime;

        /* COMPUTE SUBMISSION */

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

        computeCommandBuffer.reset();

        updateUniformBuffer(uniformMemoryMap);
        recordComputeCommandBuffer(computeCommandBuffer, imageIndex);
        device.resetFences(computeInFlightFence);

        vk::PipelineStageFlags stage[] = {
            vk::PipelineStageFlagBits::eComputeShader
        };

        vk::SubmitInfo submitInfo {
            .sType = vk::StructureType::eSubmitInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &imageAvailableSemaphore,
            .pWaitDstStageMask = stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &computeCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &computeFinishedSemaphore, // to signal when compute commands are done executing
        };

        // signal computeInFlightFence when the command buffer can be reused
        computeQueue.submit(submitInfo, computeInFlightFence);

        /* PRESENTATION */

        vk::PresentInfoKHR presentInfo {
            .sType = vk::StructureType::ePresentInfoKHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &computeFinishedSemaphore, // wait until the compute is done
            .swapchainCount = 1,
            .pSwapchains = {&swapChain},
            .pImageIndices = &imageIndex
        };

        if (presentQueue.presentKHR(presentInfo) != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        frameCounter++;
    }

    void recordComputeCommandBuffer(vk::CommandBuffer &commandBuffer, uint32_t imageIndex) {
        commandBuffer.begin({
            .sType = vk::StructureType::eCommandBufferBeginInfo
        });

        // Transition the layout of the image to one compute shaders can output to (VK_IMAGE_LAYOUT_GENERAL)
        vk::ImageMemoryBarrier layoutTransition = {
            .sType = vk::StructureType::eImageMemoryBarrier,
            // https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
            .srcAccessMask = vk::AccessFlagBits::eNone, // flush these caches from things happening before VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            .dstAccessMask = vk::AccessFlagBits::eShaderWrite, // invalidate these caches for VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT and later, the layout changed
            .oldLayout = vk::ImageLayout::eUndefined, // discard the previous contents of the image
            .newLayout = vk::ImageLayout::eGeneral,
            // we aren't transferring ownership
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(0), nullptr, nullptr,
            layoutTransition
        );

        size_t dispatchWidth = swapChainExtent.width / params.TILE_SIZE + 1;
        size_t dispatchHeight = swapChainExtent.height / params.TILE_SIZE + 1;

        // todo: potentially write these for the specific buffers rather than the whole thing
        vk::MemoryBarrier memoryBarrier {
            .sType = vk::StructureType::eMemoryBarrier,
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
        };

        // Generate (missing/new) rays
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, genPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, genLayout, 0, genDescriptorSet, nullptr);
        commandBuffer.dispatch(dispatchWidth, dispatchHeight, 1);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(0), memoryBarrier, nullptr, nullptr
        );

        // Intersect the geometry
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, intPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, intLayout, 0, intDescriptorSet, nullptr);
        commandBuffer.dispatch(dispatchWidth, dispatchHeight, 1);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(0), memoryBarrier, nullptr, nullptr
        );

        // Shade and update rays
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, shadePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, shadeLayout, 0, shadeDescriptorSet, nullptr);
        commandBuffer.dispatch(dispatchWidth, dispatchHeight, 1);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(0), memoryBarrier, nullptr, nullptr
        );


        // image barrier here?
        // Post process the accumulated rays
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, postPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, postLayout, 0, {postDescriptorSet, frameDescriptorSet[imageIndex]}, nullptr);
        commandBuffer.dispatch(dispatchWidth, dispatchHeight, 1);

        // Transition the image to a presentable layout (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        layoutTransition = {
            .sType = vk::StructureType::eImageMemoryBarrier,
            // https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite, // because the shader just wrote to it, caches must be flushed
            .dstAccessMask = vk::AccessFlagBits::eNone, // bottom of pipe has no access
            .oldLayout = vk::ImageLayout::eGeneral, // discard the previous contents of the image
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            // we aren't transferring ownership
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        computeCommandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::DependencyFlags(0), memoryBarrier, nullptr,layoutTransition
        );

        commandBuffer.end();
    }

    void updateUniformBuffer(void* uniformMemoryMap) {
        CameraControlsUniform ubo = scene.getCameraControls();
        ubo.time = frameCounter;
        // to support scaled monitors
        ubo.resolution = glm::uvec2(swapChainExtent.width, swapChainExtent.height);

        // not super efficient as the memory has to be host-visible and writeable
        // if this matters at all, push constants could be a solution
        memcpy(uniformMemoryMap, &ubo, sizeof(ubo));
    }
};

int main() {
    RayTracerProgram prog;

    try {
        prog.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}