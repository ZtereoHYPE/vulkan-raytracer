#include "main.hpp"

const int TILE_SIZE = 16;

class RayTracerProgram {
   public:
    void run() {
        initVulkan();
        mainLoop();
        // cleanup();
    }

   private:
    Scene scene{ "scenes/scene.yaml" };
    Window window{ "Vulkan RayTracer", 600, 600 };

    vk::Device device;
    vk::SwapchainKHR swapChain;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    vk::Queue presentQueue;
    vk::Queue computeQueue;
    vk::Pipeline computePipeline;
    vk::PipelineLayout computePipelineLayout;

    std::vector<vk::DescriptorSet> computeDescriptorSets;
    vk::CommandBuffer computeCommandBuffer;

    vk::Fence computeInFlightFence;
    vk::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    vk::Semaphore computeFinishedSemaphore = VK_NULL_HANDLE;

    void* uniformMemoryMap = nullptr;
    void* computeSSBOMemoryMap = nullptr;

    // performance measure
    uint32_t frameCounter = 0;
    std::chrono::_V2::system_clock::time_point lastFrame;

    // std::move_only_function<void()> cleanup;

    void initVulkan() try {
        vk::Instance instance = createInstance();
        vk::DebugUtilsMessengerEXT debugMsgr = setupDebugMessenger(instance);

        vk::SurfaceKHR surface = window.createVulkanSurface(instance);

        vk::PhysicalDevice physicalDevice = pickPhysicalDevice(instance, surface);

        QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface); //queue families that will be used
        device = createLogicalDevice(physicalDevice, queueFamilies);

        presentQueue = device.getQueue(queueFamilies.presentFamily.value(), 0);
        computeQueue = device.getQueue(queueFamilies.computeFamily.value(), 0);

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

        vk::DescriptorSetLayout computeDescriptorSetLayout = createComputeDescriptorSetLayout(device);

        computePipeline = createComputePipeline(device, computeDescriptorSetLayout, computePipelineLayout);

        auto [uniformBuffer, uniformMemory, uniMap] = createMappedBuffer(physicalDevice, device, sizeof(CameraControlsUniform), vk::BufferUsageFlagBits::eUniformBuffer);
        this->uniformMemoryMap = uniMap;

        auto [infoSize, triSize] = scene.getBufferSizes();

        printf("log: Total scene size: %lu\n", infoSize + triSize);

        auto [computeSSBO, computeSSBOMemory, ssboMap] = createMappedBuffer(physicalDevice, device, infoSize + triSize, vk::BufferUsageFlagBits::eStorageBuffer);
        this->computeSSBOMemoryMap = ssboMap;

        auto [accumulatorImage, accumulatorView, accumulatorMemory] = createImage(physicalDevice, device, swapChainExtent, vk::Format::eR8G8B8A8Unorm);

        vk::CommandPool computeCommandPool = createCommandPool(device, queueFamilies.computeFamily.value());
        transitionImageLayout(device, computeCommandPool, computeQueue, accumulatorImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        vk::DescriptorPool descriptorPool = createDescriptorPool(device, swapChainImageViews.size());
        vk::Sampler sampler = createSampler(device);

        computeCommandBuffer = createCommandBuffer(device, computeCommandPool);

        computeDescriptorSets =  createComputeDescriptorSets(device,
                                                            computeDescriptorSetLayout, 
                                                            descriptorPool, 
                                                            uniformBuffer, 
                                                            computeSSBO, 
                                                            infoSize,
                                                            accumulatorView, 
                                                            swapChainImageViews, 
                                                            sampler);

        vk::SemaphoreCreateInfo semaphoreInfo {
            .sType = vk::StructureType::eSemaphoreCreateInfo
        };
        imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
        computeFinishedSemaphore = device.createSemaphore(semaphoreInfo);
        computeInFlightFence = device.createFence({
            .sType = vk::StructureType::eFenceCreateInfo,
            .flags = vk::FenceCreateFlagBits::eSignaled // it should start pre-signaled so the first frame doesnt have to wait forever
        });

        // cleanup = std::bind([
        //     instance = std::move(instance),
        //     debugMsgr = std::move(debugMsgr),
        //     surface = std::move(surface),
        //     physicalDevice = std::move(physicalDevice),
        //     swapChainImageViews = std::move(swapChainImageViews),
        //     computeDescriptorSetLayout = std::move(computeDescriptorSetLayout),
        //     computeCommandPool = std::move(computeCommandPool),
        //     descriptorPool = std::move(descriptorPool),
        //     sampler = std::move(sampler)
        // ] {});

        printf("log: initialized vulkan context\n");

    } catch (std::runtime_error &error) {
        // error handler
        std::cout << "A runtime error occurred! The program will now terminate" << std::endl;

        throw; // rethrow to halt execution
    }

    void mainLoop() {
        pushWorldData(computeSSBOMemoryMap);

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

        auto fps = 1000.0 / duration.count();
        std::cout << "FPS: " << fps << std::endl;
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

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);

        // Transition the layout of the image to one compute shaders can output to (VK_IMAGE_LAYOUT_GENERAL)
        // todo: this ideally happens before waiting for the fence, or the fence shouldn't exist at all
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

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, computeDescriptorSets[imageIndex], nullptr);

        // todo calculate the bounds better: this dispatches an extra unit row when width % TILE_SIZE == 0
        commandBuffer.dispatch(swapChainExtent.width / TILE_SIZE + 1, swapChainExtent.height / TILE_SIZE + 1, 1);

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
            vk::DependencyFlags(0), nullptr, nullptr,
            layoutTransition
        );

        commandBuffer.end();
    }

    void updateUniformBuffer(void* uniformMemoryMap) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();

        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        auto origin = gpu::vec4(0, 1, 3, 0);
        //auto origin = glm::vec4(-0.45206, 0.322176, 0.2, 0);

        float ratio = swapChainExtent.width / (float) swapChainExtent.height;
        float u, v;

        if (ratio > 1) {
            u = ratio;
            v = 1;
        } else {
            u = 1;
            v = 1/ratio;
        }

        CameraControlsUniform ubo {
            .resolution = gpu::vec2(swapChainExtent.width, swapChainExtent.height),
            .viewportUv = gpu::vec2(u, v),
            .focalLength = 1.0,
            .focusDistance = 4.8,
            .apertureRadius = 0.0,
            .time = frameCounter,
            .origin = origin,
            .rotation = rotate(glm::identity<glm::mat4>(), 3.14f, glm::vec3(0, 1, 0)), // glm::identity<glm::mat4>(),
        };

        // not super efficient, kinda like staging buffers we need push constants or whatever
        memcpy(uniformMemoryMap, &ubo, sizeof(ubo));
    }

    void pushWorldData(void* shaderBufferMemoryMap) {
        BufferBuilder infoBuffer = BufferBuilder {};
        BufferBuilder triangleBuffer = BufferBuilder {};

        scene.populateBuffers(infoBuffer, triangleBuffer);

        // write to the gpu memory
        infoBuffer.write(shaderBufferMemoryMap);
        triangleBuffer.write(shaderBufferMemoryMap + infoBuffer.getOffset()); // gcc-ism
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