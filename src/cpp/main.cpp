#include "main.hpp"

const int TILE_SIZE = 16;

class RayTracerProgram {
   public:
    void run() {
        window.setResizedCallbackVariable(&framebufferResized);
        initVulkan();
        mainLoop();
        //cleanup();
    }

   private:
    Window window = Window("Vulkan", 800, 600);

    VkDevice device;
    VkSwapchainKHR swapChain;
    VkExtent2D swapChainExtent;
    std::vector<VkImage> swapChainImages;
    VkQueue presentQueue;
    VkQueue computeQueue;
    VkPipeline computePipeline;
    VkPipelineLayout computePipelineLayout;

    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkCommandBuffer computeCommandBuffer;

    VkFence computeInFlightFence;
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore computeFinishedSemaphore;

    void* uniformMemoryMap;
    void* computeSSBOMemoryMap;

    uint32_t frameCounter = 0;
    bool framebufferResized = false;
    timespec lastFrame;

    std::function<std::vector<VkDescriptorSet>(VkDevice,uint)> createDescriptorSets;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<RayTracerProgram*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    VkBuffer createUniformBuffer(VkPhysicalDevice physicalDevice, VkDevice device, 
                            VkDeviceMemory &uniformBufferMemory,
                            void* &uniformBuffersMap) {

        VkDeviceSize size = sizeof(UniformBufferObject);
        VkBuffer uniformBuffer;
        createBuffer(physicalDevice, device, size, 
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                     uniformBuffer, 
                     uniformBufferMemory);

        vkMapMemory(device, uniformBufferMemory, 0, size, 0, &uniformBuffersMap);

        return uniformBuffer;
    }

    VkBuffer createShaderBuffer(VkPhysicalDevice physicalDevice, VkDevice device, 
                            VkDeviceMemory& shaderBufferMemory, 
                            void*& shaderBufferMapped) {

        VkBuffer shaderBuffer;
        VkDeviceSize size = 2048; // 2 KB

        createBuffer(physicalDevice, device, size, 
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    shaderBuffer, 
                    shaderBufferMemory);

        vkMapMemory(device, shaderBufferMemory, 0, VK_WHOLE_SIZE, 0, &shaderBufferMapped);

        memset(shaderBufferMapped, 0, size);

        return shaderBuffer;
    }

    void initVulkan() {
        VkInstance instance = createInstance();
        VkDebugUtilsMessengerEXT debugMsgr = setupDebugMessenger(instance);

        VkSurfaceKHR surface = createVulkanWindowSurface(&window, instance);

        VkPhysicalDevice physicalDevice = pickPhysicalDevice(instance, surface);

        QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface); //queue families that will be used
        device = createLogicalDevice(physicalDevice, queueFamilies);

        vkGetDeviceQueue(device, queueFamilies.presentFamily.value(), 0, &presentQueue);
        vkGetDeviceQueue(device, queueFamilies.computeFamily.value(), 0, &computeQueue);

        VkFormat swapChainImageFormat;

        swapChain = createSwapChain(&window,
                                    physicalDevice, 
                                    device, 
                                    surface, 
                                    swapChainImages, 
                                    swapChainImageFormat, 
                                    swapChainExtent,
                                    queueFamilies);

        std::vector<VkImageView> swapChainImageViews = createSwapchainViews(device, swapChainImages, swapChainImageFormat);

        VkDescriptorSetLayout computeDescriptorSetLayout = createComputeDescriptorSetLayout(device); 

        computePipeline = createComputePipeline(device, computeDescriptorSetLayout, computePipelineLayout);

        VkCommandPool commandPool = createCommandPool(device, physicalDevice, queueFamilies);

        VkDeviceMemory uniformMemory;
        VkBuffer uniformBuffer = createUniformBuffer(physicalDevice, device, uniformMemory, uniformMemoryMap);

        VkDeviceMemory computeSSBOMemory;
        VkBuffer computeSSBO = createShaderBuffer(physicalDevice, device, computeSSBOMemory, computeSSBOMemoryMap);

        VkImageView accumulatorView;
        VkDeviceMemory accumulatorMemory;
        VkImage accumulatorImage = createImage(physicalDevice, device, swapChainExtent, VK_FORMAT_R8G8B8A8_UNORM, accumulatorView, accumulatorMemory);

        transitionImageLayout(device, commandPool, computeQueue, accumulatorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorPool descriptorPool = createDescriptorPool(device, swapChainImageViews.size());
        VkSampler sampler = createSampler(device);

        computeCommandBuffer = createCommandBuffer(device, commandPool);

        // store in closure to be able to redefine later
        createDescriptorSets = [computeDescriptorSetLayout, descriptorPool, uniformBuffer, computeSSBO, accumulatorView, swapChainImageViews, sampler]
                               (VkDevice device, uint offset) 
                               {return createComputeDescriptorSets(device, computeDescriptorSetLayout, descriptorPool, uniformBuffer, computeSSBO, offset, accumulatorView, swapChainImageViews, sampler);};

        createSyncObjects();

        printf("log: initialized vulkan context\n");
    }

    void mainLoop() {
        pushWorldData(computeSSBOMemoryMap);

        while (!window.shouldClose()) {
            glfwPollEvents();
            drawFrame();
        }

        // wait until all submitted stuff is done
        vkDeviceWaitIdle(device);
    }
    
    
    void drawFrame() {

        /* COMPUTE SUBMISSION */

        // No need to fence on the presentation as we only start computing when the next swapchain image is available
        // We do need to fence on compute because we'll get a new image while the previous is still computing!
        // todo: get rid of this by limiting cpu render-ahead
        vkWaitForFences(device, 1, &computeInFlightFence, VK_TRUE, INT_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        // If the swapchain is completely out of date, drop the frame.
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            //recreateSwapChain();
            return;

        // Suboptimal swapchain images are still considered good as they can still be presented
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        vkResetCommandBuffer(computeCommandBuffer, 0);

        updateUniformBuffer(uniformMemoryMap);
        recordComputeCommandBuffer(computeCommandBuffer, imageIndex);

        vkResetFences(device, 1, &computeInFlightFence);
        VkPipelineStageFlags stage[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

        VkSubmitInfo submitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
        submitInfo.pWaitDstStageMask = stage;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffer;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &computeFinishedSemaphore; // to signal when compute commands are done

        // signal computeInFlightFence when the command buffer can be reused
        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        };

        /* PRESENTATION */

        VkPresentInfoKHR presentInfo {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &computeFinishedSemaphore; // wait until the compute is done

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // Here there are no consequences as we already presented the frame, so suboptimal = bad
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            //recreateSwapChain(window, physicalDevice, device, surface, swapChain, renderPass,);
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        frameCounter++;
    }

    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // Transition the layout of the image to one compute shaders can output to (VK_IMAGE_LAYOUT_GENERAL)
        // todo: this ideally happens before waiting for the fence, or the fence shouldn't exist at all
        VkImageMemoryBarrier computeBarrier {};
        computeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        computeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // discard the previous contents of the image
        computeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        computeBarrier.image = swapChainImages[imageIndex];
        computeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        computeBarrier.subresourceRange.baseMipLevel = 0;
        computeBarrier.subresourceRange.levelCount = 1;
        computeBarrier.subresourceRange.baseArrayLayer = 0;
        computeBarrier.subresourceRange.layerCount = 1;

        // we aren't transferring ownership
        computeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        computeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        computeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        computeBarrier.dstAccessMask = VK_ACCESS_NONE; // todo: not sure if good

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &computeBarrier
        );

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[imageIndex], 0, 0);

        // todo calculate the bounds better
        vkCmdDispatch(commandBuffer, swapChainExtent.width / TILE_SIZE + 1, swapChainExtent.height / TILE_SIZE + 1, 1);

        // Transition the image to a presentable layout (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        VkImageMemoryBarrier presentBarrier {};
        presentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.image = swapChainImages[imageIndex];
        presentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentBarrier.subresourceRange.baseMipLevel = 0;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.baseArrayLayer = 0;
        presentBarrier.subresourceRange.layerCount = 1;

        // we aren't transferring ownership
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        presentBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        presentBarrier.dstAccessMask = VK_ACCESS_NONE; // todo: not sure if good

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &presentBarrier
        );

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void updateUniformBuffer(void* uniformMemoryMap) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();

        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        //auto origin = glm::vec4(std::sin(time)*3, 0, 0, 0);
        auto origin = glm::vec4(0, 0, 0, 0);

        float ratio = swapChainExtent.width / (float) swapChainExtent.height;
        float u, v;

        if (ratio > 1) {
            u = ratio;
            v = 1;
        } else {
            u = 1;
            v = 1/ratio;
        }

        UniformBufferObject ubo {
            .resolution = glm::vec2(swapChainExtent.width, swapChainExtent.height),
            .viewportUv = glm::vec2(u, v),
            .focalLength = 0.6,
            .focusDistance = 4.8,
            .apertureRadius = 0.0,
            .time = frameCounter,
            .origin = origin,
            .rotation = glm::lookAt(glm::vec3(origin), glm::vec3(0, 0, -3.5), glm::vec3(0, 1, 0)),
        };

        // not super efficient, kinda like staging buffers we need push constants or whatever
        memcpy(uniformMemoryMap, &ubo, sizeof(ubo));
    }

    void pushWorldData(void* shaderBufferMemoryMap) {
        BufferBuilder infoBuffer = BufferBuilder {};
        BufferBuilder triangleBuffer = BufferBuilder {};

        infoBuffer.append(3); // mesh_count
        infoBuffer.pad(12); // alignment rules

        // Mesh 1: Ground sphere
        infoBuffer.append((Mesh) {
            .is_sphere = 1,
            .sphere_radius = 99.0,
            .triangle_count = 0,
            .offset = (uint) triangleBuffer.getRelativeOffset<Triangle>(),
            .material = {
                .baseColor = glm::vec4(0.5, 0.5, 0.5, 1),
                .emissiveStrength = glm::vec4(0),
                .reflectivity = 1,
                .roughness = 0,
                .isGlass = false,
                .ior = 0,
                .shadeSmooth = false,
            },
        });
        triangleBuffer.append(glm::vec3(0.0, -100.0, 5.0), sizeof(Triangle));
        
        // Mesh 2: Sun sphere
        infoBuffer.append((Mesh) {
            .is_sphere = true,
            .sphere_radius = 200.0,
            .offset = (uint) triangleBuffer.getRelativeOffset<Triangle>(),
            .material = {
                .emissiveStrength = glm::vec4(15.0, 15.0, 15.0, 1),
            }
        });
        triangleBuffer.append(glm::vec3(-500.0, 200.0, 700.0), sizeof(Triangle));

        // Mesh 3: Test triangle
        infoBuffer.append<Mesh>({
            .is_sphere = false,
            .triangle_count = 1,
            .offset = (uint) triangleBuffer.getRelativeOffset<Triangle>(),
            .material = {
                .baseColor = glm::vec4(1.0, 1.0, 1.0, 1),
                .emissiveStrength = glm::vec4(0),
                .reflectivity = 0,
                .roughness = 0,
                .isGlass = true,
                .ior = 1.4,
                .shadeSmooth = false,
            }
        });
        triangleBuffer.append<Triangle>({
            .vertices = {
                glm::vec4(-3.0, 0.4, 7.0, 0),
                glm::vec4(2.0, -0.4, 3.0, 0),
                glm::vec4(2.0, 1.0, 5.0, 0)
            }
        });

        // create descriptor sets given known offset within the buffer
        computeDescriptorSets = this->createDescriptorSets(device, infoBuffer.getOffset());

        // write to the gpu memory
        infoBuffer.write(shaderBufferMemoryMap);
        triangleBuffer.write(shaderBufferMemoryMap + infoBuffer.getOffset());
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // it should start pre-signaled so the first frame doesnt have to wait forever

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create sync stuff!");
        }
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