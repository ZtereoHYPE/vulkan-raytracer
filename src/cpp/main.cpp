#include "main.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

class RayTracerProgram {
   public:
    void run() {
        window.setResizedCallbackVariable(&framebufferResized);
        initVulkan();
        mainLoop();
        //cleanup();
    }

   private:
    Window window = Window("Vulkan", 200, 150);

    VkDevice device;
    VkSwapchainKHR swapChain;
    VkExtent2D swapChainExtent;
    VkRenderPass renderPass;
    VkQueue presentQueue;
    VkQueue graphicsQueue;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;

    // there is one of these per concurrently rendered image
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    std::vector<void*> uniformMemoryMaps;
    void* shaderBufferMemoryMap;

    uint32_t currentFrame = 0;
    uint32_t frameCounter = 0;
    bool framebufferResized = false;
    timespec lastFrame;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<RayTracerProgram*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    std::vector<VkBuffer> createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, 
                            std::vector<VkDeviceMemory>& uniformBuffersMemory,
                            std::vector<void*>& uniformBuffersMaps) {

        std::vector<VkBuffer> uniformBuffers;
        VkDeviceSize size = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMaps.resize(MAX_FRAMES_IN_FLIGHT);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(physicalDevice, device, size, 
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                            uniformBuffers[i], 
                            uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMaps[i]);
        }

        return uniformBuffers;
    }

    VkBuffer createShaderBuffer(VkPhysicalDevice physicalDevice, VkDevice device, 
                            VkDeviceMemory& shaderBufferMemory, 
                            void*& shaderBufferMapped) {

        VkBuffer shaderBuffer;
        VkDeviceSize size = 1024; // 1 KB

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

        // depends on surface (device might not be able to present on a specific surface)
        VkPhysicalDevice physicalDevice = pickPhysicalDevice(instance, surface);


        QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice, surface); //queue families that will be used
        device = createLogicalDevice(physicalDevice, queueFamilies);

        vkGetDeviceQueue(device, queueFamilies.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilies.presentFamily.value(), 0, &presentQueue);

        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;

        swapChain = createSwapChain(window,
                                    physicalDevice, 
                                    device, 
                                    surface, 
                                    swapChainImages, 
                                    swapChainImageFormat, 
                                    swapChainExtent,
                                    queueFamilies);

        std::vector<VkImageView> swapChainImageViews = createImageViews(device, swapChainImages, swapChainImageFormat);

        renderPass = createRenderPass(device, swapChainImageFormat);

        VkDescriptorSetLayout descriptorSetLayout = createDescriptorSetLayout(device); 

        pipeline = createGraphicsPipeline(device, descriptorSetLayout, renderPass, pipelineLayout);

        framebuffers = createFramebuffers(device, renderPass, swapChainExtent, swapChainImageViews);
        
        VkCommandPool commandPool = createCommandPool(device, physicalDevice, queueFamilies);

        std::vector<VkDeviceMemory> uniformMemory;
        std::vector<VkBuffer> uniformBuffers = createUniformBuffers(physicalDevice, device, uniformMemory, uniformMemoryMaps);

        VkDeviceMemory shaderBufferMemory;
        VkBuffer shaderBuffer = createShaderBuffer(physicalDevice, device, shaderBufferMemory, shaderBufferMemoryMap);

        VkDescriptorPool descriptorPool = createDescriptorPools(device);

        descriptorSets = createDescriptorSets(descriptorSetLayout, descriptorPool, uniformBuffers, shaderBuffer);
        commandBuffers = createCommandBuffers(commandPool);

        createSyncObjects();

        printf("log: initialized vulkan context\n");
    }

    void mainLoop() {
        pushWorldData(shaderBufferMemoryMap);

        while (!window.shouldClose()) {
            glfwPollEvents();
            drawFrame();
        }

        // wait until all submitted stuff is done
        vkDeviceWaitIdle(device);
    }
    
    
    void drawFrame() {
        timespec currentTime;
        clock_gettime(CLOCK_REALTIME, &currentTime);

        long diff = currentTime.tv_nsec - lastFrame.tv_nsec;
        int fps = 1000l * 1000l * 1000l / diff;

        //printf("log: log: FPS/avg: %d\n", fps);
        
        lastFrame = currentTime;

        // wait for previous frame to complete (so we don't over-write the same command buffer while the GPU is reading it)
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        // suboptimal is considereda success code as we can still present it
        //  since recreating the swapchain means dropping the frame, 
        //  suboptimal frame is better than no frame
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            //recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        // only reset once we know for sure we are submitting work
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(uniformMemoryMaps[currentFrame]);

        VkSubmitInfo submitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // sync stuff: only wait for the semaphore at the COLOR_ATTACHMENT_OUTPUT stage (writing to the color buffer)
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores; // to signal when the command buffer is done

        // signal inFlightFence when the queue buffer can be reused
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // submit the result back to the swapchain for presentation
        VkPresentInfoKHR presentInfo {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores; // wait until command buffer done executing

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // here there are no consequences as we already presented the frame, so suboptimal = bad
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            // todo: optimize this and re-enable resizing
            //recreateSwapChain(window, physicalDevice, device, surface, swapChain, renderPass,);
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        frameCounter++;
    }


    // uses index of the swapchain image we wanna write to
    // note: record ~= "write to"
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr; // we arent inhereting any state from primary command buffers (we *are* primary)

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor; // used by attachments with VK_ATTACHMENT_LOAD_OP_CLEAR

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // inline = no secondary buffers
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline); // bind the graphics pipeline

        // specify the dynamic state
        VkViewport viewport {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor {};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // uniforms!
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        //vkCmdDraw(commandBuffer, vertices.size(), 1, 0, 0);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0); // we only draw 1 triangle that covers clip space and it's hardcoded in the shader

        vkCmdEndRenderPass(commandBuffer);


        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void updateUniformBuffer(void* uniformMemoryMap) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();

        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        auto origin = glm::vec3(0, 0, 0);

        float ratio = swapChainExtent.width / (float) swapChainExtent.height;
        float u, v;

        if (ratio > 1) {
            u = ratio;
            v = 1;
        } else {
            u = 1;
            v = 1/ratio;
        }

        float size = 2.0;

        UniformBufferObject ubo {
            .resolution = glm::vec2(swapChainExtent.width, swapChainExtent.height),
            .viewportUv = glm::vec2(u, v) * size,
            .focalLength = 1.5,
            .time = frameCounter,
            .origin = origin,
        };

        // not super efficient, kinda like staging buffers we need push constants or whatever
        memcpy(uniformMemoryMap, &ubo, sizeof(ubo));
    }

    void pushWorldData(void* shaderBufferMemoryMap) {
        SphereShaderBufferObject* sbo = reinterpret_cast<SphereShaderBufferObject*>(shaderBufferMemoryMap);

        sbo->count = 4;
        sbo->spheres[0].center = glm::vec3(3.0, 0.5, 5.0);
        sbo->spheres[0].radius = 1.5;
        sbo->spheres[0].emissive = false;
        sbo->spheres[0].color = glm::vec3(0.99, 0.43, 0.33);

        sbo->spheres[1].center = glm::vec3(0.0, 0.0, 5.0);
        sbo->spheres[1].radius = 1.0;
        sbo->spheres[1].emissive = false;
        sbo->spheres[1].color = glm::vec3(0.48, 0.62, 0.89);

        sbo->spheres[2].center = glm::vec3(0.0, -100.0, 5.0);
        sbo->spheres[2].radius = 99.0;
        sbo->spheres[2].emissive = false;
        sbo->spheres[2].color = glm::vec3(0.89, 0.7, 0.48);

        sbo->spheres[3].center = glm::vec3(-500.0, 200.0, 700.0);
        sbo->spheres[3].radius = 200.0;
        sbo->spheres[3].emissive = true;
        sbo->spheres[3].color = glm::vec3(1, 0.99, 0.9);
    }

    std::vector<VkDescriptorSet> createDescriptorSets(VkDescriptorSetLayout descriptorSetLayout, 
                                                      VkDescriptorPool descriptorPool,
                                                      std::vector<VkBuffer>& uniformBuffers,
                                                      VkBuffer& shaderBuffer) {

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

        // we create a descriptor set per frame in flight with the same
        VkDescriptorSetAllocateInfo allocInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = layouts.size();
        allocInfo.pSetLayouts = layouts.data();

        std::vector<VkDescriptorSet> descriptorSets;
        descriptorSets.resize(layouts.size());

        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor sets");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo uniformBufferInfo {};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = VK_WHOLE_SIZE;

            // and one for the sbo
            VkDescriptorBufferInfo shaderBufferInfo {};
            shaderBufferInfo.buffer = shaderBuffer; // they all point to the same buffer
            shaderBufferInfo.offset = 0;
            shaderBufferInfo.range = VK_WHOLE_SIZE;


            VkWriteDescriptorSet descriptorWrites[2] = {};

            // write descriptor for the uniform buffer
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

            // idem for the shader buffer
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &shaderBufferInfo;

            vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);
        }

        return descriptorSets;
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // it should start pre-signaled so the first frame doesnt have to wait forever

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create sync stuff!");
            }
        }
    }

    std::vector<VkCommandBuffer> createCommandBuffers(VkCommandPool commandPool) {
        std::vector<VkCommandBuffer> commandBuffers;
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo {};

        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  //directly submitted to queue, but not called from primary
        allocInfo.commandBufferCount = commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        printf("log: Created command buffer\n");

        return commandBuffers;
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