#include "drawing-tools.hpp"

using namespace vk;

/**
 * Helper function to begin a single-time command to the GPU.
 *
 * Used for setting up images or buffers.
 */
CommandBuffer beginSingleTimeCommands(Device const &device, CommandPool const &commandPool) {
    CommandBuffer commandBuffer = device.allocateCommandBuffers({
        .sType = StructureType::eCommandBufferAllocateInfo,
        .commandPool = commandPool,
        .level = CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]; // a bit ugly since allocateCommandBuffers returns an array but we only want the first.

    commandBuffer.begin({
        .sType = StructureType::eCommandBufferBeginInfo,
        .flags = CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    return commandBuffer;
}



/**
 * Helper function to end and issue a single-time command to the GPU.
 *
 * Used for setting up images or buffers.
 * The commandBuffer is an rvalue reference to force its destruction in this function.
 */
void endSingleTimeCommands(Queue const &queue, CommandBuffer &&commandBuffer) {
    commandBuffer.end();

    SubmitInfo submitInfo {
        .sType = StructureType::eSubmitInfo,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };

    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

/**
 * Drawing utility function to manually transition the layout of an image.
 * This function makes a few assumption about the image, namely that it doesn't have
 * mipmaps and that it's a 1-layer image, as well as the transition isn't changing
 * the queue that owns the image.
 *
 * A good resource to understand image transition and synchronization is:
 * https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
 */
void transitionImageCommand(CommandBuffer const &commandBuffer,
                            Image const &image,
                            AccessFlags flushCaches,
                            AccessFlags invalidateCaches,
                            ImageLayout oldLayout,
                            ImageLayout newLayout,
                            PipelineStageFlags transitionStart,
                            PipelineStageFlags transitionEndBy) {

    ImageMemoryBarrier layoutTransition = {
        .sType = StructureType::eImageMemoryBarrier,
        .srcAccessMask = flushCaches,
        .dstAccessMask = invalidateCaches,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // we assume we aren't transferring ownership
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    commandBuffer.pipelineBarrier(
        transitionStart,
        transitionEndBy,
        DependencyFlags(0), nullptr, nullptr,
        layoutTransition
    );
}

/**
 * Uses a staging buffer to populate a buffer with some data.
 * The destination buffer must be created with eTransferDst usage flag
 *
 * Should only be used while setting up the initial state of buffers,
 * and not while drawing frames.
 */
void singleTimePopulateBuffer(PhysicalDevice const &physicalDevice,
                              Device const &device,
                              CommandPool const &commandPool,
                              Queue const &queue,
                              BufferBuilder memory,
                              Buffer &dst) {

    size_t size = memory.getOffset();

    // create a staging buffer
    DeviceMemory stagingMemory;
    void *stagingBufferMap;
    Buffer stagingBuffer = createMappedBuffer(
        physicalDevice,
        device,
        size,
        BufferUsageFlagBits::eTransferSrc,
        stagingMemory,
        stagingBufferMap
    );

    // write to it
    memory.write(stagingBufferMap);

    // perform a single time copy
    CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    BufferCopy copyRegion {.size = size};
    commandBuffer.copyBuffer(stagingBuffer, dst, copyRegion);
    endSingleTimeCommands(queue, std::move(commandBuffer));

    // free the staging buffer and memory
    device.freeMemory(stagingMemory);
    device.destroyBuffer(stagingBuffer);
}

/**
 * Issues a command to initially transition an image to a required
 * layout.
 *
 * Should only be used while setting up the vulkan context and not while drawing
 * frames.
 */
void singleTimeTransitionImageLayout(Device const &device,
                                     CommandPool const &commandPool,
                                     Queue const &queue,
                                     Image const &image,
                                     ImageLayout oldLayout,
                                     ImageLayout newLayout) {

    CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    // barriers are an easy way to transition image layouts
    ImageMemoryBarrier barrier {
        .sType = StructureType::eImageMemoryBarrier,
        // the image is not being accessed until the barrier is done, so we don't need to specify the stages
        .srcAccessMask = AccessFlagBits::eNone,
        .dstAccessMask = AccessFlagBits::eNone,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        // we aren't transferring ownership
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    commandBuffer.pipelineBarrier(
        PipelineStageFlagBits::eBottomOfPipe, PipelineStageFlagBits::eTopOfPipe, // doesn't matter as we manually sync this
        Flags<DependencyFlagBits>(), // no dependencies
        nullptr,
        nullptr,
        barrier
    );

    endSingleTimeCommands(queue, std::move(commandBuffer));
}

/**
 * Issues a command to copy a buffer from a source to a destination.
 *
 * Should only be used while setting up the vulkan context and not while drawing
 * frames.
 */
void singleTimeCopyBuffer(Device const &device,
                          CommandPool const &commandPool,
                          Queue const &queue,
                          Buffer &src,
                          Buffer &dst,
                          DeviceSize size) {

    // we need to create a command buffer to submit a command to do this
    CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    BufferCopy copyRegion {.size = size};
    commandBuffer.copyBuffer(src, dst, copyRegion);

    endSingleTimeCommands(queue, std::move(commandBuffer));
}
