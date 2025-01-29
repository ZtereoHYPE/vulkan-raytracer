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

/**
 * Dumps an image to the location specified by params.DUMP_FILE in raw binary format.
 * 
 * Makes another image to copy it to a known format, and then a cpu-mapped buffer
 * to copy that image to.
 */
void dumpImageView(PhysicalDevice const &physicalDevice,
                   Device const &device,
                   CommandPool const &commandPool,
                   Queue const &queue,
                   Image const &framebuffer,
                   ImageLayout layout,
                   Extent2D imageExtent) {

    // Step 1: create an image with known tiling, format, and size
    Image image = device.createImage({ 
        .sType = StructureType::eImageCreateInfo,
        .imageType = ImageType::e2D,
        .format = vk::Format::eB8G8R8A8Unorm,
        .extent = {imageExtent.width, imageExtent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = SampleCountFlagBits::e1,
        .tiling = ImageTiling::eLinear,
        .usage = ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst,
        .sharingMode = SharingMode::eExclusive,
        .initialLayout = ImageLayout::eUndefined
    });
    MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
    DeviceMemory imageMemory = device.allocateMemory({
        .sType = StructureType::eMemoryAllocateInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, MemoryPropertyFlagBits::eDeviceLocal),
    });
    device.bindImageMemory(image, imageMemory, 0);


    // Step 2: Create a CPU-mapped buffer
    size_t size = 4 * imageExtent.width * imageExtent.height;

    void *map;
    DeviceMemory bufferMemory;
    Buffer mappedBuffer = createMappedBuffer(physicalDevice, device, size, BufferUsageFlagBits::eTransferDst, bufferMemory, map);


    // Step 3: Copy the framebuffer to the image, and then to the cpu buffer
    CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);
    transitionImageCommand(commandBuffer, image,
                           AccessFlagBits::eNone,
                           AccessFlagBits::eTransferRead,
                           ImageLayout::eUndefined,
                           ImageLayout::eGeneral,
                           PipelineStageFlagBits::eTopOfPipe,
                           PipelineStageFlagBits::eTransfer);

    ImageSubresourceLayers subresource {
        .aspectMask = ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
    };
    ImageBlit imageBlit {
        .srcSubresource = subresource,
        .srcOffsets = std::array {
            Offset3D {0, 0, 0},
            Offset3D {(int32_t)imageExtent.width, (int32_t)imageExtent.height, 1},
        },
        .dstSubresource = subresource,
        .dstOffsets = std::array {
            Offset3D {0, 0, 0},
            Offset3D {(int32_t)imageExtent.width, (int32_t)imageExtent.height, 1},
        },
    };
    commandBuffer.blitImage(framebuffer, layout, image, ImageLayout::eGeneral, 1, &imageBlit, Filter::eNearest);

    commandBuffer.pipelineBarrier(PipelineStageFlagBits::eTransfer,
                                  PipelineStageFlagBits::eTransfer,
                                  DependencyFlags(0),
                                  nullptr, nullptr, nullptr);

    BufferImageCopy bufferRegion {
        .bufferOffset = 0,
        .imageSubresource = subresource,
        .imageOffset = {0, 0, 0},
        .imageExtent = {imageExtent.width, imageExtent.height, 1},
    };
    commandBuffer.copyImageToBuffer(image, ImageLayout::eGeneral, mappedBuffer, 1, &bufferRegion);

    endSingleTimeCommands(queue, std::move(commandBuffer));


    // Step 4: dump to file
    std::ofstream fout(params.DUMP_FILE, std::ios::binary);
    fout.write(static_cast<char*>(map), static_cast<long>(size));
    fout.close();

    // free the staging buffer and memory
    device.freeMemory(bufferMemory);
    device.destroyBuffer(mappedBuffer);
    device.freeMemory(imageMemory);
    device.destroyImage(image);
}