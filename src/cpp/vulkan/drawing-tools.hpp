#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <iostream>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "../util/buffer-builder.hpp"
#include "init.hpp"

/**
 * This header contains useful functions for repeated or long actions related
 * to rendering or drawing in some way, rather than being used solely to initialize
 * the vulkan context.
 */

void transitionImageCommand(vk::CommandBuffer const &commandBuffer,
                            vk::Image const &image,
                            vk::AccessFlags flushCaches,
                            vk::AccessFlags invalidateCaches,
                            vk::ImageLayout oldLayout,
                            vk::ImageLayout newLayout,
                            vk::PipelineStageFlags transitionStart,
                            vk::PipelineStageFlags transitionEndBy);

void singleTimeCopyBuffer(vk::Device const &device,
                          vk::CommandPool const &commandPool,
                          vk::Queue const &queue,
                          vk::Buffer const &src,
                          vk::Buffer const &dst,
                          vk::DeviceSize size);

void singleTimePopulateBuffer(vk::PhysicalDevice const &physicalDevice,
                              vk::Device const &device,
                              vk::CommandPool const &commandPool,
                              vk::Queue const &queue,
                              BufferBuilder memory,
                              vk::Buffer &dst);

void singleTimeTransitionImageLayout(vk::Device const &device,
                                     vk::CommandPool const &commandPool,
                                     vk::Queue const &queue,
                                     vk::Image const &image,
                                     vk::ImageLayout oldLayout,
                                     vk::ImageLayout newLayout);

void dumpImageView(vk::PhysicalDevice const &physicalDevice,
                   vk::Device const &device,
                   vk::CommandPool const &commandPool,
                   vk::Queue const &queue,
                   vk::Image const &image,
                   vk::ImageLayout layout,
                   vk::Extent2D imageExtent);