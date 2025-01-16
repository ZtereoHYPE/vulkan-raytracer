#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vector>
#include <vulkan/vulkan.hpp>

/**
 * Object containing various stages of a compute pipeline.
 * It handles binding the right descriptor sets and pipeline when needed.
 */
class PipelineStages {
    size_t stages = 0;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::Pipeline> pipelines;
    std::vector<vk::PipelineLayout> pipelineLayouts;

public:
    void addDescriptorSet(vk::DescriptorSet set);
    void addDescriptorSets(std::vector<vk::DescriptorSet> sets);
    void addStage(vk::Pipeline pipeline, vk::PipelineLayout layout);
    void bindStage(vk::CommandBuffer commandBuffer, size_t stage, std::vector<size_t> sets);
};
