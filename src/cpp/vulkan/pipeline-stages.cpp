#include "pipeline-stages.hpp"

using namespace vk;

/**
 * Adds a descriptor set to the sets managed by this PipelineStages object.
 */
void PipelineStages::addDescriptorSet(DescriptorSet set) {
    descriptorSets.push_back(set);
}

/**
 * Adds several descriptor sets to the sets managed by this PipelineStages object.
 */
void PipelineStages::addDescriptorSets(std::vector<DescriptorSet> sets) {
    for (auto set : sets) descriptorSets.push_back(set);
}

/**
 * Adds a stage to the pipeline stages.
 * This is composed by the pipeline, its layout, and the descriptor sets the pipeline needs to bind
 */
void PipelineStages::addStage(Pipeline pipeline, PipelineLayout layout) {
    pipelines.push_back(pipeline);
    pipelineLayouts.push_back(layout);
    stages++;
}

/**
 * Binds a stage of to the command buffer, picking the right pipeline and descriptor sets
 */
void PipelineStages::bindStage(vk::CommandBuffer commandBuffer, size_t stage, std::vector<size_t> setIdxs) {
    if (stage >= stages)
        throw std::runtime_error("Stage does not exist");

    Pipeline pipeline = pipelines[stage];
    PipelineLayout layout = pipelineLayouts[stage];
    std::vector<DescriptorSet> sets;
    for (auto idx : setIdxs) sets.push_back(descriptorSets[idx]);

    commandBuffer.bindPipeline(PipelineBindPoint::eCompute, pipeline);
    commandBuffer.bindDescriptorSets(PipelineBindPoint::eCompute, layout, 0, sets, nullptr);
}
