#pragma once

#include <span>

#include "pch.hpp"
#include "util/gpu-types.hpp"
#include "scene.hpp"

const uint BHV_MAX_DEPTH = 16;
const gpu::vec3 MAX_VAL = gpu::vec3(1e30, 1e30, 1e30);
const gpu::vec3 MIN_VAL = gpu::vec3(-1e30, -1e30, -1e30);

struct Triangle; // forward declaration

// todo: this can be compacted to 2 vec4s (last is amt and idx/left)
struct BvhNode {
    /* Offset and stride within respective buffer */
    gpu::u32 idx;
    gpu::u32 amt;

    /* Children indices */
    gpu::u32 left;

    /* Bounds of the node */
    gpu::vec3 min = MAX_VAL;
    gpu::vec3 max = MIN_VAL;

    void initialize(std::vector<Triangle> &triangles, std::span<uint> &indices, uint offset);
    void expand(Triangle tri);
    float area();
};

class BvhBuilder {
    std::vector<BvhNode> bvhList{};
    std::vector<Triangle> &triangles;

public:
    BvhBuilder(std::vector<Triangle> &triangles) : triangles(triangles) {};
    std::vector<BvhNode> buildBvh();

private:
    void buildRecursively(size_t nodeIdx, std::span<uint> ordering, uint depth, uint offset, float parentCost);
    float splitCost(std::span<uint> &indices, size_t axis, float location);
    std::tuple<size_t, float> findBestSplit(std::span<uint> &indices);
    
    static void applyOrdering(std::vector<Triangle>& items, const std::vector<uint>& ordering);
    static void swap(uint &left, uint &right);
};
