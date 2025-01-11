#pragma once

#include <span>

#include "util/gpu-types.hpp"
#include "config/scene.hpp"

const uint BHV_MAX_DEPTH = 64;
const int SPLIT_ATTEMPTS = 8; // set to -1 for trying at every possible position
const gpu::vec3 MAX_VAL = gpu::vec3(1e30, 1e30, 1e30);
const gpu::vec3 MIN_VAL = gpu::vec3(-1e30, -1e30, -1e30);

// forward declarations
struct Triangle; 
struct Material;

struct BvhNode {
    // This layout matches the same compact layout as the shader
    union {
        gpu::vec3 min = MAX_VAL;
        struct {
            int pad[3];
            gpu::u32 idx;
        } idx;
    } min_idx;

    union {
        gpu::vec3 max = MIN_VAL;
        struct {
            int pad[3];
            gpu::u32 amt;
        } amt;
    } max_amt;

    void initialize(std::vector<Triangle> &triangles, std::span<uint> &indices, uint offset);
    void expand(Triangle tri);
    // todo: isLeaf and other helper methods
    float area();
};

class BvhBuilder {
    std::vector<BvhNode> bvhList{};
    std::vector<Triangle> &triangles;
    std::vector<Material> &materials;

public:
    BvhBuilder(std::vector<Triangle> &triangles, std::vector<Material> materials) 
        : triangles(triangles), materials(materials) {};
    std::vector<BvhNode> buildBvh();

private:
    void buildRecursively(size_t nodeIdx, std::span<uint> ordering, uint depth, uint offset, float parentCost);
    std::tuple<size_t, float> findBestSplit(uint nodeIdx, std::span<uint> &indices);
    float splitCost(std::span<uint> &indices, size_t axis, float location);
    void applyMotionBlur(size_t nodeIdx);
    
    static void applyOrdering(std::vector<Triangle>& items, const std::vector<uint>& ordering);
    static void swap(uint &left, uint &right);
};
