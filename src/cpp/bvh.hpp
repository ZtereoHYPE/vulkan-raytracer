#pragma once

#include <span>

#include "config/scene.hpp"
#include "util/util.hpp"

const uint BHV_MAX_DEPTH = 64;
const int SPLIT_ATTEMPTS = 8; // set to -1 for trying at every possible position
const glm::vec3 MAX_VAL = glm::vec3(1e30, 1e30, 1e30);
const glm::vec3 MIN_VAL = glm::vec3(-1e30, -1e30, -1e30);

// forward declarations
struct Triangle; 
struct Material;

struct BvhNode {
    // This layout matches the same compact layout as the shader
    glm::vec3 min = MAX_VAL;
    uint idx;
    glm::vec3 max = MIN_VAL;
    uint amt;

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
