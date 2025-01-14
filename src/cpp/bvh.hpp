#pragma once

#include <span>

#include "config/scene.hpp"
#include "util/util.hpp"
#include "config/parameters.hpp"

const glm::vec3 MAX_VAL = glm::vec3(1e30, 1e30, 1e30);
const glm::vec3 MIN_VAL = glm::vec3(-1e30, -1e30, -1e30);

// forward declarations
struct Triangle; 
struct Material;

/*
 * Class that represents a node of the BVH tree.
 */
struct BvhNode {
    // This layout matches the same compact layout as the shader
    glm::vec3 min = MAX_VAL;
    uint idx;
    glm::vec3 max = MIN_VAL;
    uint amt;

    void initialize(std::vector<Triangle> &triangles, std::span<uint> &indices, uint offset);
    void expand(Triangle tri);
    float area();
};

/*
 * Builder class for the BVH tree.
 */
class BvhBuilder {
    std::vector<BvhNode> bvhList{};
    std::vector<Triangle> &triangles;
    std::vector<Material> &materials;

public:
    BvhBuilder(std::vector<Triangle> &triangles, std::vector<Material> materials)
        : triangles(triangles), materials(materials) {};
    std::vector<BvhNode> build();

private:
    void buildRecursively(size_t nodeIdx, std::span<uint> ordering, uint depth, uint offset, float parentCost);
    std::tuple<size_t, float> findBestSplit(uint nodeIdx, std::span<uint> &indices);
    float splitCost(std::span<uint> &indices, size_t axis, float location);
    void applyMotionBlur(size_t nodeIdx);
    
    static void applyOrdering(std::vector<Triangle>& items, const std::vector<uint>& ordering);
};
