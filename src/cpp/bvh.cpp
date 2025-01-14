#include "bvh.hpp"

/** Helper function to get the minimum coordinate of a triangle given the needed axis */
float axisMin(Triangle tri, size_t axis) {
    float min = tri.vertices[0][axis];
    for (int vtx = 1; vtx < 3; ++vtx) {
        float val = tri.vertices[vtx][axis];
        if (val < min) min = val;
    }

    return min;
}

/** Helper function to get the maximum coordinate of a triangle given the needed axis */
float axisMax(Triangle tri, size_t axis) {
    float max = tri.vertices[0][axis];
    for (int vtx = 1; vtx < 3; ++vtx) {
        float val = tri.vertices[vtx][axis];
        if (val > max) max = val;
    }

    return max;
}

/** Helper function to swap two uints */
void swap(uint &left, uint &right) {
    uint tmp = left;
    left = right;
    right = tmp;
}

/** Returns surface area of bounding box. Used for SAH */
float BvhNode::area() {
    // if the node is uninitialized / expanded with 0 trianges thne this yields NaN
    if (max == MIN_VAL || min == MAX_VAL) return 0;
    glm::vec3 len = max - min;
    return 2 * (len[0] * len[1] + len[0] * len[2] + len[1] * len[2]);
}

/** Expands the BVN node's bounds to include the given triangle / sphere */
void BvhNode::expand(Triangle tri) {
    min = glm::min(min, tri.minBound());
    max = glm::max(max, tri.maxBound());
}

/**
 * Initializes a node based on the given data.
 * Afterwards, this node is ready to be used as a leaf node.
 */
void BvhNode::initialize(std::vector<Triangle> &triangles, std::span<uint> &indices, uint offset) {
    // Expand the node's bounds
    for (auto idx : indices) 
        expand(triangles[idx]);

    // Populate the various fields
    idx = offset;           // offset into the buffer
    amt = indices.size();   // amt size => leaf
}

/**
 * Builder method to finally the BVH tree.
 * The nodes are stored linearly in a vector which is returned, and the triangles
 * are reordered using an index array during the construction of the BVH to avoid
 * excessive memory writes.
 * Motion blur is applied as an additional step.
 */
std::vector<BvhNode> BvhBuilder::build() {
    // Create triangle index list with linear sequence (0,1,2...)
    std::vector<uint> indices(triangles.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Push the initial node
    bvhList.push_back(BvhNode());

    // Build the BVH
    buildRecursively(0, std::span(indices), 0, 0, 1e30);

    // Applies the indices to the triangles.
    applyOrdering(triangles, indices);

    // Apply motion blur
    applyMotionBlur(0);

    return bvhList;
}

/**
 * Builds the BVH by recursively performing the following steps:
 * - Initialize the node as leaf
 * - Find the best split location and split the indices
 * - If no improvements are yielded over the parent node return. This node is now a leaf.
 * - Else, change the node to be a parent node, push the two children, and recurse on them.
 */
void BvhBuilder::buildRecursively(size_t nodeIdx, std::span<uint> indices, uint depth, uint offset, float parentCost) {
    // Initialize the node's data.
    bvhList[nodeIdx].initialize(triangles, indices, offset);

    if (depth >= params.MAX_BVH_DEPTH || indices.size() <= 1) return;

    auto [splitAxis, splitPos] = findBestSplit(nodeIdx, indices);
    float bestCost = splitCost(indices, splitAxis, splitPos);

    // If we aren't improving over the parent, return.
    if (bestCost >= parentCost) return;
    
    // Perform the split
    int leftIdx = 0;
    int rightIdx = indices.size() - 1;
    while (leftIdx <= rightIdx) {
        Triangle tri = triangles[indices[leftIdx]];
        float center = (axisMin(tri, splitAxis) + axisMax(tri, splitAxis)) / 2;

        if (center < splitPos) {
            ++leftIdx;
        } else {
            // This ends up ordering the index array
            swap(indices[leftIdx], indices[rightIdx--]);
        }
    }

    // Avoid creating empty nodes
    if ((leftIdx == 0) || (rightIdx == indices.size() - 1)) return;

    // Add the children indices and recurse down
    uint leftNodeIdx = bvhList.size();

    bvhList[nodeIdx].idx = leftNodeIdx; // child index
    bvhList[nodeIdx].amt = 0;           // not a leaf

    BvhNode leftNode{}, rightNode{};
    bvhList.push_back(leftNode);
    bvhList.push_back(rightNode);

    buildRecursively(leftNodeIdx, indices.subspan(0, leftIdx), depth + 1, offset, bestCost);
    buildRecursively(leftNodeIdx + 1, indices.subspan(leftIdx, indices.size() - leftIdx), depth + 1, offset + leftIdx, bestCost);
}

/**
 * Finds the best location and axis to perform a split by attempting to split the
 * volume SPLIT_ATTEMPTS times, and minimising a cost function.
 */
std::tuple<size_t, float> BvhBuilder::findBestSplit(uint nodeIdx, std::span<uint> &indices) {
    // Get the node's AABB information
    BvhNode node = bvhList[nodeIdx];
    glm::vec3 startPos = node.min;
    glm::vec3 dimentions = (node.max - node.min) / (params.BVH_SPLIT_ATTEMPTS + 1);

    // Use the SAH to find the best split position by trying to split at SPLIT_ATTEMPT uniform intervals
    size_t bestAxis = -1;
    float bestPos;
    float bestCost = 1e30;
    for (size_t axis : {0, 1, 2}) {
        if (params.BVH_SPLIT_ATTEMPTS == -1) {
            // Try at every possible position
            for (uint const triIdx : indices) {
                Triangle const tri = triangles[triIdx];
                float pos = (axisMin(tri, axis) + axisMax(tri, axis)) / 2;
                float cost = splitCost(indices, axis, pos);

                if (cost < bestCost) {
                    bestAxis = axis;
                    bestPos = pos;
                    bestCost = cost;
                }
            }
        } else {
            // Try SPLIT_ATTEMPTS splits and pick the best
            for (uint attempt = 1; attempt <= params.BVH_SPLIT_ATTEMPTS; ++attempt) {
                float pos = startPos[axis] + dimentions[axis] * attempt;
                float cost = splitCost(indices, axis, pos);

                if (cost < bestCost) {
                    bestAxis = axis;
                    bestPos = pos;
                    bestCost = cost;
                }
            }
        }
    }

    if (bestAxis == -1)
        throw std::runtime_error("No good split was found! This should never happen.");

    return std::make_tuple(bestAxis, bestPos);
}

/**
 * The Surface Area Heuristic estimates the "cost" of a split, to be minimized.
 *
 * The reasoning behind it is that the factor that a BVH wants to minimize is 
 * the amount of intersection checks, and a larger area is more likely to be 
 * intersected. 
 * This means that a very small area with very few triangles has a low cost,
 * but a large area with a lot of triangles might end up costing a lot if hit,
 * and since it's very likely to be hit its cost will be high:
 * 
 * Cost = TriangleAmountLeft * AreaLeft + TriangleAmountRight * AreaRight
 */
float BvhBuilder::splitCost(std::span<uint> &indices, size_t axis, float location) {
    size_t leftAmount = 0, rightAmount = 0;
    BvhNode nodeLeft{}, nodeRight{};

    for (size_t index : indices) {
        Triangle tri = triangles[index];
        float center = (axisMin(tri, axis) + axisMax(tri, axis)) / 2;

        if (center < location) {
            nodeLeft.expand(tri);
            ++leftAmount;
        } else {
            nodeRight.expand(tri);
            ++rightAmount;
        }
    }

    return leftAmount * nodeLeft.area() + rightAmount * nodeRight.area();
}

/**
 * Recursively stretches the boundaries of bounding boxes to adapt for motion blur.
 * 
 * Unfortunately, this makes the BVH significantly less efficient for large
 * motion blur values, as the aabb effectively needs to contain the whole motion.
 * 
 * This is done as a separate step to keep the main BVH construction algorithm simple.
 */
void BvhBuilder::applyMotionBlur(size_t nodeIdx) {
    BvhNode &node = bvhList[nodeIdx];

    if (node.amt != 0) {
        // if it's a leaf, then recalculate bounding box based on motion blur vector
        size_t offset = node.idx;

        // get the mesh material from the first triangle
        glm::vec3 motionBlur = materials[triangles[offset].materialIdx].motionBlur;

        // the objects are offset from 0 * motionBlur to 1 * motionBlur
        node.min = min(node.min, node.min + motionBlur);
        node.max = max(node.max, node.max + motionBlur);
        
    } else {
        size_t child = node.idx;

        // recurse downwards
        applyMotionBlur(child + 0);
        applyMotionBlur(child + 1);

        // adapt the current node's size based in its children
        node.min = min(node.min, bvhList[child + 0].min);
        node.max = max(node.max, bvhList[child + 0].max);
        node.min = min(node.min, bvhList[child + 1].min);
        node.max = max(node.max, bvhList[child + 1].max);
    }
}

/** Apply the ordering of indices to a vector of items. */
void BvhBuilder::applyOrdering(std::vector<Triangle>& items, const std::vector<uint>& indices) {
    size_t const size = items.size();

    std::vector<Triangle> sorted{size};
    for (size_t idx = 0; idx != size; ++idx) {
        sorted[idx] = items[indices[idx]];
    }

    items.swap(sorted);
}

