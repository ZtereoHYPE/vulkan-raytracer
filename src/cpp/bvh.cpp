#include "bvh.hpp"
#include <iterator>
#include <algorithm>

float axisMin(Triangle tri, size_t axis) {
    float min = tri.vertices[0][axis];
    for (int vtx = 1; vtx < 3; ++vtx) {
        float val = tri.vertices[vtx][axis];
        if (val < min) min = val;
    }

    return min;
}

float axisMax(Triangle tri, size_t axis) {
    float max = tri.vertices[0][axis];
    for (int vtx = 1; vtx < 3; ++vtx) {
        float val = tri.vertices[vtx][axis];
        if (val > max) max = val;
    }

    return max;
}

float BvhNode::area() {
    // if the node is uninitialized / expanded with 0 trianges thne this yields NaN
    if (max_amt.max == MIN_VAL || min_idx.min == MAX_VAL) return 0;
    gpu::vec3 len = max_amt.max - min_idx.min;
    return 2 * (len[0] * len[1] + len[0] * len[2] + len[1] * len[2]);
}

void BvhNode::expand(Triangle tri) {
    min_idx.min = gpu::min(min_idx.min, tri.minBound());
    max_amt.max = gpu::max(max_amt.max, tri.maxBound());
}

void BvhNode::initialize(std::vector<Triangle> &triangles, std::span<uint> &indices, uint offset) {
    // Expand the node's bounds
    for (auto idx : indices) 
        expand(triangles[idx]);

    // Populate the various fields
    min_idx.idx.idx = offset;           // offset into the buffer
    max_amt.amt.amt = indices.size();   // amt size => leaf
}

std::vector<BvhNode> BvhBuilder::buildBvh() {
    // Create triangle index list with linear sequence (0,1,2...)
    std::vector<uint> indices(triangles.size());
    std::iota(indices.begin(), indices.end(), 0);

    bvhList.push_back(BvhNode());
    buildRecursively(0, std::span(indices), 0, 0, 1e30);
    applyMotionBlur(0);

    // Applies the indices to the triangles. This avoids lots of memory movement
    // while the BVH is being built, and significantly speeds the process up.
    applyOrdering(triangles, indices);

    return bvhList;
}

/**
 * Builds the BVH by perfroming a recursive algorithm
 *
 * Inspired by https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
 */
void BvhBuilder::buildRecursively(size_t nodeIdx, std::span<uint> indices, uint depth, uint offset, float parentCost) {
    // Initialize the node's data.
    bvhList[nodeIdx].initialize(triangles, indices, offset);

    if (depth >= BHV_MAX_DEPTH || indices.size() <= 1) return;

    auto [splitAxis, splitPos] = findBestSplit(indices);
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
            swap(indices[leftIdx], indices[rightIdx--]);
        }
    }

    // Avoid creating empty nodes
    if ((leftIdx == 0) || (rightIdx == indices.size() - 1)) return;

    // Add the children indices and recurse down
    uint leftNodeIdx = bvhList.size();

    bvhList[nodeIdx].min_idx.idx.idx = leftNodeIdx; // child index
    bvhList[nodeIdx].max_amt.amt.amt = 0;           // not a leaf

    BvhNode leftNode{}, rightNode{};
    bvhList.push_back(leftNode);
    bvhList.push_back(rightNode);

    buildRecursively(leftNodeIdx, indices.subspan(0, leftIdx), depth + 1, offset, bestCost);
    buildRecursively(leftNodeIdx + 1, indices.subspan(leftIdx, indices.size() - leftIdx), depth + 1, offset + leftIdx, bestCost);
}

std::tuple<size_t, float> BvhBuilder::findBestSplit(std::span<uint> &indices) {
    // Use the SAH to find the best split position by trying to split at every single center
    size_t splitAxis = -1;
    float splitPos;
    float bestCost = 1e30;
    for (size_t axis : {0, 1, 2}) {
        for (uint triIdx : indices) {
            Triangle tri = triangles[triIdx];
            float center = (axisMin(tri, axis) + axisMax(tri, axis)) / 2;
            float cost = splitCost(indices, axis, center);

            if (cost < bestCost) {
                bestCost = cost;
                splitAxis = axis;
                splitPos = center;
            }
        }
    }

    if (splitAxis == -1)
        throw std::runtime_error("the axys has not been set");

    return std::make_tuple(splitAxis, splitPos);
}

/*
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
 */
void BvhBuilder::applyMotionBlur(size_t nodeIdx) {
    BvhNode &node = bvhList[nodeIdx];

    if (node.max_amt.amt.amt != 0) {
        // if it's a leaf, then recalculate bounding box based on motion blur vector
        size_t offset = node.min_idx.idx.idx;
        size_t amount = node.max_amt.amt.amt;

        gpu::vec3 motionBlur = materials[triangles[offset].materialIdx].motionBlur;

        for (size_t idx = offset; idx < amount + offset; ++idx) {
            Triangle tri = triangles[idx];

            // the rays are distributed from 0 * motionBlur to 1 * motionBlur
            node.min_idx.min = gpu::min(node.min_idx.min, tri.minBound() + motionBlur);
            node.max_amt.max = gpu::max(node.max_amt.max, tri.maxBound() + motionBlur);
        }
        
    } else {
        size_t child = node.min_idx.idx.idx;

        // recurse downwards
        applyMotionBlur(child + 0);
        applyMotionBlur(child + 1);

        // adapt the current node's size based in its children
        node.min_idx.min = gpu::min(node.min_idx.min, bvhList[child + 0].min_idx.min);
        node.max_amt.max = gpu::max(node.max_amt.max, bvhList[child + 1].min_idx.min);
    }
}

void BvhBuilder::applyOrdering(std::vector<Triangle>& items, const std::vector<uint>& indices) {
    size_t const size = items.size();

    std::vector<Triangle> sorted{size};
    for (size_t idx = 0; idx != size; ++idx) {
        sorted[idx] = items[indices[idx]];
    }

    items.swap(sorted);
}

void BvhBuilder::swap(uint &left, uint &right) {
    uint tmp = left;
    left = right;
    right = tmp;
}
