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

    // Applies the indices to the triangles. This avoids lots of memory movement
    // while the BVH is being built, and significantly speeds the process up.
    applyOrdering(triangles, indices);

    return bvhList;
}

/**
* Builds the thing by perfroming a recursive algorithm!!1
*
* Inspired by https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
*/
void BvhBuilder::buildRecursively(size_t nodeIdx, std::span<uint> indices, uint depth, uint offset, float parentCost) {
    // Initialize the node's data.
    bvhList[nodeIdx].initialize(triangles, indices, offset);

    if (depth >= BHV_MAX_DEPTH || indices.size() <= 1) return;

    //printf(" bounds %g %g %g > %g %g %g\n", bvhList[nodeIdx].min[0],bvhList[nodeIdx].min[1],bvhList[nodeIdx].min[2],bvhList[nodeIdx].max[0],bvhList[nodeIdx].max[1],bvhList[nodeIdx].max[2]);

    //std::cout << "idx amt: " << indices.size() << "\n";

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

            //std::cout << "cost/axis/pos: " << cost << "/" << axis << "/" << center << "\n";

            if (cost < bestCost) {
                bestCost = cost;
                splitAxis = axis;
                splitPos = center;
            }
        }
    }

    if (splitAxis == -1)
        throw std::runtime_error("the axys has not been set");

    // Partition all the objects based on whether they are closer to left or right
    //gpu::vec3 boundSizes = node.max - node.min;
    //size_t splitAxis = std::distance(boundSizes.values.begin(), std::max_element(boundSizes.values.begin(), boundSizes.values.end())); 
    //float splitPos = (node.max.values[splitAxis] + node.min.values[splitAxis]) / 2.0;
    //float bestCost = parentCost/2;

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
