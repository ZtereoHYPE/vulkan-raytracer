#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <numeric>
#include <iostream>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

#include "../util/buffer-builder.hpp"
#include "../util/gpu-types.hpp"
#include "../bvh.hpp"

struct Material {
    gpu::vec3 baseColor;
    gpu::vec3 emission;
    gpu::f32 reflectiveness;
    gpu::f32 roughness;
    gpu::f32 ior;
    gpu::boolean isGlass;
    gpu::boolean shadeSmooth;
    gpu::vec3 motionBlur;
};

struct Triangle {
    gpu::u32 materialIdx;
    gpu::boolean isSphere;
    gpu::vec3 vertices[3];
    gpu::vec3 normals[3];

    // todo: these could be cached for much faster BVH building
    gpu::vec3 minBound() const;
    gpu::vec3 maxBound() const;
};

struct CameraControlsUniform {
    gpu::uvec2 resolution;
    gpu::vec2 viewportUv;
    gpu::f32 focalLength;
    gpu::f32 focusDistance;
    gpu::f32 apertureRadius;
    gpu::u32 time;
    gpu::vec3 location;
    glm::mat4 rotation;
};

struct BvhNode; // forward declaration

struct SceneComponents {
    CameraControlsUniform camera;

    std::vector<BvhNode> bvh;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
};

class Scene {
    YAML::Node root;
    SceneComponents components;

public:
    explicit Scene(std::filesystem::path path = "scene.yaml");

    /* Return the size of the BHV, Material, and Triangle buffers respectively */
    std::tuple<size_t, size_t, size_t> getBufferSizes();

    /* Return the camera information stored in the config, if any. */
    CameraControlsUniform getCameraControls();

    /* Write the scene to memory */
    void writeBuffers(void *memory);

private:
    /* This method performs some very basic validation on the scene file */
    void validateFile();

    /* Load the camera controls from the file */
    void loadCameraControls();

    /* These methods handle the loading of the various meshes and their triangles */
    void loadMeshes();
    void loadTriMesh(YAML::Node trimesh);
    void loadSphere(YAML::Node sphere);

    /* This method builds a BVH for the scene */
    void buildBVH();

    /* Returns the Material object from the current material node */
    static Material getMaterial(YAML::Node);
};

inline void assertTrue(bool value); // used in validation

template<typename T>
void applyOrdering(std::vector<T> &items, const std::vector<uint>& ordering);
gpu::vec3 vecMax(gpu::vec3 left, gpu::vec3 right);
gpu::vec3 vecMin(gpu::vec3 left, gpu::vec3 right);
