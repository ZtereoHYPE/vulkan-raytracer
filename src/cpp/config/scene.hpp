#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <numeric>
#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../util/buffer-builder.hpp"
#include "../bvh.hpp"
#include "../util/util.hpp"

struct Material {
    alignas(16) glm::vec3 baseColor;
    alignas(16) glm::vec3 emission;
    float reflectiveness;
    float roughness;
    float ior;
    uint isGlass; // bool
    uint shadeSmooth; // bool
    alignas(16) glm::vec3 motionBlur;
};

struct Triangle {
    alignas(16) glm::vec4 vertices[3];
    alignas(16) glm::vec4 normals[3];
    uint materialIdx;
    uint isSphere;

    glm::vec3 minBound() const;
    glm::vec3 maxBound() const;
};

struct CameraControlsUniform {
    glm::uvec2 resolution;
    glm::vec2 viewportUv;
    float focalLength;
    float focusDistance;
    float apertureRadius;
    uint time;
    alignas(16) glm::vec3 location;
    alignas(16) glm::mat4 rotation;
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
