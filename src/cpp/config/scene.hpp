#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <numeric>
#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL // for euler angles
#include <glm/gtx/euler_angles.hpp>

#include "../util/buffer-builder.hpp"
#include "../bvh.hpp"
#include "../util/util.hpp"

/* EXAMPLE CONFIG:
 *
 *  version: 0.2
 *
 *  camera:
 *    resolution: [300, 400]
 *    location: [1, 2, 3.4]
 *    rotation: [0, 90, 0]  # XYZ euler rotation
 *    focal_length: 1.1
 *    focus_distance: 5.4
 *    aperture_radius: 0  # DoF disabled
 *
 *  scene:
 *    - Mesh Name:
 *        type: TriMesh
 *        material:
 *          base_color: [0.7, 0.7, 0.7]
 *        data:
 *          vertices: [0.3, 0.5, -4, ...]
 *          normals: [0, 1, 0, ...]
 *
 *    - Sun:
 *        type: Sphere
 *        material:
 *          base_color: [1, 1, 1]
 *          emission: [10, 10, 10]
 *        data:
 *          center: [100, 100, 100]
 *          radius: 50
 */

/**
 * Material struct aligned and padded to match GLSL version.
 */
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

/**
 * Triangle struct aligned and padded to match GLSL version.
 */
struct Triangle {
    alignas(16) glm::vec4 vertices[3];
    alignas(16) glm::vec4 normals[3];
    uint materialIdx;
    uint isSphere;

    glm::vec3 minBound() const;
    glm::vec3 maxBound() const;
};

/**
 * Camera controls struct aligned and padded to match GLSL version.
 */
struct CameraControls {
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

/**
 * Class that represents and prepares the rendered Scene from a file.
 */
class Scene {
    YAML::Node root;

    CameraControls cameraControls;
    std::vector<BvhNode> bvh;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;

   public:
    explicit Scene(std::filesystem::path path = "scene.yaml");
    std::tuple<size_t, size_t, size_t> getBufferSizes();
    CameraControls getCameraControls();
    size_t writeBuffers(void *memory);

   private:
    void validateFile();
    void loadCameraControls();
    void loadMeshes();
    void loadTriMesh(YAML::Node trimesh);
    void loadSphere(YAML::Node sphere);
    void buildBVH();

    static Material getMaterial(YAML::Node);
};

inline void assertTrue(bool value);
