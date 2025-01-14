#include "scene.hpp"

const std::string CONFIG_VERSION = "0.2";

/**
 * Triangle method to get its minimum bound.
 * Adapts to whether the triangle actually represents a sphere.
 */
glm::vec3 Triangle::minBound() const {
    if (isSphere) {
        const float radius = vertices[1][0];
        return vertices[0] - glm::vec4(radius, radius, radius, 0);
    } else {
        return glm::min(glm::min(vertices[0], vertices[1]), vertices[2]);
    }
}

/**
 * Triangle method to get its maximum bound.
 * Adapts to whether the triangle actually represents a sphere.
 */
glm::vec3 Triangle::maxBound() const {
    if (isSphere) {
        const float radius = vertices[1][0];
        return vertices[0] + glm::vec4(radius, radius, radius, 0);
    } else {
        return glm::max(glm::max(vertices[0], vertices[1]), vertices[2]);
    }
    
}

/**
 * Construct a scene by validating, loading the file, and building a BVH.
 */
Scene::Scene(std::filesystem::path path)
:
    root(YAML::LoadFile(path))
{
    validateFile();
    loadCameraControls();
    loadMeshes();
    buildBVH();
}

/* Return the size of the BHV, Material, and Triangle buffers respectively */
std::tuple<size_t, size_t, size_t> Scene::getBufferSizes() {
    BufferBuilder bvhBuf, matBuf, triBuf;
    for (auto node : bvh)
        bvhBuf.append(node);

    for (auto material : materials)
        matBuf.append(material);

    for (auto triangle : triangles)
        triBuf.append(triangle);

    return std::make_tuple(bvhBuf.getOffset(), matBuf.getOffset(), triBuf.getOffset());
}

/* Return the camera information stored in the config, if any. */
CameraControls Scene::getCameraControls() {
    return cameraControls;
}

/* Write the scene to memory */
void Scene::writeBuffers(void *memory) {
    BufferBuilder memoryBuf;
    for (auto node : bvh)
        memoryBuf.append(node);

    for (auto material : materials)
        memoryBuf.append(material);

    for (auto triangle : triangles)
        memoryBuf.append(triangle);

    memoryBuf.write(memory);
}

/* This method performs some very basic validation on the scene file */
void Scene::validateFile() {
    typedef std::string str;

    // check the version is correct
    if (root["version"].as<str>() != CONFIG_VERSION) {
        throw std::runtime_error("Scene file is of incompatible version!");
    }

    // check that the data fields in each mesh match the type
    for (auto mesh : root["scene"]) {
        if (mesh["type"].as<str>() == "TriMesh") {
            assertTrue(mesh["data"]["vertices"].IsSequence());
            assertTrue(mesh["data"]["normals"].IsSequence());

            // make sure the size of vertices and normals are correct
            auto vertices = mesh["data"]["vertices"].as<std::list<float>>();
            auto normals = mesh["data"]["normals"].as<std::list<float>>();

            assertTrue(vertices.size() == normals.size());
            assertTrue(vertices.size() % 9 == 0);
            assertTrue(!vertices.empty());

        } else if (mesh["type"].as<str>() == "Sphere") {
            assertTrue(mesh["data"]["center"].IsSequence());
            assertTrue(mesh["data"]["radius"].IsScalar());
        }
    }

    // ensure that the camera data exists
    auto camera = root["camera"];
    assertTrue(camera["resolution"].IsSequence());
    assertTrue(camera["focal_length"].IsScalar());
    assertTrue(camera["focus_distance"].IsScalar());
    assertTrue(camera["aperture_radius"].IsScalar());
    assertTrue(camera["location"].IsSequence());
    assertTrue(camera["rotation"].IsSequence());

    assertTrue(camera["resolution"].size() == 2);
    assertTrue(camera["location"].size() == 3);
    assertTrue(camera["rotation"].size() == 3);
}

/* Load the camera controls from the file */
void Scene::loadCameraControls() {
    auto camera = root["camera"];

    // Load constant parameters
    CameraControls ubo {
        .resolution = toVec(camera["resolution"].as<std::array<uint, 2>>()),
        .focalLength =      camera["focal_length"].as<float>(),
        .focusDistance =    camera["focus_distance"].as<float>(),
        .apertureRadius =   camera["aperture_radius"].as<float>(),
        .location =   toVec(camera["location"].as<std::array<float, 3>>()),
    };

    // Calculate rotation matrix
    std::array<float, 3> rotations = camera["rotation"].as<std::array<float, 3>>();
    glm::mat4 rotMatrix = glm::eulerAngleXYZ(
        glm::radians(rotations[0]),
        glm::radians(rotations[1]),
        glm::radians(rotations[2])
    );
    ubo.rotation = rotMatrix;

    // Calculate UV
    float ratio = ubo.resolution[0] / (float) ubo.resolution[1];
    float u, v;
    if (ratio > 1) {
        u = ratio;
        v = 1;
    } else {
        u = 1;
        v = 1/ratio;
    }
    ubo.viewportUv = glm::vec2(u, v);

    cameraControls = ubo;
}

/** Load meshes from the yaml file to the triangle vector */
void Scene::loadMeshes() {
    typedef std::string str;

    for (auto mesh : root["scene"]) {
        if (mesh["type"].as<str>() == "TriMesh") {
            loadTriMesh(static_cast<YAML::Node>(mesh));

        } else if (mesh["type"].as<str>() == "Sphere") {
            loadSphere(static_cast<YAML::Node>(mesh));
        }
    }
}

/** Load triangle meshes */
void Scene::loadTriMesh(YAML::Node mesh) {
    auto verts = mesh["data"]["vertices"].as<std::vector<float>>();
    auto norms = mesh["data"]["normals"].as<std::vector<float>>();

    // this is guaranteed by validateFile
    size_t triangleAmt = verts.size() / 9;

    // append the material 
    uint materialIdx = materials.size();
    materials.push_back(getMaterial(mesh["material"]));

    // append the triangles
    for (size_t i = 0; i < triangleAmt; ++i) {
        Triangle tri {
            .materialIdx = materialIdx,
            .isSphere = false,
        };
        for (size_t j = 0; j < 3; j++) {
            size_t off = i * 9 + j * 3;
            tri.vertices[j] = glm::vec4(verts[off], verts[off+1], verts[off+2], 0);
            tri.normals[j] = glm::vec4(norms[off], norms[off+1], norms[off+2], 0);
        }

        triangles.push_back(tri);
    }
}

/** Load spheres by containing them in a triangle */
void Scene::loadSphere(YAML::Node sphere) {
    // append the material
    uint materialIdx = materials.size();
    materials.push_back(getMaterial(sphere["material"]));

    auto center = sphere["data"]["center"].as<std::vector<float>>();

    Triangle tri{};
    tri.vertices[0] = glm::vec4(center[0], center[1], center[2], 0);
    tri.vertices[1] = glm::vec4(sphere["data"]["radius"].as<float>(), 0, 0, 0);
    tri.materialIdx = materialIdx;
    tri.isSphere = true;

    // append the sphere. Unfortunately this wastes an entire triangle.
    triangles.push_back(tri);
}

/* Build a BVH for the scene */
void Scene::buildBVH() {
    using namespace std::chrono;
    std::cout << "building BVH..." << "\n";

    auto pre = high_resolution_clock::now();
    bvh = BvhBuilder(triangles, materials).build();
    auto post = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(post - pre);
    std::cout << "done! (" << duration.count() << "ms)\n";
}

/* Returns the Material object from the current material node */
Material Scene::getMaterial(YAML::Node node) {
    // default color value
    std::array<float, 3> def = {0, 0, 0};

    // return the material, with all values defaulting to 0 / false if missing
    return {
        .baseColor =  toVec(node["base_color"].as<std::array<float, 3>>(def)),
        .emission =   toVec(node["emission"].as<std::array<float, 3>>(def)),
        .reflectiveness =   node["reflectiveness"].as<float>(0.0),
        .roughness =        node["roughness"].as<float>(0.0),
        .ior =              node["ior"].as<float>(0.0),
        .isGlass =          node["is_glass"].as<bool>(false),
        .shadeSmooth =      node["smooth_shading"].as<bool>(false),
        .motionBlur = toVec(node["motion_blur"].as<std::array<float, 3>>(def)),
    };
}

/** Helper function used for file validation */
void assertTrue(bool value) {
    if (!value) {
        throw std::runtime_error("The configuration has one or more mistakes.");
    }
}