#include "scene.hpp"

gpu::vec3 Triangle::minBound() const {
    if (isSphere) {
        return vertices[0] + gpu::vec3(vertices[1][0],vertices[1][0],vertices[1][0]);
    } else {
        return gpu::min(gpu::min(vertices[0], vertices[1]), vertices[2]);
    }
}

gpu::vec3 Triangle::maxBound() const {
    if (isSphere) {
        return vertices[0] - gpu::vec3(vertices[1][0],vertices[1][0],vertices[1][0]);
    } else {
        return gpu::max(gpu::max(vertices[0], vertices[1]), vertices[2]);
    }
    
}

/* EXAMPLE CONFIG:
 *
 *  version: 0.1
 * 
 *  camera:
 *    resolution: [300, 400]
 *    location: [1, 2, 3.4]
 *    rotation: [0, 90, 0]  # in degrees for each axys
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

const std::string CONFIG_VERSION = "0.1";

Scene::Scene(std::filesystem::path path)
:
    root(YAML::LoadFile(path))
{
    validateFile();
    loadCameraControls();
    loadMeshes();
    buildBVH();
}

std::tuple<size_t, size_t, size_t> Scene::getBufferSizes() {
    // todo: perhaps find a more efficient way to calculate them
    BufferBuilder bvhBuf, matBuf, triBuf;
    for (auto node : components.bvh)
        bvhBuf.append(node);

    for (auto material : components.materials)
        matBuf.append(material);

    for (auto triangle : components.triangles)
        triBuf.append(triangle);

    return std::make_tuple(bvhBuf.getOffset(), matBuf.getOffset(), triBuf.getOffset());
}

CameraControlsUniform Scene::getCameraControls() {
    // return components.camera;

    return {
        .resolution = gpu::vec2(900, 900),
        .focalLength = 1.0,
        .focusDistance = 4.8,
        .apertureRadius = 0.0,
        .location = gpu::vec4(0, 1, 3, 0),
        .rotation = rotate(glm::identity<glm::mat4>(), 3.14f, glm::vec3(0, 1, 0)), // glm::identity<glm::mat4>(),
    };
}

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
}

void Scene::loadCameraControls() {
    typedef std::string str;

    // todo: load them!
}

void Scene::writeBuffers(void *memory) {
    BufferBuilder memoryBuf;
    for (auto node : components.bvh)
        memoryBuf.append(node);

    for (auto material : components.materials)
        memoryBuf.append(material);

    for (auto triangle : components.triangles)
        memoryBuf.append(triangle);

    memoryBuf.write(memory);
}

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

void Scene::loadTriMesh(YAML::Node mesh) {
    auto verts = mesh["data"]["vertices"].as<std::vector<float>>();
    auto norms = mesh["data"]["normals"].as<std::vector<float>>();

    // this is guaranteed by validateFile
    size_t triangleAmt = verts.size() / 9;

    // append the material 
    uint materialIdx = components.materials.size();
    components.materials.push_back(getMaterial(mesh["material"]));

    // append the triangles
    for (size_t i = 0; i < triangleAmt; ++i) {
        Triangle tri{
            .materialIdx = materialIdx,
            .isSphere = false
        };

        for (size_t j = 0; j < 3; j++) {
            size_t off = i * 9 + j * 3;
            tri.vertices[j] = gpu::vec3(verts[off], verts[off+1], verts[off+2]);
            tri.normals[j] = gpu::vec3(norms[off], norms[off+1], norms[off+2]);
        }

        components.triangles.push_back(tri);
    }
}

void Scene::loadSphere(YAML::Node sphere) {
    // append the material
    uint materialIdx = components.materials.size();
    components.materials.push_back(getMaterial(sphere["material"]));

    auto center = sphere["data"]["center"].as<std::vector<gpu::f32>>();

    Triangle tri{};
    tri.vertices[0] = gpu::vec3(center[0], center[1], center[2]);
    tri.vertices[1] = gpu::vec3(sphere["data"]["radius"].as<float>(), 0, 0);
    tri.materialIdx = materialIdx;
    tri.isSphere = true;

    // append the sphere. Unfortunately this wastes an entire triangle.
    components.triangles.push_back(tri);
}

void Scene::buildBVH() {
    std::cout << "building BVH..." << "\n";
    components.bvh = BvhBuilder(components.triangles).buildBvh();
    std::cout << "done!" << "\n";
}

Material Scene::getMaterial(YAML::Node node) {
    // default color value
    std::array<float, 3> def = {0, 0, 0};

    // return the material, with all values defaulting to 0 / false if missing
    return {
        .baseColor =        node["base_color"].as<std::array<gpu::f32, 3>>(def),
        .emission =         node["emission"].as<std::array<gpu::f32, 3>>(def),
        .reflectiveness =   node["reflectiveness"].as<gpu::f32>(0.0),
        .roughness =        node["roughness"].as<gpu::f32>(0.0),
        .ior =              node["ior"].as<gpu::f32>(0.0),
        .isGlass =          node["is_glass"].as<bool>(false),
        .shadeSmooth =      node["smooth_shading"].as<bool>(false),
    };
}

void assertTrue(bool value) {
    if (!value) {
        throw std::runtime_error("The configuration has one or more mistakes.");
    }
}