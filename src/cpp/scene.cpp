#include "scene.hpp"

/* EXAMPLE CONFIG:
 *
 *  version: 0.1
 *  scene:
 *      - Mesh Name:
 *          type: TriMesh
 *          material:
 *              base_color: [0.7, 0.7, 0.7]
 *          data:
 *              vertices: [0.3, 0.5, -4, ...]
 *              normals: [0, 1, 0, ...]  
 *
 *      - Sun:
 *          type: Sphere
 *          material:
 *              base_color: [1, 1, 1]
 *              emission: [10, 10, 10]
 *          data:
 *              center: [100, 100, 100]
 *              radius: 50
 */

const std::string CONFIG_VERSION = "0.1";

Scene::Scene(std::filesystem::path path)
:
    sceneFile(YAML::LoadFile(path))
{
    validateFile();
}

std::pair<size_t, size_t> Scene::getBufferSizes() {
    BufferBuilder meshes, triangles;

    // quite expensive but also guaranteed not to break
    populateBuffers(meshes, triangles);
    return std::pair(meshes.getOffset(), triangles.getOffset());
}

void Scene::populateBuffers(BufferBuilder &meshes, BufferBuilder &triangles) {
    typedef std::string str;

    meshes.append((gpu::u32)sceneFile["scene"].size()); // mesh_count
    meshes.pad(12); // alignment rules

    for (auto mesh : sceneFile["scene"]) {
        if (mesh["type"].as<str>() == "TriMesh") {
            populateTriMesh(mesh, meshes, triangles);

        } else if (mesh["type"].as<str>() == "Sphere") {
            populateSphere(mesh, meshes, triangles);
        }
    }
}

void Scene::validateFile() {
    typedef std::string str;

    // check the version is correct
    if (sceneFile["version"].as<str>() != CONFIG_VERSION) {
        throw std::runtime_error("Scene file is of incompatible version!");
    }

    // check that the data fields in each mesh match the type
    for (auto mesh : sceneFile["scene"]) {
        if (mesh["type"].as<str>() == "TriMesh") {
            assertTrue(mesh["data"]["vertices"].IsSequence());
            assertTrue(mesh["data"]["normals"].IsSequence());

            // make sure the size of vertices and normals are correct
            auto vertices = mesh["data"]["vertices"].as<std::list<float>>();
            auto normals = mesh["data"]["normals"].as<std::list<float>>();

            assertTrue(vertices.size() == normals.size());
            assertTrue(vertices.size() % 9 == 0);
            assertTrue(vertices.size() != 0);

        } else if (mesh["type"].as<str>() == "Sphere") {
            assertTrue(mesh["data"]["center"].IsSequence());
            assertTrue(mesh["data"]["radius"].IsScalar());
        }
    }
}

void Scene::populateTriMesh(YAML::Node mesh, BufferBuilder &meshes, BufferBuilder &triangles) {
    Material meshMaterial = getMaterial(mesh["material"]);
    auto verts = mesh["data"]["vertices"].as<std::vector<float>>();
    auto norms = mesh["data"]["normals"].as<std::vector<float>>();

    // this is ensured with validateFile
    gpu::u32 triangleAmt = verts.size() / 9;

    // append the mesh to the mesh info buffer
    meshes.append<Mesh>({
        .triangle_count = triangleAmt,
        .offset = (gpu::u32) triangles.getRelativeOffset<Triangle>(),
        .material = meshMaterial
    });

    // append the triangles
    for (size_t i = 0; i < triangleAmt; ++i) {
        Triangle tri;

        for (int j = 0; j < 3; j++) {
            int off = i * 9 + j * 3;
            tri.vertices[j] = gpu::vec3(verts[off], verts[off+1], verts[off+2]);
            tri.normals[j] = gpu::vec3(norms[off], norms[off+1], norms[off+2]);
        }

        triangles.append(tri);
    }
}

void Scene::populateSphere(YAML::Node mesh, BufferBuilder &meshes, BufferBuilder &triangles) {
    Material meshMaterial = getMaterial(mesh["material"]);

    // append the mesh to the mesh info buffer
    meshes.append<Mesh>({
        .triangle_count = 0,
        .offset = (gpu::u32) triangles.getRelativeOffset<Triangle>(),
        .material = meshMaterial
    });

    auto center = mesh["data"]["center"].as<std::vector<gpu::f32>>();

    // append the sphere
    triangles.append<Sphere>({
        .center = gpu::vec3(center[0], center[1], center[2]),
        .radius = mesh["data"]["radius"].as<float>(),
    }, sizeof(Triangle));
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

