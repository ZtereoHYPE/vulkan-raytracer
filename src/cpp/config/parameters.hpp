#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../util/util.hpp"

/**
 * Class containing all the program's tweakable parameters.
 * This is loaded at runtime from a file allowing easy deployment and modification.
 */
class Parameters {
    YAML::Node root;
    std::vector<std::string> validationLayersMemory; // required as c-style strings are just pointers

   public:
    // Vulkan parameters
    bool const USE_LLVMPIPE;
    bool const USE_VALIDATION_LAYERS;
    std::vector<char const *> const VALIDATION_LAYERS;

    // BVH parameters
    int const MAX_BVH_DEPTH;
    int const BVH_SPLIT_ATTEMPTS; // set to -1 for trying at every possible position

    // Rendering parameters
    int const TILE_SIZE;
    std::string SCENE_FILE;

    explicit Parameters(std::filesystem::path path = "parameters.yaml");
};

/** Global instance of parameters accessible to any file including this header */
extern Parameters params;