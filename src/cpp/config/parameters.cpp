#include "parameters.hpp"

Parameters params; // instantiate global

Parameters::Parameters(std::filesystem::path path)
:
      root(YAML::LoadFile(path)),
      validationLayersMemory(root["validation_layers"].as<std::vector<std::string>>()),

      USE_LLVMPIPE(root["use_llvmpipe"].as<bool>()),
      USE_VALIDATION_LAYERS(root["use_validation_layers"].as<bool>()),
      VALIDATION_LAYERS(toCStr(validationLayersMemory)),
      MAX_BVH_DEPTH(root["max_bvh_depth"].as<int>()),
      BVH_SPLIT_ATTEMPTS(root["bvh_split_attempts"].as<int>()),
      TILE_SIZE(root["tile_size"].as<int>()),
      SCENE_FILE(root["scene_file"].as<std::string>())
{};