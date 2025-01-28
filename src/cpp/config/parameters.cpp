#include "parameters.hpp"

Parameters params; // instantiate global

/** Constructor that assigns all the parameters from the file */
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
      SCENE_FILE(root["scene_file"].as<std::string>()),
      SHADER_DIR(root["shader_dir"].as<std::string>()),
      HEADLESS(root["offscreen_rendering"].as<bool>()),
      DUMP_FILE(root["dump_file"].as<std::string>()),
      FRAME_COUNT(root["frame_count"].as<int>())
{};