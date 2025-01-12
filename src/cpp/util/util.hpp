#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <glm/glm.hpp>

std::vector<char> readFile(const std::string& filename);

inline glm::vec3 toVec(std::array<float, 3> const array) {
    return {array[0], array[1], array[2]};
}

inline glm::vec2 toVec(std::array<float, 2> const array) {
    return {array[0], array[1]};
}

inline glm::ivec2 toVec(std::array<int, 2> const array) {
    return {array[0], array[1]};
}

inline glm::uvec2 toVec(std::array<uint, 2> const array) {
    return {array[0], array[1]};
}

inline std::vector<char const *> toCStr(std::vector<std::string> const &strings) {
    std::vector<char const *> cStrs{};
    for (auto const &str : strings) {cStrs.push_back(str.c_str());}
    return cStrs;
}

inline glm::vec3 operator/(glm::vec3 left, float right) {
    return left / glm::vec3(right);
}
