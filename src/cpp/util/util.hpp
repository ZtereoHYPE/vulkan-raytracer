#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>
#include <glm/glm.hpp>

std::vector<char> readFile(const std::string& filename);

/**
 * Conversion function to convert arrays of numbers to vectors.
 */
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

/** Function to convert a vector of std::string's into c strings (char *) */
inline std::vector<char const *> toCStr(std::vector<std::string> const &strings) {
    std::vector<char const *> cStrs{};
    for (auto const &str : strings) {cStrs.push_back(str.c_str());}
    return cStrs;
}

/** Operator overload to divide a vector by a float */
inline glm::vec3 operator/(glm::vec3 left, float right) {
    return left / glm::vec3(right);
}
