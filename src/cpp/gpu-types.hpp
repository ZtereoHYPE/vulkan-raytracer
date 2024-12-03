#pragma once

#include "pch.hpp"

/*
    This is a generic, N-agnostic implementation of the below classes.

    template<typename T, unsigned N, unsigned Align>
    class alignas(Align) vector {
        std::array<T, N> values;

        public: 
            template <typename... Args>
            requires (sizeof...(Args) == N && (std::is_same_v<Args, T> && ...))
            vector(Args... args) : values({args...}) {};

            operator glm::vec<N, T, glm::defaultp>() const {
                return to_glm_vec(std::make_index_sequence<N>{});
            }

        private:
            template <std::size_t... I>
            glm::vec<N, T, glm::defaultp> to_glm_vec(std::index_sequence<I...>) const {
                return glm::vec<N, T, glm::defaultp>(values[I]...); // Unpacks the array
            }
    };
*/

/*
 * Template-based vector classes.
 *
 * These class represents a generic vector class that can adapt to any type,
 * and be converted to the respective glm type for functions such as glm::lookAt.
 * 
 * While a single implementation could be written, the three are kept separate
 * for readability purposes
 */
template<typename T>
struct alignas(8) vector2 {
    std::array<T, 2> values;

    // constructors, set the values in the array
    vector2() = default; // default constructor 0-initializes
    vector2(T x, T y) : values({x, y}) {};
    vector2(std::array<T, 2> xy) : values(xy) {};

    // coonversion operator, allows the type to be converted to glm::vec2
    operator glm::vec<2, T, glm::defaultp>() {
        return glm::vec<2, T, glm::defaultp>(values[0], values[1]);
    };
};

template<typename T>
struct alignas(16) vector3 {
    std::array<T, 3> values;

    vector3() = default;
    vector3(T x, T y, T z) : values({x, y, z}) {};
    vector3(std::array<T, 3> xyz) : values(xyz) {};

    operator glm::vec<3, T, glm::defaultp>() {
        return glm::vec<3, T, glm::defaultp>(values[0], values[1], values[2]);
    };
};

template<typename T>
struct alignas(16) vector4 {
    std::array<T, 4> values;

    vector4() = default;
    vector4(T x, T y, T z, T w) : values({x, y, z, w}) {};
    vector4(std::array<T, 4> xyzw) : values(xyzw) {};

    operator glm::vec<4, T, glm::defaultp>() {
        return glm::vec<4, T, glm::defaultp>(values[0], values[1], values[2], values[3]);
    };
};

/* 
 * GPU Namespace:
 *
 * This namespace contains custom implementations of all of the types 
 * that need to be uploaded to the GPU.
 * 
 * The reason why GLM's types aren't used is because they obey different padding 
 * rules which end up causing alignment issues when uploaded to the GPU.
 */
namespace gpu {
    /* floating point type */
    typedef float f32;

    /* unsigned integer type */
    typedef uint u32;

    /* signed integer type */
    typedef int i32;

    /* boolean type (4-aligned) */
    typedef uint boolean;

    /*
     * Implementations of the various vector types.
     */
    typedef vector2<gpu::f32> vec2;
    typedef vector3<gpu::f32> vec3;
    typedef vector4<gpu::f32> vec4;

    typedef vector2<gpu::u32> uvec2;
    typedef vector3<gpu::u32> uvec3;
    typedef vector4<gpu::u32> uvec4;

    typedef vector2<gpu::i32> ivec2;
    typedef vector3<gpu::i32> ivec3;
    typedef vector4<gpu::i32> ivec4;
}