#version 460

#extension GL_GOOGLE_include_directive : enable
#include "imports/ray.frag"

struct HitRecord {
    bool has_hit;
    vec3 pos;
    vec3 normal;
    float t;
};

struct Sphere {
    float radius;
    vec3 center;
};

// todo: look into inline uniform buffers for speed and small data
// todo: alternatively, look into push constants
layout(std140, set = 0, binding = 0) uniform CameraUBO {
    vec2 resolution;
    vec2 uv;
    float focal_len;
    vec3 origin;
} camera;

#define SAMPLES 8

layout(std140, set = 0, binding = 1) readonly buffer SphereSBO {
    uint count;
    Sphere spheres[];
} sphere_sbo;

layout(location = 0) out vec4 out_color;

Ray calculate_ray(vec2 frag_coord) {

    /*  We are starting from -uv/2 on each axys, and we need to reach +uv/2.
        Expressed as a formula:
            -uv/2 + pixel_delta * res = +uv/2
            pixel_delta = uv / res

        The Y component is inverted because as the pixels' y axys grows in value
        we want to descend in the 3D space.

        Some of this could be moved to the CPU as a uniform, but is left here
        for now for clarity and because the performance impact is minor.
    */
    vec2 pixel_delta = camera.uv/camera.resolution;
    pixel_delta *= vec2(1, -1);

    vec3 pixel_origin = vec3(-camera.uv.x/2, camera.uv.y/2, camera.focal_len) + // the top left is at x=-u but y=v (up is +)
                        camera.origin +  // to place it correctly in world-space
                        vec3(pixel_delta/2, 0);  // to center the pixel
                        
    vec3 pixel_center = pixel_origin + vec3(pixel_delta * frag_coord, 0);

    Ray ray;
    ray.direction = normalize(pixel_center - camera.origin); // vector from origin to center
    ray.origin = camera.origin;

    return ray;
}

vec3 background_color(Ray ray) {
    float blend = 0.5 * ray.direction.y + 0.5;
    return mix(vec3(0.6, 0.8, 1.0), vec3(0.2, 0.4, 1.0), blend);
}

HitRecord hit_sphere(Sphere sphere, Ray ray) {
    /*  line (parametric):
            x = orig_x + dir_x * t
            y = orig_y + dir_y * t   ->   p = orig + dir * t
            z = orig_z + dir_z * t

        sphere formula:
            (x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2 <= radius^2    ->    (p - center).(p - center) <= radius^2

        now we plug the line in the sphere formula and solve for t:
            (orig + dir * t - center).(orig + dir * t - center) <= r^2

            [...]

            a = dir.dir
            b = 2dir.(orig - center)
            c = (center - orig).(center - orig) - radius^2

        referring to the quadratic formula, we can tell whether the ray intersected the sphere by the discriminant:
            hit_sphere = (b^2 - 4ac >= 0)
    */
    vec3 delta = (sphere.center - ray.origin);

    // todo: apply the simpler formula for b multiples of 2
    float a = dot(ray.direction, ray.direction);
    float b = -2 * dot(ray.direction, delta);
    float c = dot(delta, delta) - sphere.radius * sphere.radius;

    float discriminant = b*b - 4*a*c;
    // todo: is there a better way to do the sqrt here? is it even a bottleneck?
    float t = (-b - sqrt(discriminant)) / 2*a;

    HitRecord hit;
    hit.t = t;
    hit.has_hit = !(discriminant < 0 || t < 0);
    hit.pos = ray_at(ray, t);
    hit.normal = normalize(hit.pos - sphere.center);

    return hit;
}

void main() {
    Ray ray = calculate_ray(gl_FragCoord.xy);

    HitRecord closest_hit;
    closest_hit.t = 1.0 / 0.0; // infinity
    closest_hit.has_hit = false;

    for (int i = 0; i < sphere_sbo.count; i++) {
        HitRecord hit = hit_sphere(sphere_sbo.spheres[i], ray);

        if (hit.has_hit && hit.t < closest_hit.t) {
            closest_hit = hit;
        }
    }
    
    if (closest_hit.has_hit) {
        out_color = vec4(closest_hit.normal*0.5 + vec3(0.5), 1.0);
    } else {
        out_color = vec4(background_color(ray), 1.0);
    }
}
