#version 450

// todo: look into inline uniform buffers for speed and small data
layout(binding = 0) uniform UniformBufferObject {
    uint width;
    uint height;
} viewport;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(gl_FragCoord.x / ubo.width, gl_FragCoord.y / ubo.height, 0.0, 1.0);
}
