#version 450

layout(binding = 1) uniform sampler2D texSampler; // 采样器描述符在 glsl 中由 uniform 表示

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord); // 使用内置的 texture 函数进行采样
}
