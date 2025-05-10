#version 450

// 使用 UBO 描述的 MVP 矩阵
// binding 指令类似于属性的 location 指令，我们将在描述符布局中引用此绑定
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;
// 实际上可以为管线布局绑定多个描述符集，通过这种方式，可以实现逐对象的 UBO 和全局 UBO
// layout(set = 0, binding = 0) uniform UniformBufferObject {
//     mat4 model;
//     mat4 view;
//     mat4 proj;
// } ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
