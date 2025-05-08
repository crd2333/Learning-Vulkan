#version 450

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

// 因为创建顶点缓冲比较复杂，先把顶点数据硬编码在着色器中，之后再讨论
void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0); // 内置的 gl_VertexIndex 变量包含当前顶点的索引
    fragColor = colors[gl_VertexIndex];
}
