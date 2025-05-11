#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {

    // gl_PointCoord 是一个内置变量，表示当前片段在 point sprite 中的坐标（因为 point sprite 是一个有大小的正方形，其大小由 vertex shader 中的 gl_PointSize 指定，坐标范围是 [0, 1]）
    vec2 coord = gl_PointCoord - vec2(0.5);
    // outColor = vec4(fragColor, 0.5);
    outColor = vec4(fragColor, 0.5 - length(coord)); // 这里对 alpha 的设置是为了仅在半径 0.5 之内的范围内显示颜色（alpha < 0 被丢弃），也就是把一个个正方形绘制成圆形
}
