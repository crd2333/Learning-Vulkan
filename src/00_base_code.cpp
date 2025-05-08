#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
// 把 #include <vulkan/vulkan.h> 替换为上述两行，这样 GLFW 将包含自己的定义并自动加载 Vulkan 头文件

#include <iostream>
#include <stdexcept>
#include <cstdlib>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    void initWindow() {
        glfwInit(); // 初始化 glfw 的第一个调用

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 告诉 glfw 不要创建 OpenGL 上下文（因为其最初是为 OpenGL 设计）
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // 暂时禁用窗口大小调整

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {

    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
