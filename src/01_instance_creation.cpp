#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>

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

    VkInstance instance; // Instance 是应用程序和 Vulkan 库之间的连接

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance()
    {
        VkApplicationInfo appInfo{}; // 需要填充一个结构体传输信息，其实是可选的，但可能会向驱动程序提供对优化有用的信息
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Vulkan 的许多结构体要求在 sType 中显式指定类型
        // pNext 通过初始化值设定为 nullptr，可以在将来指向扩展信息
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{}; // 必选的结构体，告诉 Vulkan 驱动程序要使用哪些全局扩展和验证层
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions; // Vulkan 是平台无关的，需要一个扩展来与窗口系统交互
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount); // 通过 glfw 内置函数方便地获取扩展

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0; // 验证层先留空，之后讨论

        // 对象创建函数如 vkCreateInstance，其参数的一般模式为
        // 1. 指向结构体的指针（包含创建信息）
        // 2. 指向分配器的指针（通常为 nullptr）
        // 3. 指向对象的指针（创建后存储对象的指针）
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }

        // 可以通过 vkEnumerateInstanceExtensionProperties 函数列出所有可用的扩展
        uint32_t extensionCount = 0;
        // 第一个参数可选，能够按特定的验证层过滤扩展，这里先忽略
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr); // 先得到扩展数量
        std::vector<VkExtensionProperties> extensions(extensionCount); // 创建一个 vector 来存储扩展信息
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()); // 然后再获取具体信息
        std::cout << "available extensions:\n";
        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }
        // 所有的扩展都以某个公司的缩写开头，指明这是哪个公司的扩展，其中 KHR 是 Khronos Group 的缩写，表明这是官方扩展，比如：
        // VK_NV_external_memory_capabilities —— NVIDIA
        // VK_KHR_portability_enumeration —— Khronos Group
        // VK_LUNARG_direct_driver_loading —— LunarG
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
