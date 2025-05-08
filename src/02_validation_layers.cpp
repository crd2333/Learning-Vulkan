#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>

/*
 * Vulkan 的设计理念是尽量减少驱动程序开销，其体现之一就是错误检查很有限
 * 例如枚举设置不正确、必须参数传递了 nullptr 等错误只会导致崩溃而没有显式报错
 * Vulkan 引入一个优雅的验证层概念来添加检查到 API 中，常见操作包括
 *   对照规范检查参数值以检测误用
 *   跟踪对象的创建和销毁以查找资源泄漏
 *   通过跟踪调用来源的线程来检查线程安全性
 *   将每个调用及其参数记录到标准输出
 *   跟踪 Vulkan 调用以进行性能分析和重放
 * 从而我们可以在调试中启用验证层，发布构建时禁用它们。Vulkan 本身并不附带验证层，而是在 LunarG Vulkan SDK 中提供
 */

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// 用两个宏来控制验证层的启用和禁用
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    // vkCreateDebugUtilsMessengerEXT 是个扩展函数，不会自动加载，需要用 vkGetInstanceProcAddr 查找其地址，这个函数对此进行封装
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger); // 找到函数之后调用，创建出 VkDebugUtilsMessengerEXT，通过 pDebugMessenger 返回
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator
) {
    // 销毁也是同理
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

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

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger; // Vulkan 中连调试回调也需要进行显式创建和销毁管理

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
            // 现在如果把上面那句故意删掉，就会报下面的错误（但并不是像教程里那样的红色？）
            // validation layer : vkDestroyInstance() : Object Tracking - For VkInstance 0x1bed934b560, VkDebugUtilsMessengerEXT 0xfd5b260000000001 has not been destroyed.
            // The Vulkan spec states : All child objects created using instance must have been destroyed prior to destroying instance(https ://vulkan.lunarg.com/doc/view/1.4.309.0/windows/antora/spec/latest/chapters/initialization.html#VUID-vkDestroyInstance-instance-00629)
            // validation layer : vkDestroyInstance() : Object Tracking - For VkInstance 0x1bed934b560, VkDebugUtilsMessengerEXT 0xfd5b260000000001 has not been destroyed.
            // The Vulkan spec states : All child objects created using instance must have been destroyed prior to destroying instance(https ://vulkan.lunarg.com/doc/view/1.4.309.0/windows/antora/spec/latest/chapters/initialization.html#VUID-vkDestroyInstance-instance-00629)
            // validation layer : Unloading layer library C : \WINDOWS\System32\DriverStore\FileRepository\nvlti.inf_amd64_071505319ec619da\.\nvoglv64.dll
            // validation layer : Unloading layer library C : \WINDOWS\System32\DriverStore\FileRepository\nvlti.inf_amd64_071505319ec619da\.\nvoglv64.dll
            // validation layer : Unloading layer library C : \WINDOWS\System32\DriverStore\FileRepository\u0386187.inf_amd64_e45f6929dd019ce6\B385477\.\amdvlk64.dll
            // validation layer : Unloading layer library D : \VulkanSDK\1.4.309.0\Bin\.\VkLayer_khronos_validation.dll
        }

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // vkCreateDebugUtilsMessengerEXT 调用需要已经创建好有效的 instance
        // 这反过来意味着 instance 的创建和销毁 (vkCreateInstance, vkDestroyInstance) 调用时没有调试信息
        // 为此需要为这两个函数创建单独的 debug messenger，传给 VkInstanceCreateInfo.pNext
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo); // 这个创建销毁专用的 vkCreateDebugUtilsMessengerEXT 也需要像常规的那样填充信息
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // 填充结构体，封装为一个函数方便多次调用
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback; // 回调函数的指针
        createInfo.pUserData = nullptr; // Optional
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // 根据是否启用验证层返回所需的扩展列表（是否要再加个 debug_utils）
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // VK_EXT_debug_utils，可以避免拼写错误
        }

        return extensions;
    }

    // 检查验证层是否可用
    bool checkValidationLayerSupport() {
        //  vkEnumerateInstanceLayerProperties 函数的用法类似于 vkEnumerateInstanceExtensionProperties
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) { // 检查 validationLayers 里的层都合法
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    // 回调函数，用 Vulkan 的宏定义
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,  // 指定消息的严重性
        VkDebugUtilsMessageTypeFlagsEXT messageType,             // 指定消息的类型（与规范或性能无关的事件、违反规范的事件、潜在非最佳使用等）
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, // 消息本身的结构体
        void* pUserData                                            // 用户数据，在回调的设置期间指定，可以借此将自己的数据传递给它
    ) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
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
