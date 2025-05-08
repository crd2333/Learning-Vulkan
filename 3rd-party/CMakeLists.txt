# 设置文件夹名称
set(third_party_folder "3rd-party")

# 禁用构建共享库
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# GLFW配置
if(NOT TARGET glfw)
    option(GLFW_BUILD_EXAMPLES "" OFF)
    option(GLFW_BUILD_TESTS "" OFF)
    option(GLFW_BUILD_DOCS "" OFF)
    option(GLFW_INSTALL "" OFF)
    add_subdirectory(glfw)
    set_target_properties(glfw PROPERTIES FOLDER ${third_party_folder}/glfw)
    set_target_properties(update_mappings PROPERTIES FOLDER ${third_party_folder}/glfw)
endif()

# GLM配置
if(NOT TARGET glm)
    option(GLM_TEST_ENABLE "" OFF)
    add_subdirectory(glm)
    set_target_properties(glm PROPERTIES FOLDER ${third_party_folder}/glm)
endif()

# STB配置
if(NOT TARGET stb)
    file(GLOB stb_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/stb/*.h")
    add_library(stb INTERFACE ${stb_sources})
    target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
    set_target_properties(stb PROPERTIES FOLDER ${third_party_folder}/stb)
endif()

# TinyObjLoader配置
if(NOT TARGET tinyobjloader)
    option(TINYOBJLOADER_BUILD_TEST_LOADER "" OFF)
    option(TINYOBJLOADER_BUILD_EXAMPLES "" OFF)
    add_subdirectory(tinyobjloader)
    set_target_properties(tinyobjloader PROPERTIES FOLDER ${third_party_folder}/tinyobjloader)
    if(TARGET uninstall)
        set_target_properties(uninstall PROPERTIES FOLDER ${third_party_folder}/tinyobjloader)
    endif()
endif()

# 添加Vulkan包含目录
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/VulkanSDK/include")