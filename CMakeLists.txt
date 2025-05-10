cmake_minimum_required(VERSION 3.8)

project(VulkanTutorial)

option(USE_WINDOWS_GUI "Build applications with Windows GUI subsystem" OFF) # 直接在 Windows 上运行，不开控制台

# 确保src和shaders目录存在
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# 添加第三方库
add_subdirectory(3rd-party)

# 定义glslang验证器目标
add_executable(glslang::validator IMPORTED)
if(WIN32)
  set(GLSLANG_VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/bin/Win32/glslangValidator.exe")
elseif(APPLE)
  set(GLSLANG_VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/bin/MacOS/glslangValidator")
else()
  set(GLSLANG_VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/bin/Linux/glslangValidator")
endif()
set_property(TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")

# 设置STB包含目录
set(STB_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/stb")

function(add_shaders_target TARGET)
  cmake_parse_arguments("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  set(SHADERS_DIR ${SHADER_CHAPTER_NAME}/shaders)
  add_custom_command(
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
  set(SHADERS ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv)
  # 检查着色器源文件中是否包含计算着色器
  string(FIND "${SHADER_SOURCES}" "comp" COMPUTE_SHADER_INDEX)
  if(${COMPUTE_SHADER_INDEX} GREATER -1)
    set(SHADERS ${SHADERS} ${SHADERS_DIR}/comp.spv)
  endif()
  add_custom_command(
    OUTPUT ${SHADERS}
    COMMAND ${GLSLANG_VALIDATOR}
    ARGS --target-env vulkan1.0 ${SHADER_SOURCES} --quiet
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
  )
  add_custom_target(${TARGET} DEPENDS ${SHADERS})
endfunction()

function(add_chapter CHAPTER_NAME)
  cmake_parse_arguments(CHAPTER "" "SHADER" "LIBS;TEXTURES;MODELS" ${ARGN})

  # 从src目录获取源文件

  # 从src目录获取源文件
  add_executable(${CHAPTER_NAME} src/${CHAPTER_NAME}.cpp)

  # 添加此段代码，根据全局选项设置窗口模式
  if(WIN32 AND USE_WINDOWS_GUI AND NOT CHAPTER_CONSOLE)
    target_link_options(${CHAPTER_NAME} PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
  endif()

  set_target_properties(${CHAPTER_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CHAPTER_NAME})
  set_target_properties(${CHAPTER_NAME} PROPERTIES CXX_STANDARD 17)

  # 添加章节名称的定义，使其在源代码中可用
  target_compile_definitions(${CHAPTER_NAME} PRIVATE CHAPTER_NAME="${CHAPTER_NAME}")

  # 链接Vulkan和GLFW
  if(WIN32)
    target_link_libraries(${CHAPTER_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/lib/Win32/vulkan-1.lib" glfw)
  elseif(APPLE)
    target_link_libraries(${CHAPTER_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/lib/MacOS/libvulkan.1.dylib" glfw)
  else()
    target_link_libraries(${CHAPTER_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/lib/Linux/libvulkan.so.1" glfw)
  endif()

  # 添加Vulkan头文件包含目录
  target_include_directories(${CHAPTER_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/VulkanSDK/include"
    ${STB_INCLUDEDIR})

  if(DEFINED CHAPTER_SHADER)
    set(CHAPTER_SHADER_TARGET ${CHAPTER_NAME}_shader)
    # 从shaders目录获取着色器文件
    file(GLOB SHADER_SOURCES
          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${CHAPTER_SHADER}.frag"
          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${CHAPTER_SHADER}.vert"
          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${CHAPTER_SHADER}.comp")
    add_shaders_target(${CHAPTER_SHADER_TARGET} CHAPTER_NAME ${CHAPTER_NAME} SOURCES ${SHADER_SOURCES})
    add_dependencies(${CHAPTER_NAME} ${CHAPTER_SHADER_TARGET})
  endif()

  # 链接其他库
  if(DEFINED CHAPTER_LIBS)
    foreach(LIB ${CHAPTER_LIBS})
      if(TARGET ${LIB})
        target_link_libraries(${CHAPTER_NAME} ${LIB})
      endif()
    endforeach()
  endif()

  # 复制模型和纹理资源
  if(DEFINED CHAPTER_MODELS)
    file(COPY ${CHAPTER_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/models)
  endif()
  if(DEFINED CHAPTER_TEXTURES)
    file(COPY ${CHAPTER_TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/textures)
  endif()
endfunction()

# 添加所有章节
add_chapter(00_base_code)
add_chapter(01_instance_creation)
add_chapter(02_validation_layers)
add_chapter(03_physical_device_selection)
add_chapter(04_logical_device)
add_chapter(05_window_surface)
add_chapter(06_swap_chain_creation)
add_chapter(07_image_views)
add_chapter(08_graphics_pipeline)
add_chapter(09_shader_modules
            SHADER 09_shader_base)
add_chapter(10_fixed_functions
            SHADER 09_shader_base)
add_chapter(11_render_passes
            SHADER 09_shader_base)
add_chapter(12_graphics_pipeline_complete
            SHADER 09_shader_base)
add_chapter(13_framebuffers
            SHADER 09_shader_base)
add_chapter(14_command_buffers
            SHADER 09_shader_base)
add_chapter(15_hello_triangle
            SHADER 09_shader_base)
add_chapter(16_frames_in_flight
            SHADER 09_shader_base)
add_chapter(17_swap_chain_recreation
            SHADER 09_shader_base)
add_chapter(18_vertex_input
            SHADER 18_shader_vertexbuffer
            LIBS glm)
add_chapter(19_vertex_buffer
            SHADER 18_shader_vertexbuffer
            LIBS glm)
add_chapter(20_staging_buffer
            SHADER 18_shader_vertexbuffer
            LIBS glm)
add_chapter(21_index_buffer
            SHADER 18_shader_vertexbuffer
            LIBS glm)
add_chapter(22_descriptor_set_layout
            SHADER 22_shader_ubo
            LIBS glm)
add_chapter(23_descriptor_sets
            SHADER 22_shader_ubo
            LIBS glm)
add_chapter(23_descriptor_sets_variant
            SHADER 22_shader_ubo_variant
            LIBS glm)
add_chapter(24_texture_image
            SHADER 22_shader_ubo TEXTURES resources/texture.jpg
            LIBS glm)
add_chapter(25_sampler
            SHADER 22_shader_ubo TEXTURES resources/texture.jpg
            LIBS glm)
add_chapter(26_texture_mapping
            SHADER 26_shader_textures TEXTURES resources/texture.jpg
            LIBS glm)
add_chapter(27_depth_buffering
            SHADER 27_shader_depth
            TEXTURES resources/texture.jpg
            LIBS glm)
add_chapter(28_model_loading
            SHADER 27_shader_depth
            MODELS resources/viking_room.obj
            TEXTURES resources/viking_room.png
            LIBS glm tinyobjloader)
add_chapter(29_mipmapping
            SHADER 27_shader_depth
            MODELS resources/viking_room.obj
            TEXTURES resources/viking_room.png
            LIBS glm tinyobjloader)
add_chapter(30_multisampling
            SHADER 27_shader_depth
            MODELS resources/viking_room.obj
            TEXTURES resources/viking_room.png
            LIBS glm tinyobjloader)
add_chapter(31_compute_shader
            SHADER 31_shader_compute
            LIBS glm)