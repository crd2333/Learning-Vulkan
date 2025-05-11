# Learning Vulkan
这个仓库是我学习 [Vulkan 教程](https://tutorial.vulkan.net.cn/)（[英文版 Vulkan Tutorial](https://vulkan-tutorial.com/)）的环境搭建与（中文）代码注释记录，vulkan tutorial 最初的源代码可以在 [(Github) Overv/VulkanTutorial](https://github.com/Overv/VulkanTutorial/tree/main/code) 找到。

其实基本就是对着教程一步步敲注释（把教程中一些翻译得不通顺的地方改一改），对部分疑难点加入自己的见解与网络资料，对部分疑似有问题的地方进行修正。如果有读者也跟着啃 vulkan tutorial 的话，可以参考这个仓库的注释，或许我的理解能帮到你（但说不定也有错误，笑）。

个人觉得 vulkan 不愧于它的复杂度，感觉完整把它的设置流程走完，能对图形渲染的底层原理有更深刻的理解。相应的，可能先有 OpenGL 的基础会更容易上手（比如啃一遍或一部分的 [Learn OpenGL CN](https://learnopengl-cn.github.io/)）。

- 再附上一些跟 vulkan tutorial 类似的旨在一步步学习的网络文档（中文，英文能看但啃起来太累x）
  1. [Vulkan 入门精要](https://fuxiii.github.io/Essentials.of.Vulkan/)
  2. [Vulkan 极客教程](https://geek-docs.com/vulkan/vulkan-tutorial/vulkan-tutorial-index.html)
  3. [EasyVulkan](https://easyvulkan.github.io/)

至于环境搭建，这里就不说了，vulkan 本身相比环境搭建可要难太多了，我这里附上我所用的 `CmakeLists.txt` 供参考。如果实在配不好，就去问 AI 吧（因为我也是问 AI 配的 XD）。

完整学完了 31 章内容，~~现在可以自己去手搓一个 RHI 和引擎辣~~。才学完一个基本的三角形、长方体绘制呢，更多 fancy 的功能都还没加，还是看看远方的 [Vulkan Examples](https://github.com/SaschaWillems/Vulkan) 吧（以后有时间再看）。