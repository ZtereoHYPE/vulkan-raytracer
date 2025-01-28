#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

/*
 * Wrapper around GLFW used to create a window and obtain its surface.
 */
struct Window {
    GLFWwindow* glfwWindow;
    const char *title;
    uint width, height;

    bool hasInit = false;
    bool windowShouldClose = false;

    static std::vector<const char*> getRequiredExtensions();

    Window(const char *title, uint initialWidth, uint initialHeight);
    Window(const Window &obj) = delete; // do not allow copies of this class

    void init();
    vk::SurfaceKHR createVulkanSurface(vk::Instance const &instance) const;
    void getFramebufferSize(int *width, int *height);
    void pollEvents();
    bool shouldClose();
    void setShouldClose(bool);
    void terminate() const;
};
