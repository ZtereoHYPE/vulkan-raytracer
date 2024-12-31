#pragma once

#include "pch.hpp"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>


struct Window {
    GLFWwindow* glfwWindow;

    static std::vector<const char*> getRequiredExtensions();

    Window(const char *title, uint initialWidth, uint initialHeight);
    Window(const Window &obj) = delete; // do not allow copies of this class

    vk::SurfaceKHR createVulkanSurface(vk::Instance const &instance) const;
    void getFramebufferSize(int *width, int *height);
    void waitEvents();
    void pollEvents();
    bool shouldClose();
    void terminate() const;
};
