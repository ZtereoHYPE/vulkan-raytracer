#pragma once

#include "pch.hpp"
#include <unordered_map>
#include <GLFW/glfw3.h>


struct Window {
    GLFWwindow* glfwWindow;
    bool* resizedCallback = nullptr;

    static std::vector<const char*> getRequiredExtensions();

    Window(const char *title, int initialWidth, int initialHeight);
    Window(const Window &obj) = delete; // do not allow copies of this class

    VkSurfaceKHR createVulkanSurface(VkInstance instance);
    void setResizedCallbackVariable(bool* callback);
    void getFramebufferSize(int *width, int *height);
    void waitEvents();
    void pollEvents();
    bool shouldClose();
    void terminate();
};
