#pragma once

#include "pch.hpp"
#include <unordered_map>
#include <GLFW/glfw3.h>


struct Window {
    GLFWwindow* glfwWindow;
    bool* resizedCallback = nullptr;
    std::unordered_map<int, bool*> keypressCallbacks;

    Window(const char *title, int initialWidth, int initialHeight);
    void setResizedCallbackVariable(bool* callback);
    void getFramebufferSize(int *width, int *height);
    void waitEvents();
    void pollEvents();
    bool shouldClose();
    void setKeypressCallback(int key, bool *callback);
    void terminate();
};

VkSurfaceKHR createVulkanWindowSurface(Window* window, VkInstance instance);

