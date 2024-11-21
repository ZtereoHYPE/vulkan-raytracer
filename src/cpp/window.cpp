#include "window.hpp"

Window::Window(const char *title, int initialWidth, int initialHeight) {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // todo: re-enable resizing when cleaned up code
    auto glfwWindow = glfwCreateWindow(initialWidth, initialHeight, title, nullptr, nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);

    this->glfwWindow = glfwWindow;

    printf("log: initialized glfw\n");
}

void Window::terminate() {
    glfwDestroyWindow(this->glfwWindow);
    glfwTerminate();
}

void Window::getFramebufferSize(int *width, int *height) {
    glfwGetFramebufferSize(glfwWindow, width, height);
}

void Window::waitEvents() {
    glfwWaitEvents();
}

void Window::pollEvents() {
    glfwPollEvents();
}

bool Window::shouldClose() {
    return glfwWindowShouldClose(glfwWindow);
}

VkSurfaceKHR Window::createVulkanSurface(VkInstance instance) {
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(instance, glfwWindow, nullptr, &surface) != VK_SUCCESS) {
        std::runtime_error("Failed to create window surface!");
    }

    return surface;
}

std::vector<const char*> Window::getRequiredExtensions() {
    uint32_t count = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&count);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);

    return extensions;
}