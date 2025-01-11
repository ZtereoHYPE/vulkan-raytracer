#include "window.hpp"

Window::Window(const char *title, uint initialWidth, uint initialHeight) {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // todo: re-enable resizing when cleaned up code
    auto glfwWindow = glfwCreateWindow((int)initialWidth, (int)initialHeight, title, nullptr, nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);

    this->glfwWindow = glfwWindow;

    printf("log: initialized glfw\n");
}

void Window::terminate() const {
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

vk::SurfaceKHR Window::createVulkanSurface(vk::Instance const &instance) const {
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(instance, glfwWindow, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }

    return surface;
}

std::vector<const char*> Window::getRequiredExtensions() {
    uint32_t count = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&count);

    std::vector extensions(glfwExtensions, glfwExtensions + count);

    return extensions;
}