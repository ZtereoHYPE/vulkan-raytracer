#include "window.hpp"

/** Construct a window, initializing the glfwWindow. */
Window::Window(const char *title, uint initialWidth, uint initialHeight) {
    glfwInit();

    // do not use OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto glfwWindow = glfwCreateWindow((int)initialWidth, (int)initialHeight, title, nullptr, nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);
    this->glfwWindow = glfwWindow;

    printf("log: initialized glfw\n");
}

/** Terminate and close the window */
void Window::terminate() const {
    glfwDestroyWindow(this->glfwWindow);
    glfwTerminate();
}

/** Set the size of the window's framebuffer */
void Window::getFramebufferSize(int *width, int *height) {
    glfwGetFramebufferSize(glfwWindow, width, height);
}

/** Poll events that happened to the window such as pressing the close button */
void Window::pollEvents() {
    glfwPollEvents();
}

/** Returns whether the window should close */
bool Window::shouldClose() {
    return glfwWindowShouldClose(glfwWindow);
}

/** Creates and returns a surface Vulkan can render to */
vk::SurfaceKHR Window::createVulkanSurface(vk::Instance const &instance) const {
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(instance, glfwWindow, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }

    return surface;
}

/** Returns the list of the names of the Vulkan extensions required for rendering to the window */
std::vector<const char*> Window::getRequiredExtensions() {
    uint32_t count = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&count);

    std::vector extensions(glfwExtensions, glfwExtensions + count);

    return extensions;
}