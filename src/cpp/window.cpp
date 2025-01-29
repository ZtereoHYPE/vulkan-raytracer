#include "window.hpp"

/** Construct a window, initializing the glfwWindow. */
Window::Window(const char *title, uint initialWidth, uint initialHeight)
:
    title(title),
    width(initialWidth),
    height(initialHeight)
{}

void Window::init() {
    glfwInit();

    // do not use OpenGL
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    auto glfwWindow = glfwCreateWindow((int)width, (int)height, title, nullptr, nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);
    this->glfwWindow = glfwWindow;

    hasInit = true;

    printf("log: initialized glfw\n");
}

/** Terminate and close the window */
void Window::terminate() const {
    if (!hasInit) return;

    glfwDestroyWindow(this->glfwWindow);
    glfwTerminate();
}

/** Get the size of the window's framebuffer */
void Window::getFramebufferSize(int *width, int *height) {
    if (!hasInit) {
        *width = this->width; 
        *height = this->height;
    } else
        glfwGetFramebufferSize(glfwWindow, width, height);
}

/** Poll events that happened to the window such as pressing the close button */
void Window::pollEvents() {
    if (!hasInit) return;
    glfwPollEvents();
}

/** Returns whether the window should close */
bool Window::shouldClose() {
    if (!hasInit) 
        return windowShouldClose;
    else
        return glfwWindowShouldClose(glfwWindow);
}

void Window::setShouldClose(bool shouldClose) {
    if (!hasInit)
        this->windowShouldClose = shouldClose;
    else 
        glfwSetWindowShouldClose(glfwWindow, shouldClose);
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