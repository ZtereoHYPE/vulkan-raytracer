#include "window.hpp"

// todo: switch to a real callback
void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto ctx = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    if (ctx->resizedCallback != nullptr) {
        *(ctx->resizedCallback) = true;
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto ctx = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    
    if (ctx->keypressCallbacks.contains(key)) {
        if (action == GLFW_PRESS) {
            *ctx->keypressCallbacks[key] = true;
        } else if (action == GLFW_RELEASE) {
            *ctx->keypressCallbacks[key] = false;
        } else {
            printf("Unknown action encountered!");
        }
    }
}


Window::Window(const char *title, int initialWidth, int initialHeight) {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // todo: re-enable resizing when cleaned up code
    auto glfwWindow = glfwCreateWindow(initialWidth, initialHeight, title, nullptr, nullptr);

    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, framebufferResizeCallback);
    glfwSetKeyCallback(glfwWindow, keyCallback);

    this->glfwWindow = glfwWindow;

    printf("log: initialized glfw\n");
}

void Window::terminate() {
    glfwDestroyWindow(this->glfwWindow);
    glfwTerminate();
}

void Window::setResizedCallbackVariable(bool* callback) {
    resizedCallback = callback;
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

void Window::setKeypressCallback(int key, bool *callback) {
    keypressCallbacks[key] = callback;
}

VkSurfaceKHR createVulkanWindowSurface(Window* window, VkInstance instance) {
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(instance, window->glfwWindow, nullptr, &surface) != VK_SUCCESS) {
        std::runtime_error("Failed to create window surface!");
    }

    return surface;
}