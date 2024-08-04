CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw3 -lvulkan -ldl -lpthread -lwayland-client

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest shaders
	./VulkanTest

shaders: ./shaders/shader.frag ./shaders/shader.vert
	glslc shaders/shader.vert -o shaders/vert.spv
	glslc shaders/shader.frag -o shaders/frag.spv

clean:
	rm -f VulkanTest