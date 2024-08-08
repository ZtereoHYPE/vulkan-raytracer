# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -O2 
LDFLAGS := -lglfw3 -lvulkan -ldl -lpthread -lwayland-client
SHADER_COMPILER := glslc

# Directories
SRC_DIR := src
BUILD_DIR := build
SHADER_DIR := src/shaders
SHADER_BUILD_DIR := build/shaders

# Source files
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Shader files
VERTEX_SHADERS := $(wildcard $(SHADER_DIR)/*.vert)
FRAGMENT_SHADERS := $(wildcard $(SHADER_DIR)/*.frag)
COMPILED_VERTEX_SHADERS := $(VERTEX_SHADERS:$(SHADER_DIR)/%.vert=$(SHADER_BUILD_DIR)/%.vert.spv)
COMPILED_FRAGMENT_SHADERS := $(FRAGMENT_SHADERS:$(SHADER_DIR)/%.frag=$(SHADER_BUILD_DIR)/%.frag.spv)

# Executable
TARGET := VulkanTest

# Rules
all: $(TARGET) run

$(TARGET): $(OBJ_FILES) $(COMPILED_VERTEX_SHADERS) $(COMPILED_FRAGMENT_SHADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SHADER_BUILD_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

$(SHADER_BUILD_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

run:
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(SHADER_BUILD_DIR) $(TARGET)

.PHONY: all clean
