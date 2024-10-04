# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -O2 -flto
LDFLAGS := -lglfw3 -lvulkan -ldl -lpthread -lwayland-client
SHADER_COMPILER := glslc

# Directories
SRC_DIR := src
BUILD_DIR := build
SHADER_DIR := src/shaders
SUBDIR_SHADER_FILES := $(shell find $(SHADER_DIR) -mindepth 2 -name "*.vert" -o -name "*.frag")
SHADER_BUILD_DIR := build/shaders

# Source files
SRC_FILES := $(shell find $(SRC_DIR) -name "*.cpp")
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Precompiled header
PCH_HEADER := $(SRC_DIR)/pch.hpp
PCH_NAME := precompiled.hpp
PCH_FILE := $(BUILD_DIR)/$(PCH_NAME).gch

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

# Compile the precompiled header (pch.h)
$(PCH_FILE): $(PCH_HEADER)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(PCH_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(BUILD_DIR) -include $(PCH_NAME) -c $< -o $@

$(SHADER_BUILD_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert $(SUBDIR_SHADER_FILES)
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

$(SHADER_BUILD_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag $(SUBDIR_SHADER_FILES)
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

run:
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(SHADER_BUILD_DIR) $(TARGET)

.PHONY: all clean
