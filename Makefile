# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++26 -g -Wno-pointer-arith -O2
#CXXFLAGS := -std=c++26 -O2 -flto
LDFLAGS := -lglfw3 -lvulkan -ldl -lyaml-cpp
SHADER_COMPILER := glslc

# Directories
SRC_DIR := src/cpp
BUILD_DIR := build
SHADER_DIR := src/shaders
SUBDIR_SHADER_FILES := $(shell find $(SHADER_DIR) -mindepth 2 -name "*.comp")
SHADER_BUILD_DIR := build/shaders

# Source files
SRC_FILES := $(shell find $(SRC_DIR) -name "*.cpp")
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Precompiled header
PCH_HEADER := $(SRC_DIR)/pch.hpp
PCH_NAME := precompiled.hpp
PCH_FILE := $(BUILD_DIR)/$(PCH_NAME).gch

# Shader files
COMPUTE_SHADERS := $(wildcard $(SHADER_DIR)/*.comp)
COMPILED_COMPUTE_SHADERS := $(COMPUTE_SHADERS:$(SHADER_DIR)/%.comp=$(SHADER_BUILD_DIR)/%.comp.spv)

# Executable
TARGET := VulkanTest

# Rules
all: $(TARGET) run
compile: $(TARGET)

$(TARGET): $(OBJ_FILES) $(COMPILED_COMPUTE_SHADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS)

# Compile the precompiled header (pch.h)
$(PCH_FILE): $(PCH_HEADER)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.hpp $(PCH_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(BUILD_DIR) -include $(PCH_NAME) -c $< -o $@

$(SHADER_BUILD_DIR)/%.comp.spv: $(SHADER_DIR)/%.comp $(SUBDIR_SHADER_FILES)
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

run:
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(SHADER_BUILD_DIR) $(TARGET)

.PHONY: all clean
