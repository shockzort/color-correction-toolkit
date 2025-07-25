# Color Correction System Makefile

BUILD_DIR := build
BUILD_TYPE := Release

.PHONY: all build clean test test-unit test-integration format

all: build

# Create build directory and configure
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ..

# Build the project
build: configure
	@cd $(BUILD_DIR) && make -j$(shell nproc)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)

# Run all tests
test: build
	@cd $(BUILD_DIR) && ctest --verbose

# Run unit tests only
test-unit: build
	@cd $(BUILD_DIR) && ctest -R "UnitTests"

# Run integration tests only  
test-integration: build
	@cd $(BUILD_DIR) && ctest -R "IntegrationTests"

# Run processing test data (placeholder)
test-processing: build
	@echo "Processing test data..."
	@cd $(BUILD_DIR) && ./color_calibrator --test-mode

# Format code with clang-format
format:
	@find modules -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i
	@echo "Code formatted successfully"

# Debug build
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug ..
	@cd $(BUILD_DIR) && make -j$(shell nproc)

# Install (placeholder)
install: build
	@cd $(BUILD_DIR) && make install

# Help target
help:
	@echo "Available targets:"
	@echo "  build        - Build the project (Release mode)"
	@echo "  debug        - Build in Debug mode"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-processing  - Run processing test data"
	@echo "  format       - Format code with clang-format"
	@echo "  install      - Install the project"
	@echo "  help         - Show this help message"