# Student Meal Portal System Makefile
# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -std=c++17 -Wall -Wextra -g -DDEBUG
TARGET = student_portal
SOURCE = student_meal_portal.cpp

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCE)
	@echo "Compiling Student Meal Portal System..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "✅ Compilation successful! Run with: ./$(TARGET)"

# Debug build
debug: $(SOURCE)
	@echo "Compiling Student Meal Portal System (Debug mode)..."
	$(CXX) $(DEBUG_FLAGS) -o $(TARGET)_debug $(SOURCE)
	@echo "✅ Debug compilation successful! Run with: ./$(TARGET)_debug"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TARGET) $(TARGET)_debug
	@echo "✅ Clean completed!"

# Install (copy to /usr/local/bin)
install: $(TARGET)
	@echo "Installing Student Meal Portal System..."
	@sudo cp $(TARGET) /usr/local/bin/
	@echo "✅ Installation completed! You can now run 'student_portal' from anywhere"

# Uninstall
uninstall:
	@echo "Uninstalling Student Meal Portal System..."
	@sudo rm -f /usr/local/bin/$(TARGET)
	@echo "✅ Uninstallation completed!"

# Run the program
run: $(TARGET)
	@echo "Starting Student Meal Portal System..."
	@./$(TARGET)

# Quick compile and run
quick: clean $(TARGET) run

# Check for required tools
check:
	@echo "Checking system requirements..."
	@which $(CXX) > /dev/null || (echo "❌ g++ not found! Please install gcc/g++" && exit 1)
	@$(CXX) --version | head -1
	@echo "✅ System requirements satisfied!"

# Help target
help:
	@echo "Student Meal Portal System - Makefile Help"
	@echo "=========================================="
	@echo "Available targets:"
	@echo "  all      - Build the main executable (default)"
	@echo "  debug    - Build debug version with extra information"
	@echo "  clean    - Remove all build artifacts"
	@echo "  install  - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall- Remove from /usr/local/bin (requires sudo)"
	@echo "  run      - Build and run the program"
	@echo "  quick    - Clean, build, and run in one command"
	@echo "  check    - Verify system requirements"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make           # Build the program"
	@echo "  make run       # Build and run"
	@echo "  make debug     # Build debug version"
	@echo "  make quick     # Clean, build, and run"

# Phony targets (not files)
.PHONY: all debug clean install uninstall run quick check help