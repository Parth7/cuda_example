# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/parthparakh/CLionProjects/cuda_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cuda_example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_example.dir/flags.make

CMakeFiles/cuda_example.dir/main.cpp.o: CMakeFiles/cuda_example.dir/flags.make
CMakeFiles/cuda_example.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda_example.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda_example.dir/main.cpp.o -c /Users/parthparakh/CLionProjects/cuda_example/main.cpp

CMakeFiles/cuda_example.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_example.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/parthparakh/CLionProjects/cuda_example/main.cpp > CMakeFiles/cuda_example.dir/main.cpp.i

CMakeFiles/cuda_example.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_example.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/parthparakh/CLionProjects/cuda_example/main.cpp -o CMakeFiles/cuda_example.dir/main.cpp.s

# Object files for target cuda_example
cuda_example_OBJECTS = \
"CMakeFiles/cuda_example.dir/main.cpp.o"

# External object files for target cuda_example
cuda_example_EXTERNAL_OBJECTS =

cuda_example: CMakeFiles/cuda_example.dir/main.cpp.o
cuda_example: CMakeFiles/cuda_example.dir/build.make
cuda_example: CMakeFiles/cuda_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cuda_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_example.dir/build: cuda_example

.PHONY : CMakeFiles/cuda_example.dir/build

CMakeFiles/cuda_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_example.dir/clean

CMakeFiles/cuda_example.dir/depend:
	cd /Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/parthparakh/CLionProjects/cuda_example /Users/parthparakh/CLionProjects/cuda_example /Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug /Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug /Users/parthparakh/CLionProjects/cuda_example/cmake-build-debug/CMakeFiles/cuda_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_example.dir/depend

