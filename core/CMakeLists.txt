cmake_minimum_required(VERSION 3.10)
project(cmfo_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    language/matrix_engine.cpp
)

# Create shared library
add_library(cmfo_core SHARED ${SOURCES})

# Include directories
target_include_directories(cmfo_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/language>
    $<INSTALL_INTERFACE:include>
)

# Export for installation (optional, but good practice)
set_target_properties(cmfo_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
