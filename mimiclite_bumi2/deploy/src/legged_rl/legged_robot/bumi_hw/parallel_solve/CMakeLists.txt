cmake_minimum_required(VERSION 3.0.2)
project(parallel_solve)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "i686")
    set(TARGET_ARCH "x86_64")
    message(STATUS "Building for x86_64 architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(TARGET_ARCH "aarch64")
    message(STATUS "Building for ARM64 architecture")
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
#     set(TARGET_ARCH "armv7")
#     message(STATUS "Building for ARMv7 architecture")
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}. Assuming x86_64")
    set(TARGET_ARCH "x86_64")
endif()

set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/${TARGET_ARCH}")
set(LIBRARY_FILE "${LIBRARY_DIR}/libparallel_ankle_cpp.a")

if(NOT EXISTS ${LIBRARY_FILE})
    message(FATAL_ERROR "Library file not found for architecture ${TARGET_ARCH}: ${LIBRARY_FILE}")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(parallel_ankle_cpp STATIC IMPORTED)
set_target_properties(parallel_ankle_cpp PROPERTIES
  IMPORTED_LOCATION ${LIBRARY_FILE}
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES parallel_ankle_cpp
  CATKIN_DEPENDS roscpp 
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(FILES ${LIBRARY_FILE}
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
              GROUP_READ GROUP_EXECUTE 
              WORLD_READ WORLD_EXECUTE
  # RENAME "libparallel_ankle_cpp.so"
)

