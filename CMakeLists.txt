CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(flextop_fast_distance)

# We need to know where libtorch is installed so we can access the headers and libraries.
FIND_PACKAGE(Torch REQUIRED)

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 14)

# set debug mode
SET (CMAKE_BUILD_TYPE Debug)

# Specify where to build
SET (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/build")

# Set flags for linking on mac
IF(APPLE)
	#SET (CMAKE_INSTALL_NAME_DIR "@rpath")
	SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# flextopFD library
ADD_LIBRARY(flextopFD SHARED "")
TARGET_INCLUDE_DIRECTORIES(flextopFD PUBLIC ${TORCH_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(flextopFD "${TORCH_LIBRARIES}")

# test executable
ADD_EXECUTABLE(TestFlextopFD "")
TARGET_INCLUDE_DIRECTORIES(TestFlextopFD PUBLIC ${TORCH_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(TestFlextopFD PUBLIC "src")
TARGET_LINK_LIBRARIES(TestFlextopFD "${TORCH_LIBRARIES}")
TARGET_LINK_LIBRARIES(TestFlextopFD flextopFD)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)



