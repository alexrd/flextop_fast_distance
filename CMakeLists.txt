CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(flextop_features)

# We need to know where libtorch is installed so we can access the headers and libraries.
SET(LIBTORCH_DIR "/usr/local" CACHE PATH "Where LIBTORCH is installed")
INCLUDE_DIRECTORIES("${LIBTORCH_DIR}/include")
#SET(CMAKE_PREFIX_PATH "${PYTORCH_DIR}")
FIND_PACKAGE(Torch REQUIRED)
LINK_DIRECTORIES("${LIBTORCH_DIR}/lib")

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 14)

# Set flags for linking on mac
IF(APPLE)
	SET (CMAKE_INSTALL_NAME_DIR "@rpath")
	SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# Collect up source files
FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

ADD_EXECUTABLE(flextopFF $src_files)

