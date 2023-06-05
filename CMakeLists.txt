CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(flextop_fast_distance)

SET(LIB_TARGET flextopFD)

# We need to know where libtorch is installed so we can access the headers and libraries.
FIND_PACKAGE(Torch REQUIRED)

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 14)

# Set flags for linking on mac
IF(APPLE)
	SET (CMAKE_INSTALL_NAME_DIR "@rpath")
	SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# Collect up source files
FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
FILE(GLOB incl_files  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
ADD_LIBRARY(${LIB_TARGET} SHARED ${src_files} ${incl_files})

TARGET_INCLUDE_DIRECTORIES(${LIB_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
TARGET_INCLUDE_DIRECTORIES(${LIB_TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(${LIB_TARGET} "${TORCH_LIBRARIES}")

INSTALL(TARGETS ${LIB_TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/plugins)
