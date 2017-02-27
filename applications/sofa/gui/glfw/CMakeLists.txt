cmake_minimum_required(VERSION 2.8.12)
project(SofaGuiGLFW)

set(HEADER_FILES
    GLFWGui.h)

set(SOURCE_FILES
    GLFWGui.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaGuiCommon SofaHelper glfw) 
if(UNIX)
    if(APPLE)
        find_library(COCOA_LIBRARY Cocoa)
        find_library(COREVIDEO_LIBRARY CoreVideo)
        find_library(IOKIT_LIBRARY IOKit)
        target_link_libraries(${PROJECT_NAME} ${COCOA_LIBRARY} ${COREVIDEO_LIBRARY} ${IOKIT_LIBRARY})
    else()
        target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES})
     endif()
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIGLFW")

sofa_install_targets(SofaGui SofaGuiGLFW "")
