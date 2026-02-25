cmake_minimum_required(VERSION 3.12)
project(SofaCUDA.GL LANGUAGES CXX CUDA)

set(SOFACUDAGL_SOURCE_DIR "src/SofaCUDA/GL")
set(HEADER_FILES
    ${SOFACUDAGL_SOURCE_DIR}/init.h
    ${SOFACUDAGL_SOURCE_DIR}/config.h.in
    ${SOFACUDAGL_SOURCE_DIR}/CudaVisualModel.h
    ${SOFACUDAGL_SOURCE_DIR}/CudaVisualModel.inl
)

set(SOURCE_FILES
    ${SOFACUDAGL_SOURCE_DIR}/init.cpp
    ${SOFACUDAGL_SOURCE_DIR}/CudaVisualModel.cpp
)

set(CUDA_SOURCES
    ${SOFACUDAGL_SOURCE_DIR}/CudaVisualModel.cu
)

sofa_find_package(SofaCUDA REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES})
target_link_libraries(${PROJECT_NAME} SofaCUDA)
target_link_libraries(${PROJECT_NAME} Sofa.GL)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
